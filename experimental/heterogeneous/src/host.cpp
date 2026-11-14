#include "strata_heterogeneous/host.hpp"

#include <algorithm>
#include <chrono>
#include <poll.h>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>
#include <sstream>
#include <sys/socket.h>
#include <unistd.h>
#include <utility>

namespace strata::heterogeneous {
void WindowSpec::validate(uint32_t rows) const {
    if (!rows || rows > max_window_tokens || token_ids.size() != rows || pos0 < 0 ||
        static_cast<uint64_t>(pos0) > static_cast<uint64_t>(INT64_MAX) - (rows - 1))
        throw std::invalid_argument("invalid token window count or position");
    for (int32_t token : token_ids)
        if (token < 0) throw std::invalid_argument("negative token id in window");
}
namespace {
constexpr uint32_t magic = 0x58485453;  // STHX in little endian
constexpr uint16_t version = 4;
constexpr size_t max_failure_bytes = 1024;

uint64_t checked_add(uint64_t a, uint64_t b) {
    if (b > std::numeric_limits<uint64_t>::max() - a) throw std::overflow_error("handoff geometry overflow");
    return a + b;
}
uint64_t checked_mul(uint64_t a, uint64_t b) {
    if (a && b > std::numeric_limits<uint64_t>::max() / a) throw std::overflow_error("handoff geometry overflow");
    return a * b;
}
uint32_t number(const std::string& s) {
    if (s.empty()) throw std::invalid_argument("empty unsigned integer");
    uint32_t value = 0;
    auto res = std::from_chars(s.data(), s.data() + s.size(), value);
    if (res.ec != std::errc() || res.ptr != s.data() + s.size())
        throw std::invalid_argument("invalid unsigned integer: " + s);
    return value;
}
void add_u16(std::vector<uint8_t>& out, uint16_t x) {
    out.push_back(static_cast<uint8_t>(x)); out.push_back(static_cast<uint8_t>(x >> 8));
}
void add_u32(std::vector<uint8_t>& out, uint32_t x) {
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<uint8_t>(x >> (8 * i)));
}
void add_u64(std::vector<uint8_t>& out, uint64_t x) {
    for (int i = 0; i < 8; ++i) out.push_back(static_cast<uint8_t>(x >> (8 * i)));
}
uint16_t get_u16(const uint8_t* p) { return static_cast<uint16_t>(p[0] | (uint16_t(p[1]) << 8)); }
uint32_t get_u32(const uint8_t* p) {
    uint32_t v = 0; for (int i = 0; i < 4; ++i) v |= uint32_t(p[i]) << (8 * i); return v;
}
uint64_t get_u64(const uint8_t* p) {
    uint64_t v = 0; for (int i = 0; i < 8; ++i) v |= uint64_t(p[i]) << (8 * i); return v;
}
void mix_u64(uint64_t& h, uint64_t v) {
    for (int i = 0; i < 8; ++i) { h ^= static_cast<uint8_t>(v >> (8 * i)); h *= 1099511628211ull; }
}
uint32_t checksum(const uint8_t* p, size_t n) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 16777619u; }
    return h;  // corruption detection, not a cryptographic authenticator
}
void check_host() {
    const uint32_t x = 1;
    if (sizeof(float) != 4 || !std::numeric_limits<float>::is_iec559 ||
        *reinterpret_cast<const uint8_t*>(&x) != 1)
        throw std::runtime_error("wire protocol requires IEEE754 float32 and a little-endian host");
}
uint32_t expected_bytes(const Frame& f) {
    const uint64_t n = checked_mul(checked_mul(f.rows, f.geometry.floats_per_row()), sizeof(float));
    if (n > max_payload_bytes) throw std::invalid_argument("handoff exceeds 16 MiB frame cap; use smaller windows");
    return static_cast<uint32_t>(n);
}
void validate_frame(const Frame& f) {
    if (f.kind != Kind::window_spec && f.window)
        throw std::invalid_argument("unexpected window specification");
    if (f.kind != Kind::commit_prefix && f.kind != Kind::committed_prefix && f.keep_count)
        throw std::invalid_argument("unexpected accepted-token count");
    if (!f.request_id || !f.plan_fingerprint || !f.rows || !f.geometry.hc || !f.geometry.n_embd)
        throw std::invalid_argument("frame has unset request, plan, rows, or geometry");
    const bool zero_identity = f.identity == std::array<uint32_t, 4>{};
    switch (f.kind) {
        case Kind::data:
            if (f.shared_checksum || !zero_identity || !f.error.empty() ||
                checked_mul(f.payload.size(), sizeof(float)) != expected_bytes(f))
                throw std::invalid_argument("data frame has wrong handoff shape");
            break;
        case Kind::shared_data:
            if (!zero_identity || !f.payload.empty() || !f.error.empty())
                throw std::invalid_argument("shared_data must carry only the mapped-buffer checksum");
            (void) expected_bytes(f);
            break;
        case Kind::window_spec:
            if (!zero_identity || f.shared_checksum || !f.payload.empty() ||
                !f.error.empty() || !f.window)
                throw std::invalid_argument("invalid window specification frame");
            f.window->validate(f.rows);
            break;
        case Kind::commit_prefix:
        case Kind::committed_prefix:
            if (!zero_identity || f.shared_checksum || !f.payload.empty() ||
                !f.error.empty() || f.keep_count > f.rows)
                throw std::invalid_argument("invalid accepted-prefix control");
            break;
        case Kind::failure:
            if (f.shared_checksum || !zero_identity || !f.payload.empty() || f.error.empty() || f.error.size() > max_failure_bytes)
                throw std::invalid_argument("invalid failure frame");
            break;
        case Kind::hello:
        case Kind::hello_shared:
        case Kind::hello_final:
        case Kind::hello_shared_final:
            if (f.shared_checksum || !f.payload.empty() || !f.error.empty() ||
                (f.identity[0] != static_cast<uint32_t>(Backend::cuda) &&
                 f.identity[0] != static_cast<uint32_t>(Backend::hip)) ||
                f.identity[2] >= f.identity[3] || f.rows != 1 ||
                f.request_id != 1 || f.window_id != 1)
                throw std::invalid_argument("invalid worker hello");
            break;
        case Kind::ready:
        case Kind::ready_resident:
            if (f.shared_checksum || f.rows != 1 || f.request_id != 1 || f.window_id != 1 ||
                !zero_identity || !f.payload.empty() || !f.error.empty())
                throw std::invalid_argument("invalid worker ready");
            break;
        case Kind::begin_request:
        case Kind::request_ready:
            if (f.shared_checksum || !zero_identity || !f.payload.empty() ||
                !f.error.empty() || f.rows != 1 || f.window_id != 0)
                throw std::invalid_argument("invalid resident request control");
            break;
        case Kind::commit:
        case Kind::committed:
            if (f.shared_checksum || !zero_identity || !f.payload.empty() || !f.error.empty())
                throw std::invalid_argument("invalid window control frame");
            break;
        default: throw std::invalid_argument("unsupported frame kind");
    }
}
using Deadline = std::chrono::steady_clock::time_point;
Deadline deadline_after(int timeout_ms) {
    if (timeout_ms < 0) throw std::invalid_argument("negative handoff timeout");
    return std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
}
void wait_ready(int fd, short events, Deadline deadline) {
    while (true) {
        auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline -
                            std::chrono::steady_clock::now()).count();
        if (left < 0) throw std::runtime_error("handoff timed out");
        struct pollfd pfd {fd, events, 0};
        int n = ::poll(&pfd, 1, static_cast<int>(std::min<int64_t>(left, INT32_MAX)));
        if (n > 0) {
            if (pfd.revents & POLLNVAL) throw std::runtime_error("handoff invalid socket");
            return; // POLLHUP/POLLERR are handled by recv/send
        }
        if (n == 0) throw std::runtime_error("handoff timed out");
        if (errno != EINTR) throw std::runtime_error(std::string("handoff poll: ") + std::strerror(errno));
    }
}
bool read_full(int fd, uint8_t* p, size_t n, Deadline deadline, bool eof_allowed = false) {
    size_t at = 0;
    while (at < n) {
        if (std::chrono::steady_clock::now() > deadline)
            throw std::runtime_error("handoff timed out");
        const ssize_t got = ::recv(fd, p + at, n - at, MSG_DONTWAIT);
        if (got > 0) { at += static_cast<size_t>(got); continue; }
        if (got == 0) {
            if (at == 0 && eof_allowed) return false;
            throw std::runtime_error("truncated handoff frame");
        }
        if (errno == EINTR) continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK) { wait_ready(fd, POLLIN, deadline); continue; }
        throw std::runtime_error(std::string("handoff read: ") + std::strerror(errno));
    }
    return true;
}
void write_full(int fd, const uint8_t* p, size_t n, Deadline deadline) {
    size_t at = 0;
    while (at < n) {
        if (std::chrono::steady_clock::now() > deadline)
            throw std::runtime_error("handoff timed out");
        const ssize_t sent = ::send(fd, p + at, n - at, MSG_NOSIGNAL | MSG_DONTWAIT);
        if (sent > 0) { at += static_cast<size_t>(sent); continue; }
        if (sent < 0 && errno == EINTR) continue;
        if (sent < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            wait_ready(fd, POLLOUT, deadline); continue;
        }
        throw std::runtime_error(std::string("handoff send: ") +
                                 (sent < 0 ? std::strerror(errno) : "zero write"));
    }
}
} // namespace

uint64_t Geometry::floats_per_row() const {
    if (!hc || !n_embd) throw std::invalid_argument("empty handoff geometry");
    return checked_add(checked_add(checked_mul(hc, n_embd), n_embd), hc);
}
void Plan::validate() const {
    if (!model_fingerprint || !total_layers || stages.empty()) throw std::invalid_argument("incomplete stage plan");
    (void) geometry.floats_per_row();
    uint32_t cursor = 0;
    std::set<std::pair<int, uint32_t>> used;
    for (const auto& s : stages) {
        if (s.backend != Backend::cuda && s.backend != Backend::hip) throw std::invalid_argument("unknown backend");
        if (s.first_layer != cursor || s.end_layer <= cursor || s.end_layer > total_layers)
            throw std::invalid_argument("layer stages must exactly partition [0,total_layers)");
        if (!used.emplace(static_cast<int>(s.backend), s.ordinal).second)
            throw std::invalid_argument("device appears more than once in stage plan");
        cursor = s.end_layer;
    }
    if (cursor != total_layers) throw std::invalid_argument("last stage does not cover final layer");
}
uint64_t Plan::fingerprint() const {
    validate();
    uint64_t h = 14695981039346656037ull;
    mix_u64(h, 1); mix_u64(h, model_fingerprint); mix_u64(h, total_layers);
    mix_u64(h, geometry.hc); mix_u64(h, geometry.n_embd); mix_u64(h, stages.size());
    for (const auto& s : stages) {
        mix_u64(h, static_cast<int>(s.backend)); mix_u64(h, s.ordinal);
        mix_u64(h, s.first_layer); mix_u64(h, s.end_layer);
    }
    return h ? h : 1; // reserved zero means uninitialized
}
Plan parse_plan(const std::string& spec, uint32_t layers, Geometry geometry, uint64_t model_fingerprint) {
    Plan out{layers, model_fingerprint, geometry, {}};
    size_t at = 0;
    while (at < spec.size()) {
        const size_t end = spec.find(',', at);
        const std::string part = spec.substr(at, end == std::string::npos ? end : end - at);
        const size_t a = part.find(':'), b = a == std::string::npos ? a : part.find(':', a + 1);
        if (a == std::string::npos || b == std::string::npos || part.find(':', b + 1) != std::string::npos)
            throw std::invalid_argument("expected backend:ordinal:start-end");
        const std::string backend = part.substr(0, a);
        const size_t dash = part.find('-', b + 1);
        if (dash == std::string::npos || part.find('-', dash + 1) != std::string::npos)
            throw std::invalid_argument("expected start-end layer range");
        Stage s;
        if (backend == "cuda") s.backend = Backend::cuda;
        else if (backend == "hip") s.backend = Backend::hip;
        else throw std::invalid_argument("unknown backend: " + backend);
        s.ordinal = number(part.substr(a + 1, b - a - 1));
        s.first_layer = number(part.substr(b + 1, dash - b - 1));
        s.end_layer = number(part.substr(dash + 1));
        out.stages.push_back(s);
        if (end == std::string::npos) break;
        at = end + 1;
        if (at == spec.size()) throw std::invalid_argument("trailing comma in stage plan");
    }
    out.validate();
    return out;
}
void validate_wire_frame(const Frame& f) { check_host(); validate_frame(f); }
std::vector<uint8_t> encode(const Frame& f) {
    validate_wire_frame(f);
    std::vector<uint8_t> payload;
    if (f.kind == Kind::data) {
        payload.resize(f.payload.size() * sizeof(float));
        std::memcpy(payload.data(), f.payload.data(), payload.size());
    } else if (f.kind == Kind::failure) payload.assign(f.error.begin(), f.error.end());
    else if (f.kind == Kind::hello || f.kind == Kind::hello_shared ||
             f.kind == Kind::hello_final || f.kind == Kind::hello_shared_final)
        for (uint32_t word : f.identity) add_u32(payload, word);
    else if (f.kind == Kind::shared_data) add_u32(payload, f.shared_checksum);
    else if (f.kind == Kind::window_spec) {
        add_u64(payload, static_cast<uint64_t>(f.window->pos0));
        add_u32(payload, static_cast<uint32_t>(f.window->token_ids.size()));
        for (int32_t token : f.window->token_ids) add_u32(payload, static_cast<uint32_t>(token));
    } else if (f.kind == Kind::commit_prefix || f.kind == Kind::committed_prefix)
        add_u32(payload, f.keep_count);
    std::vector<uint8_t> bytes;
    bytes.reserve(wire_header_bytes + payload.size());
    add_u32(bytes, magic); add_u16(bytes, version); add_u16(bytes, static_cast<uint16_t>(f.kind));
    add_u32(bytes, f.stage_index); add_u32(bytes, f.rows);
    add_u32(bytes, f.geometry.hc); add_u32(bytes, f.geometry.n_embd);
    add_u64(bytes, f.request_id); add_u64(bytes, f.window_id); add_u64(bytes, f.plan_fingerprint);
    add_u32(bytes, static_cast<uint32_t>(payload.size())); add_u32(bytes, checksum(payload.data(), payload.size()));
    if (bytes.size() != wire_header_bytes) throw std::logic_error("wire header size mismatch");
    bytes.insert(bytes.end(), payload.begin(), payload.end());
    return bytes;
}
Frame decode(const std::vector<uint8_t>& bytes) {
    check_host();
    if (bytes.size() < wire_header_bytes || get_u32(bytes.data()) != magic || get_u16(bytes.data() + 4) != version)
        throw std::invalid_argument("unsupported handoff header");
    const uint32_t n = get_u32(bytes.data() + 48);
    if (n > max_payload_bytes || n != bytes.size() - wire_header_bytes)
        throw std::invalid_argument("bad handoff payload length");
    const uint8_t* payload = bytes.data() + wire_header_bytes;
    if (checksum(payload, n) != get_u32(bytes.data() + 52)) throw std::invalid_argument("handoff checksum mismatch");
    Frame f;
    f.kind = static_cast<Kind>(get_u16(bytes.data() + 6));
    f.stage_index = get_u32(bytes.data() + 8);
    f.rows = get_u32(bytes.data() + 12);
    f.geometry = {get_u32(bytes.data() + 16), get_u32(bytes.data() + 20)};
    f.request_id = get_u64(bytes.data() + 24);
    f.window_id = get_u64(bytes.data() + 32);
    f.plan_fingerprint = get_u64(bytes.data() + 40);
    if (f.kind == Kind::data) {
        if (n % sizeof(float) != 0) throw std::invalid_argument("partial float in handoff");
        f.payload.resize(n / sizeof(float));
        if (n) std::memcpy(f.payload.data(), payload, n);
    } else if (f.kind == Kind::failure) f.error.assign(reinterpret_cast<const char*>(payload), n);
    else if (f.kind == Kind::window_spec) {
        if (n < 12 || (n - 12) % 4 != 0 || get_u32(payload + 8) != (n - 12) / 4 ||
            get_u32(payload + 8) > max_window_tokens)
            throw std::invalid_argument("invalid token window wire length");
        WindowSpec w;
        w.pos0 = static_cast<int64_t>(get_u64(payload));
        const uint32_t count = get_u32(payload + 8);
        w.token_ids.reserve(count);
        for (uint32_t i = 0; i < count; ++i)
            w.token_ids.push_back(static_cast<int32_t>(get_u32(payload + 12 + 4 * i)));
        f.window = std::move(w);
    } else if (f.kind == Kind::commit_prefix || f.kind == Kind::committed_prefix) {
        if (n != 4) throw std::invalid_argument("invalid accepted-prefix wire length");
        f.keep_count = get_u32(payload);
    } else if (f.kind == Kind::shared_data) {
        if (n != 4) throw std::invalid_argument("shared handoff digest has wrong length");
        f.shared_checksum = get_u32(payload);
    } else if (f.kind == Kind::hello || f.kind == Kind::hello_shared ||
               f.kind == Kind::hello_final || f.kind == Kind::hello_shared_final) {
        if (n != 16) throw std::invalid_argument("hello identity has wrong length");
        for (size_t i = 0; i < 4; ++i) f.identity[i] = get_u32(payload + 4 * i);
    }
    // Empty-control kinds must not smuggle unrecognized payload bytes.
    if ((f.kind == Kind::ready_resident || f.kind == Kind::begin_request ||
         f.kind == Kind::request_ready) && n != 0)
        throw std::invalid_argument("resident control payload must be empty");
    validate_frame(f);
    return f;
}
void send_frame(int fd, const Frame& frame, int timeout_ms) {
    const auto deadline = deadline_after(timeout_ms);
    auto bytes = encode(frame);
    write_full(fd, bytes.data(), bytes.size(), deadline);
}
std::optional<Frame> receive_frame(int fd, int timeout_ms) {
    const auto deadline = deadline_after(timeout_ms);
    uint8_t header[wire_header_bytes]{};
    if (!read_full(fd, header, wire_header_bytes, deadline, true)) return std::nullopt;
    const uint32_t n = get_u32(header + 48);
    if (n > max_payload_bytes) throw std::invalid_argument("oversized incoming handoff frame");
    std::vector<uint8_t> bytes(header, header + wire_header_bytes);
    bytes.resize(wire_header_bytes + n);
    if (n) read_full(fd, bytes.data() + wire_header_bytes, n, deadline);
    return decode(bytes);
}
std::optional<Frame> receive_idle_frame(int fd, int timeout_ms) {
    pollfd p{fd, POLLIN, 0};
    for (;;) {
        const int rc = ::poll(&p, 1, -1);
        if (rc < 0 && errno == EINTR) continue;
        if (rc < 0 || (p.revents & POLLNVAL))
            throw std::runtime_error("resident worker idle poll failed");
        if (rc > 0) return receive_frame(fd, timeout_ms);
    }
}
bool mock_execute(const Stage& stage, const Frame& input, Frame& output, std::string&) {
    output = input;
    // Every layer applies its own small deterministic transformation. Independent
    // CPU reference uses all layers in the same order; this tests no layer is lost.
    for (uint32_t l = stage.first_layer; l < stage.end_layer; ++l)
        for (float& f : output.payload) f += static_cast<float>(l + 1) * 0.0001f;
    return true;
}
} // namespace strata::heterogeneous
