#include "strata_heterogeneous/host.hpp"

#include <algorithm>
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
namespace {
constexpr uint32_t magic = 0x58485453;  // STHX in little endian
constexpr uint16_t version = 1;
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
    if (!f.request_id || !f.plan_fingerprint || !f.rows || !f.geometry.hc || !f.geometry.n_embd)
        throw std::invalid_argument("frame has unset request, plan, rows, or geometry");
    if (f.kind == Kind::data) {
        if (!f.error.empty() || checked_mul(f.payload.size(), sizeof(float)) != expected_bytes(f))
            throw std::invalid_argument("data frame has wrong handoff shape");
    } else if (f.kind == Kind::failure) {
        if (!f.payload.empty() || f.error.empty() || f.error.size() > max_failure_bytes)
            throw std::invalid_argument("invalid failure frame");
    } else throw std::invalid_argument("unsupported frame kind");
}
void read_full(int fd, uint8_t* p, size_t n) {
    size_t at = 0;
    while (at < n) {
        const ssize_t got = ::read(fd, p + at, n - at);
        if (got > 0) { at += static_cast<size_t>(got); continue; }
        if (got == 0) {
            throw std::runtime_error("truncated handoff frame");
        }
        if (errno == EINTR) continue;
        throw std::runtime_error(std::string("handoff read: ") + std::strerror(errno));
    }
}
void write_full(int fd, const uint8_t* p, size_t n) {
    size_t at = 0;
    while (at < n) {
        // MSG_NOSIGNAL prevents an accidental SIGPIPE when a GPU worker dies.
        const ssize_t sent = ::send(fd, p + at, n - at, MSG_NOSIGNAL);
        if (sent > 0) { at += static_cast<size_t>(sent); continue; }
        if (sent < 0 && errno == EINTR) continue;
        throw std::runtime_error(std::string("handoff send: ") + (sent < 0 ? std::strerror(errno) : "zero write"));
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
std::vector<uint8_t> encode(const Frame& f) {
    check_host(); validate_frame(f);
    std::vector<uint8_t> payload;
    if (f.kind == Kind::data) {
        payload.resize(f.payload.size() * sizeof(float));
        std::memcpy(payload.data(), f.payload.data(), payload.size());
    } else payload.assign(f.error.begin(), f.error.end());
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
    validate_frame(f);
    return f;
}
void send_frame(int fd, const Frame& frame) {
    auto data = encode(frame);
    write_full(fd, data.data(), data.size());
}
std::optional<Frame> receive_frame(int fd) {
    uint8_t header[wire_header_bytes]{};
    // Must distinguish an orderly peer close from a truncated header.
    ssize_t first;
    do { first = ::read(fd, header, 1); } while (first < 0 && errno == EINTR);
    if (first == 0) return std::nullopt;
    if (first < 0) throw std::runtime_error(std::string("handoff read: ") + std::strerror(errno));
    read_full(fd, header + 1, wire_header_bytes - 1);
    const uint32_t n = get_u32(header + 48);
    if (n > max_payload_bytes) throw std::invalid_argument("oversized incoming handoff frame");
    std::vector<uint8_t> bytes(header, header + wire_header_bytes);
    bytes.resize(wire_header_bytes + n);
    if (n) read_full(fd, bytes.data() + wire_header_bytes, n);
    return decode(bytes);
}
void serve_worker(int fd, const Plan& plan, size_t index, const Executor& execute) {
    plan.validate();
    if (index >= plan.stages.size()) throw std::invalid_argument("worker index out of range");
    const uint64_t fingerprint = plan.fingerprint();
    while (auto incoming = receive_frame(fd)) {
        Frame output = *incoming;
        output.kind = Kind::data;
        output.payload.clear(); output.error.clear();
        std::string error;
        try {
            if (incoming->kind != Kind::data || incoming->stage_index != index ||
                incoming->plan_fingerprint != fingerprint || incoming->geometry.hc != plan.geometry.hc ||
                incoming->geometry.n_embd != plan.geometry.n_embd)
                throw std::runtime_error("worker plan/stage/geometry mismatch");
            if (!execute(plan.stages[index], *incoming, output, error))
                throw std::runtime_error(error.empty() ? "worker execution failed" : error);
            if (output.kind != Kind::data || output.request_id != incoming->request_id ||
                output.window_id != incoming->window_id || output.stage_index != incoming->stage_index ||
                output.plan_fingerprint != incoming->plan_fingerprint || output.rows != incoming->rows ||
                output.geometry.hc != incoming->geometry.hc || output.geometry.n_embd != incoming->geometry.n_embd)
                throw std::runtime_error("worker changed immutable handoff metadata");
            // send_frame performs final output-shape validation before publication.
            send_frame(fd, output);
        } catch (const std::exception& ex) {
            Frame failure = *incoming;
            failure.kind = Kind::failure; failure.payload.clear();
            failure.error.assign(ex.what(), std::min(std::strlen(ex.what()), max_failure_bytes));
            send_frame(fd, failure);
            return; // fail-stop; a real worker should poison its session after a runtime error
        }
    }
}
Frame run_pipeline(const Plan& plan, const std::vector<int>& fds, Frame initial) {
    plan.validate();
    if (fds.size() != plan.stages.size()) throw std::invalid_argument("one worker fd required per stage");
    const uint64_t fingerprint = plan.fingerprint();
    if (initial.kind != Kind::data || initial.plan_fingerprint != fingerprint ||
        initial.geometry.hc != plan.geometry.hc || initial.geometry.n_embd != plan.geometry.n_embd)
        throw std::invalid_argument("initial frame doesn't match plan");
    for (size_t i = 0; i < fds.size(); ++i) {
        initial.stage_index = static_cast<uint32_t>(i);
        send_frame(fds[i], initial);
        auto answer = receive_frame(fds[i]);
        if (!answer) throw std::runtime_error("worker closed before replying");
        if (answer->request_id != initial.request_id || answer->window_id != initial.window_id ||
            answer->stage_index != initial.stage_index || answer->plan_fingerprint != fingerprint ||
            answer->rows != initial.rows || answer->geometry.hc != initial.geometry.hc ||
            answer->geometry.n_embd != initial.geometry.n_embd)
            throw std::runtime_error("stale or mismatched worker reply");
        if (answer->kind == Kind::failure)
            throw std::runtime_error("stage " + std::to_string(i) + " failed: " + answer->error);
        initial = std::move(*answer);
    }
    return initial; // only final-stage success is published
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
