#include "strata_heterogeneous/final_result.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstring>
#include <limits>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <utility>

namespace strata::heterogeneous {
namespace {
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "final-result wire format requires IEEE754 float32");
constexpr uint32_t magic = 0x52465453; // STFR little-endian
constexpr uint16_t version = 1;
constexpr uint16_t flag_logits = 1;
constexpr size_t checksum_offset = 52;
using Clock = std::chrono::steady_clock;

void u16(std::vector<uint8_t>& out, uint16_t x) {
    out.push_back(static_cast<uint8_t>(x));
    out.push_back(static_cast<uint8_t>(x >> 8));
}
void u32(std::vector<uint8_t>& out, uint32_t x) {
    for (unsigned n = 0; n < 4; ++n) out.push_back(static_cast<uint8_t>(x >> (n * 8)));
}
void u64(std::vector<uint8_t>& out, uint64_t x) {
    for (unsigned n = 0; n < 8; ++n) out.push_back(static_cast<uint8_t>(x >> (n * 8)));
}
uint16_t r16(const uint8_t* p) { return static_cast<uint16_t>(p[0] | (uint16_t(p[1]) << 8)); }
uint32_t r32(const uint8_t* p) {
    uint32_t n = 0;
    for (unsigned i = 0; i < 4; ++i) n |= uint32_t(p[i]) << (i * 8);
    return n;
}
uint64_t r64(const uint8_t* p) {
    uint64_t n = 0;
    for (unsigned i = 0; i < 8; ++i) n |= uint64_t(p[i]) << (i * 8);
    return n;
}
void set_u32(std::vector<uint8_t>& out, size_t at, uint32_t x) {
    for (unsigned i = 0; i < 4; ++i) out.at(at + i) = static_cast<uint8_t>(x >> (i * 8));
}
uint32_t digest(const uint8_t* p, size_t n) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; ++i) {
        if (i >= checksum_offset && i < checksum_offset + 4) continue;
        h ^= p[i]; h *= 16777619u;
    }
    return h;
}
size_t checked_bytes(const FinalStageResult& r) {
    const uint64_t n_tokens = r.rows;
    const uint64_t n_marks = r.nonfinite_rows.size();
    uint64_t bytes = 4 * (n_tokens + 1 + n_marks);
    if (r.logits) {
        const uint64_t values = n_tokens * r.vocab_size;
        bytes += values * sizeof(float);
    }
    if (bytes > final_result_max_payload_bytes)
        throw std::invalid_argument("final result exceeds 16 MiB payload limit");
    return static_cast<size_t>(bytes);
}
void validate_basics(const FinalStageResult& r) {
    if (!r.request_id || !r.window_id || !r.plan_fingerprint || !r.rows ||
        r.rows > final_result_max_rows || !r.vocab_size ||
        r.vocab_size > static_cast<uint32_t>(INT32_MAX))
        throw std::invalid_argument("final result has invalid identity, rows or vocabulary");
    if (r.predicted_tokens.size() != r.rows)
        throw std::invalid_argument("final result token count differs from row count");
    for (int32_t t : r.predicted_tokens)
        if (t < 0 || static_cast<uint32_t>(t) >= r.vocab_size)
            throw std::invalid_argument("final result token is outside vocabulary");
    if (r.nonfinite_rows.size() > r.rows)
        throw std::invalid_argument("too many flagged nonfinite rows");
    uint32_t last = 0;
    bool have_last = false;
    for (uint32_t index : r.nonfinite_rows) {
        if (index >= r.rows || (have_last && index <= last))
            throw std::invalid_argument("nonfinite rows must be sorted, unique and within bounds");
        last = index;
        have_last = true;
    }
    if (r.logits && r.logits->size() != uint64_t(r.rows) * r.vocab_size)
        throw std::invalid_argument("final result logits have wrong matrix shape");
    (void)checked_bytes(r);
}
Clock::time_point deadline(int ms) {
    if (ms < 0) throw std::invalid_argument("negative final result timeout");
    return Clock::now() + std::chrono::milliseconds(ms);
}
void wait_socket(int fd, short expected, Clock::time_point end) {
    for (;;) {
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(end - Clock::now()).count();
        if (remaining < 0) throw std::runtime_error("final result transfer timed out");
        pollfd pfd{fd, expected, 0};
        const int rc = ::poll(&pfd, 1, static_cast<int>(std::min<int64_t>(remaining, INT_MAX)));
        if (rc > 0) {
            if (pfd.revents & POLLNVAL) throw std::runtime_error("invalid final result socket");
            return;
        }
        if (rc == 0) throw std::runtime_error("final result transfer timed out");
        if (errno != EINTR) throw std::runtime_error(std::string("final result poll: ") + std::strerror(errno));
    }
}
void send_all(int fd, const uint8_t* bytes, size_t count, Clock::time_point end) {
    size_t at = 0;
    while (at < count) {
        if (Clock::now() > end) throw std::runtime_error("final result send timed out");
        const auto n = ::send(fd, bytes + at, count - at, MSG_DONTWAIT | MSG_NOSIGNAL);
        if (n > 0) { at += static_cast<size_t>(n); continue; }
        if (n < 0 && errno == EINTR) continue;
        if (n < 0 && (errno == EWOULDBLOCK || errno == EAGAIN)) {
            wait_socket(fd, POLLOUT, end); continue;
        }
        throw std::runtime_error(std::string("final result send: ") +
                                 (n < 0 ? std::strerror(errno) : "closed socket"));
    }
}
bool recv_all(int fd, uint8_t* bytes, size_t count, Clock::time_point end, bool allow_eof) {
    size_t at = 0;
    while (at < count) {
        if (Clock::now() > end) throw std::runtime_error("final result receive timed out");
        const auto n = ::recv(fd, bytes + at, count - at, MSG_DONTWAIT);
        if (n > 0) { at += static_cast<size_t>(n); continue; }
        if (n == 0) {
            if (allow_eof && at == 0) return false;
            throw std::runtime_error("truncated final result frame");
        }
        if (errno == EINTR) continue;
        if (errno == EWOULDBLOCK || errno == EAGAIN) {
            wait_socket(fd, POLLIN, end); continue;
        }
        throw std::runtime_error(std::string("final result recv: ") + std::strerror(errno));
    }
    return true;
}
} // namespace

void validate_final_result(const FinalStageResult& result) { validate_basics(result); }
void require_final_result_identity(const FinalStageResult& result,
                                   uint32_t final_stage_index,
                                   uint64_t request_id, uint64_t window_id,
                                   uint64_t plan_fingerprint, uint32_t rows) {
    validate_final_result(result);
    if (result.stage_index != final_stage_index || result.request_id != request_id ||
        result.window_id != window_id || result.plan_fingerprint != plan_fingerprint ||
        result.rows != rows)
        throw std::runtime_error("final result identity does not match pending verifier window");
}
std::vector<uint8_t> encode_final_result(const FinalStageResult& r) {
    validate_final_result(r);
    std::vector<uint8_t> out;
    out.reserve(final_result_header_bytes + checked_bytes(r));
    u32(out, magic); u16(out, version); u16(out, r.logits ? flag_logits : 0);
    u32(out, r.stage_index); u32(out, r.rows); u32(out, r.vocab_size); u32(out, 0);
    u64(out, r.request_id); u64(out, r.window_id); u64(out, r.plan_fingerprint);
    u32(out, static_cast<uint32_t>(checked_bytes(r))); u32(out, 0); u64(out, 0);
    for (int32_t t : r.predicted_tokens) u32(out, static_cast<uint32_t>(t));
    u32(out, static_cast<uint32_t>(r.nonfinite_rows.size()));
    for (uint32_t index : r.nonfinite_rows) u32(out, index);
    if (r.logits) for (float f : *r.logits) {
        uint32_t bits = 0;
        std::memcpy(&bits, &f, sizeof(bits));
        u32(out, bits);
    }
    set_u32(out, checksum_offset, digest(out.data(), out.size()));
    return out;
}
FinalStageResult decode_final_result(const std::vector<uint8_t>& in) {
    if (in.size() < final_result_header_bytes)
        throw std::invalid_argument("final result wire header truncated");
    const uint8_t* p = in.data();
    if (r32(p) != magic || r16(p + 4) != version || (r16(p + 6) & ~flag_logits) ||
        r32(p + 20) != 0 || r64(p + 56) != 0)
        throw std::invalid_argument("final result magic/version/flags/reserved fields invalid");
    const uint32_t payload = r32(p + 48);
    if (payload > final_result_max_payload_bytes ||
        in.size() != final_result_header_bytes + uint64_t(payload))
        throw std::invalid_argument("final result wire payload size invalid");
    if (digest(in.data(), in.size()) != r32(p + checksum_offset))
        throw std::invalid_argument("final result checksum mismatch");
    FinalStageResult r;
    r.stage_index = r32(p + 8); r.rows = r32(p + 12); r.vocab_size = r32(p + 16);
    r.request_id = r64(p + 24); r.window_id = r64(p + 32);
    r.plan_fingerprint = r64(p + 40);
    if (!r.rows || r.rows > final_result_max_rows || !r.vocab_size ||
        r.vocab_size > static_cast<uint32_t>(INT32_MAX))
        throw std::invalid_argument("final result decoded rows/vocab invalid");
    // Check the expected byte total before allocating anything based on input.
    const uint64_t base = uint64_t(r.rows) * 4 + 4;
    if (payload < base) throw std::invalid_argument("final result token/flag section truncated");
    const uint32_t markers = r32(p + final_result_header_bytes + uint64_t(r.rows) * 4);
    if (markers > r.rows) throw std::invalid_argument("final result nonfinite count invalid");
    const uint64_t expected = base + uint64_t(markers) * 4 +
        ((r16(p + 6) & flag_logits) ? uint64_t(r.rows) * r.vocab_size * 4 : 0);
    if (expected != payload) throw std::invalid_argument("final result decoded shape differs from payload");
    const uint8_t* cursor = p + final_result_header_bytes;
    r.predicted_tokens.reserve(r.rows);
    for (uint32_t i = 0; i < r.rows; ++i, cursor += 4)
        r.predicted_tokens.push_back(static_cast<int32_t>(r32(cursor)));
    cursor += 4;
    r.nonfinite_rows.reserve(markers);
    for (uint32_t i = 0; i < markers; ++i, cursor += 4)
        r.nonfinite_rows.push_back(r32(cursor));
    if (r16(p + 6) & flag_logits) {
        const uint64_t count = uint64_t(r.rows) * r.vocab_size;
        r.logits.emplace(static_cast<size_t>(count));
        for (float& f : *r.logits) {
            const uint32_t bits = r32(cursor);
            std::memcpy(&f, &bits, sizeof(bits)); cursor += 4;
        }
    }
    validate_final_result(r);
    return r;
}
void send_final_result(int fd, const FinalStageResult& result, int timeout_ms) {
    auto data = encode_final_result(result);
    send_all(fd, data.data(), data.size(), deadline(timeout_ms));
}
std::optional<FinalStageResult> receive_final_result(int fd, int timeout_ms) {
    const auto end = deadline(timeout_ms);
    std::vector<uint8_t> bytes(final_result_header_bytes);
    if (!recv_all(fd, bytes.data(), bytes.size(), end, true)) return std::nullopt;
    const uint8_t* header = bytes.data();
    if (r32(header) != magic || r16(header + 4) != version ||
        (r16(header + 6) & ~flag_logits) || r32(header + 20) ||
        r64(header + 56) || !r32(header + 12) ||
        r32(header + 12) > final_result_max_rows || !r32(header + 16) ||
        r32(header + 16) > static_cast<uint32_t>(INT32_MAX))
        throw std::runtime_error("invalid final result wire header");
    const uint32_t payload = r32(header + 48);
    if (payload > final_result_max_payload_bytes)
        throw std::runtime_error("final result wire payload exceeds capacity");
    bytes.resize(final_result_header_bytes + payload);
    if (payload) (void)recv_all(fd, bytes.data() + final_result_header_bytes, payload, end, false);
    return decode_final_result(bytes);
}
void FinalResultGate::stage(FinalStageResult result, uint32_t final_stage_index,
                            uint64_t request_id, uint64_t window_id,
                            uint64_t plan_fingerprint, uint32_t rows) {
    if (poisoned_ || staged_) throw std::logic_error("final result gate has pending or failed state");
    try {
        require_final_result_identity(result, final_stage_index, request_id,
                                      window_id, plan_fingerprint, rows);
    } catch (...) { poison(); throw; }
    staged_ = std::move(result);
    committed_ = false;
}
const FinalStageResult& FinalResultGate::provisional() const {
    if (poisoned_ || !staged_ || committed_)
        throw std::logic_error("no provisional final result available");
    return *staged_;
}
void FinalResultGate::mark_committed(uint64_t request_id, uint64_t window_id) {
    if (poisoned_ || !staged_ || committed_) throw std::logic_error("no pending result to commit");
    if (staged_->request_id != request_id || staged_->window_id != window_id) {
        poison();
        throw std::runtime_error("final result commit identity mismatch");
    }
    if (!staged_->nonfinite_rows.empty()) {
        poison();
        throw std::runtime_error("nonfinite final head result cannot be committed");
    }
    committed_ = true;
}
FinalStageResult FinalResultGate::take_committed() {
    if (poisoned_ || !staged_ || !committed_)
        throw std::logic_error("cannot publish an uncommitted final result");
    FinalStageResult result = std::move(*staged_);
    staged_.reset();
    committed_ = false;
    return result;
}
void FinalResultGate::poison() noexcept {
    poisoned_ = true;
    committed_ = false;
    staged_.reset();
}
} // namespace strata::heterogeneous
