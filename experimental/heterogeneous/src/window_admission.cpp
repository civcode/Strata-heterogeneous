#include "strata_heterogeneous/window_admission.hpp"

#include <cstring>
#include <limits>
#include <stdexcept>

namespace strata::heterogeneous {
namespace {
bool reject(std::string& error, const char* reason) {
    error = reason;
    return false;
}
} // namespace

bool activation_finite(const uint8_t* data, size_t bytes) noexcept {
    if ((!data && bytes) || (bytes % sizeof(float)) != 0 || sizeof(float) != 4) return false;
    for (size_t offset = 0; offset < bytes; offset += sizeof(float)) {
        uint32_t bits = 0;
        std::memcpy(&bits, data + offset, sizeof(bits));
        if ((bits & 0x7f800000u) == 0x7f800000u) return false;
    }
    return true;
}

WindowAdmission::WindowAdmission(const Plan& plan, size_t stage, WindowLimits limits)
    : plan_(plan), stage_(stage), limits_(limits), next_position_(limits.initial_position) {
    plan_.validate();
    if (stage >= plan_.stages.size() || limits.max_cells <= 0 ||
        limits.initial_position < 0 || limits.initial_position >= limits.max_cells ||
        limits.max_t == 0 || limits.max_t > max_window_tokens ||
        limits.n_vocab <= 0 || limits.n_vocab > std::numeric_limits<int32_t>::max())
        throw std::invalid_argument("bounded window limits/stage are invalid");
    const uint64_t stride = plan_.geometry.floats_per_row();
    if (stride == 0 || stride > static_cast<uint64_t>(max_payload_bytes) / sizeof(float))
        throw std::invalid_argument("bounded activation geometry exceeds transport cap");
    row_bytes_ = static_cast<size_t>(stride) * sizeof(float);
    plan_fingerprint_ = plan_.fingerprint();
}

bool WindowAdmission::reset_request(uint64_t request, std::string& error) {
    if (state_ != State::ready || limits_.initial_position != 0 ||
        !request || request <= first_request_)
        return reject(error, "resident reset requires an idle stage and increasing request ID");
    first_request_ = request_ = request;
    window_ = last_window_ = committed_windows_ = 0;
    rows_ = 0;
    pos0_ = next_position_ = 0;
    error.clear();
    return true;
}

bool WindowAdmission::begin(const WindowSpec& spec, uint64_t request,
                            uint64_t window, std::string& error) {
    if (state_ != State::ready || !request || !window ||
        (first_request_ && request != first_request_) ||
        (last_window_ && window <= last_window_))
        return reject(error, "window admission: stage pending, poisoned or request/window replayed");
    try { spec.validate(static_cast<uint32_t>(spec.token_ids.size())); }
    catch (const std::exception&) { return reject(error, "window admission: token/position specification invalid"); }
    const uint64_t rows = spec.token_ids.size();
    if (rows == 0 || rows > limits_.max_t ||
        rows > static_cast<uint64_t>(max_payload_bytes) / row_bytes_ ||
        spec.pos0 != next_position_ ||
        spec.pos0 < 0 || rows > static_cast<uint64_t>(limits_.max_cells - spec.pos0))
        return reject(error, "window admission: position, context, row count or payload cap violated");
    for (int32_t tok : spec.token_ids) {
        if (tok < 0 || tok >= limits_.n_vocab)
            return reject(error, "window admission: token ID outside loaded vocabulary");
    }
    request_ = request;
    window_ = window;
    if (!first_request_) first_request_ = request;
    rows_ = static_cast<uint32_t>(rows);
    pos0_ = spec.pos0;
    state_ = State::begun;
    error.clear();
    return true;
}

bool WindowAdmission::common_frame(const Frame& f) const noexcept {
    return f.stage_index == stage_ && f.request_id == request_ &&
           f.window_id == window_ && f.rows == rows_ &&
           f.plan_fingerprint == plan_fingerprint_ &&
           f.geometry.hc == plan_.geometry.hc &&
           f.geometry.n_embd == plan_.geometry.n_embd &&
           f.identity == std::array<uint32_t, 4>{} &&
           f.error.empty() && !f.window;
}

bool WindowAdmission::validate_input(const Frame& frame, bool shared_memory,
                                     std::string& error) {
    if (state_ != State::begun || !common_frame(frame) || frame.keep_count ||
        frame.kind != (shared_memory ? Kind::shared_data : Kind::data))
        return reject(error, "window admission: mismatched activation frame or missing window_spec");
    if (shared_memory) {
        if (!frame.payload.empty())
            return reject(error, "window admission: shared activation may not carry a socket payload");
    } else {
        if (frame.shared_checksum || frame.payload.size() != static_cast<size_t>(rows_) * (row_bytes_ / sizeof(float)))
            return reject(error, "window admission: socket payload length or checksum mismatch");
        if (!activation_finite(reinterpret_cast<const uint8_t*>(frame.payload.data()),
                               frame.payload.size() * sizeof(float)))
            return reject(error, "window admission: non-finite input activation");
    }
    state_ = State::input_checked;
    error.clear();
    return true;
}

bool WindowAdmission::validate_bytes(const uint8_t* bytes, size_t length,
                                      std::string& error) const {
    if (state_ != State::input_checked || length != static_cast<size_t>(rows_) * row_bytes_ ||
        !activation_finite(bytes, length))
        return reject(error, "window admission: malformed or non-finite shared input bytes");
    error.clear();
    return true;
}

bool WindowAdmission::provisional(const uint8_t* bytes, size_t length,
                                   std::string& error) {
    if (state_ != State::input_checked || length != static_cast<size_t>(rows_) * row_bytes_ ||
        !activation_finite(bytes, length))
        return reject(error, "window admission: incomplete or non-finite stage output");
    state_ = State::provisional;
    error.clear();
    return true;
}

bool WindowAdmission::provisional_final(std::string& error) {
    if (state_ != State::input_checked)
        return reject(error, "window admission: final result was not preceded by a validated input");
    state_ = State::provisional;
    error.clear();
    return true;
}

bool WindowAdmission::check_commit(const Frame& commit, std::string& error) const {
    if (state_ != State::provisional || !common_frame(commit) ||
        commit.kind != Kind::commit_prefix ||
        commit.keep_count == 0 || commit.keep_count > rows_ ||
        !commit.payload.empty() || commit.shared_checksum)
        return reject(error, "window admission: stale or unsupported accepted-prefix commit");
    error.clear();
    return true;
}

bool WindowAdmission::committed(uint32_t keep, std::string& error) {
    if (state_ != State::provisional || !keep || keep > rows_ ||
        keep > static_cast<uint64_t>(limits_.max_cells - next_position_))
        return reject(error, "window admission: commit completion does not match provisional window");
    next_position_ += keep;
    last_window_ = window_;
    ++committed_windows_;
    state_ = State::ready;
    error.clear();
    return true;
}

std::string WindowAdmission::diagnostic(const char* phase) const {
    // No token IDs, model paths, activation data, or expert IDs in logs.
    return std::string("window stage=") + std::to_string(stage_) +
        " request=" + std::to_string(request_) +
        " window=" + std::to_string(window_) +
        " pos0=" + std::to_string(pos0_) +
        " rows=" + std::to_string(rows_) +
        " next=" + std::to_string(next_position_) +
        " committed=" + std::to_string(committed_windows_) +
        " phase=" + (phase ? phase : "unknown");
}

} // namespace strata::heterogeneous
