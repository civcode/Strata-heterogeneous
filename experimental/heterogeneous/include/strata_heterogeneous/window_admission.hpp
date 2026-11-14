#pragma once

// M1.18 — CPU-only, single-window admission and publication gate.
// This file has no CUDA/HIP dependencies; only the supervised worker opts in.
#include "strata_heterogeneous/host.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::heterogeneous {

struct WindowLimits {
    int64_t initial_position = 0;
    int64_t max_cells = 0;
    int64_t n_vocab = 0;
    uint32_t max_t = 0;
};

// The normal v4 protocol remains compatible with legacy mocks. When given
// WindowLimits, the *real* worker server requires window_spec and a positive
// accepted-prefix commit for EVERY window. No durable position moves before
// the GPU verifier's successful commit/wait and the stage's commit ACK.
class WindowAdmission final {
public:
    WindowAdmission(const Plan& plan, size_t stage_index, WindowLimits limits);

    // Admission is pure host logic; no CUDA work or model pointers involved.
    bool reset_request(uint64_t request, std::string& error);
    bool begin(const WindowSpec& spec, uint64_t request, uint64_t window,
               std::string& error);
    bool validate_input(const Frame& frame, bool shared_memory, std::string& error);
    bool validate_bytes(const uint8_t* bytes, size_t length, std::string& error) const;
    // Accept only finished, finite host-visible output. Shared bytes are
    // never interpreted as constructed float objects (memcpy uint32_t bits).
    bool provisional(const uint8_t* bytes, size_t length, std::string& error);
    bool provisional_final(std::string& error); // after validated STFR result, not an activation
    bool check_commit(const Frame& commit, std::string& error) const;
    bool committed(uint32_t keep, std::string& error);
    void poison() noexcept { state_ = State::dead; }

    bool pending() const noexcept { return state_ == State::begun || state_ == State::input_checked || state_ == State::provisional; }
    bool poisoned() const noexcept { return state_ == State::dead; }
    int64_t next_position() const noexcept { return next_position_; }
    uint64_t committed_windows() const noexcept { return committed_windows_; }
    std::string diagnostic(const char* phase) const; // metadata only, never tokens or paths

private:
    enum class State { ready, begun, input_checked, provisional, dead };
    Plan plan_;
    size_t stage_ = 0;
    WindowLimits limits_;
    State state_ = State::ready;
    uint64_t plan_fingerprint_ = 0;
    uint64_t request_ = 0;
    uint64_t window_ = 0;
    uint64_t first_request_ = 0;
    uint64_t last_window_ = 0;
    uint64_t committed_windows_ = 0;
    uint32_t rows_ = 0;
    int64_t pos0_ = 0;
    int64_t next_position_ = 0;
    size_t row_bytes_ = 0;
    bool common_frame(const Frame& f) const noexcept;
};

// A no-throw, byte-wise non-finite check shared by socket and memfd paths.
// Ignores the platform's fast-math settings and C++17 mmap float lifetime.
bool activation_finite(const uint8_t* data, size_t bytes) noexcept;

} // namespace strata::heterogeneous
