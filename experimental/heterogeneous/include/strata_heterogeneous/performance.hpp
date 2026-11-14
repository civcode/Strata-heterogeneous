#pragma once

#include <chrono>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace strata::heterogeneous {

// Strict, process-local opt-ins; no change to homogeneous engine defaults.
inline bool performance_flag(const char* name) {
    const char* value = std::getenv(name);
    if (!value || std::strcmp(value, "0") == 0) return false;
    if (std::strcmp(value, "1") == 0) return true;
    throw std::invalid_argument(std::string(name) + " must be 0 or 1");
}
// Set by the native coordinator; leave ordinary worker frame/GPU timeouts intact.
inline int worker_startup_wait_ms() {
    const char* value = std::getenv("STRATA_HETERO_STARTUP_WAIT_MS");
    if (!value) return -1;
    int timeout = 0;
    const char* end = value + std::strlen(value);
    auto parsed = std::from_chars(value, end, timeout);
    if (parsed.ec != std::errc{} || parsed.ptr != end || timeout < 100 || timeout > 600000)
        throw std::invalid_argument("STRATA_HETERO_STARTUP_WAIT_MS must be 100..600000");
    return timeout;
}
inline bool performance_trace() noexcept {
    static const bool enabled = [] {
        const char* v = std::getenv("STRATA_HETERO_PERF_TRACE");
        return v && std::strcmp(v, "1") == 0;
    }();
    return enabled;
}
inline std::chrono::microseconds fence_poll_interval() noexcept {
    static const bool fast = [] {
        const char* v = std::getenv("STRATA_HETERO_FAST_FENCE");
        return v && std::strcmp(v, "1") == 0;
    }();
    return std::chrono::microseconds(fast ? 50 : 1000);
}

// Wall-clock spans are opt-in; a trace is diagnostic, not an uninstrumented
// speed measurement. A destructor reports elapsed time even on failure.
class PerformanceSpan {
    using Clock = std::chrono::steady_clock;
    const char* component_;
    const char* phase_;
    size_t stage_;
    uint64_t request_, window_;
    size_t rows_;
    bool enabled_;
    Clock::time_point start_;
public:
    PerformanceSpan(const char* component, const char* phase, size_t stage,
                    uint64_t request, uint64_t window, size_t rows) noexcept
        : component_(component), phase_(phase), stage_(stage), request_(request),
          window_(window), rows_(rows), enabled_(performance_trace()),
          start_(enabled_ ? Clock::now() : Clock::time_point{}) {}
    PerformanceSpan(const PerformanceSpan&) = delete;
    PerformanceSpan& operator=(const PerformanceSpan&) = delete;
    ~PerformanceSpan() noexcept {
        if (!enabled_) return;
        const auto end = Clock::now();
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
        const auto start_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(start_.time_since_epoch()).count();
        const auto end_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end.time_since_epoch()).count();
        std::fprintf(stderr, "HETERO_PERF component=%s phase=%s stage=%zu request=%llu "
                     "window=%llu rows=%zu us=%lld start_ns=%lld end_ns=%lld\n", component_, phase_, stage_,
                     static_cast<unsigned long long>(request_),
                     static_cast<unsigned long long>(window_), rows_,
                     static_cast<long long>(us), static_cast<long long>(start_ns), static_cast<long long>(end_ns));
    }
};

} // namespace strata::heterogeneous
