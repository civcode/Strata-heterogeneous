#pragma once

#include "strata/core/transfer_terminal.hpp"

#include <cuda_runtime.h>
#include <string>

namespace strata::core::detail {

// Some runtimes can accept GPU work and still report an issue error. In that
// case the caller's blob, HIP pinned staging, or a D2H readback buffer may be
// referenced until a positive stream completion. Do not unwind them on an
// unknown completion status.
inline void expert_cache_drain_or_exit(cudaStream_t stream, const char* operation) noexcept {
    const cudaError_t completed = cudaStreamSynchronize(stream);
    if (completed != cudaSuccess)
        core::transfer_terminal(operation, cudaGetErrorString(completed));
}

// Preserve the original failure for the caller, but establish completion first.
// Used both after issue errors and after reported sync errors (a second wait may
// prove completion even though the first wait failed).
inline bool expert_cache_status(cudaError_t status, cudaStream_t stream,
                                std::string& error, const char* operation) {
    if (status == cudaSuccess) return true;
    // Drain before building an error string: allocation can itself throw,
    // and an exception must not unwind a still-borrowed source buffer.
    expert_cache_drain_or_exit(stream, operation);
    error = std::string(operation) + ": " + cudaGetErrorString(status);
    return false;
}

// hipDeviceSynchronize/cudaDeviceSynchronize only fence the current device.
// A cache can outlive a caller's OnDevice scope (notably a failed GpuStage
// startup). Keep the owning device selected for the entire fence AND release,
// then restore the calling thread's original selection. Never free from a
// different device or unwind owners if device selection fails.
class ExpertCacheCloseDevice {
public:
    ExpertCacheCloseDevice(int owner, bool has_storage) noexcept {
        if (!has_storage) return;
        if (owner < 0)
            core::transfer_terminal("ExpertCache::close device", "cache allocation device is unknown");
        const cudaError_t got = cudaGetDevice(&previous_);
        if (got != cudaSuccess)
            core::transfer_terminal("ExpertCache::close get device", cudaGetErrorString(got));
        if (previous_ != owner) {
            const cudaError_t selected = cudaSetDevice(owner);
            if (selected != cudaSuccess)
                core::transfer_terminal("ExpertCache::close select device", cudaGetErrorString(selected));
            switched_ = true;
        }
    }
    ~ExpertCacheCloseDevice() noexcept {
        if (switched_) {
            const cudaError_t restored = cudaSetDevice(previous_);
            if (restored != cudaSuccess)
                core::transfer_terminal("ExpertCache::close restore device", cudaGetErrorString(restored));
        }
    }
    ExpertCacheCloseDevice(const ExpertCacheCloseDevice&) = delete;
    ExpertCacheCloseDevice& operator=(const ExpertCacheCloseDevice&) = delete;
private:
    int previous_ = -1;
    bool switched_ = false;
};

// close() is a lifecycle operation, not the token hot path. Other streams may
// read cache allocations, so a legacy-stream fence alone is not enough.
inline void expert_cache_close_or_exit() noexcept {
    const cudaError_t completed = cudaDeviceSynchronize();
    if (completed != cudaSuccess)
        core::transfer_terminal("ExpertCache::close completion", cudaGetErrorString(completed));
}

} // namespace strata::core::detail
