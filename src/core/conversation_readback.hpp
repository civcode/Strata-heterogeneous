#pragma once

#include "strata/core/progress.hpp"
#include "conversation_completion.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace strata::core::conversation_detail {

struct ReadbackOps {
    cudaError_t allocate(void** p, size_t n) { return cudaHostAlloc(p, n, cudaHostAllocPortable); }
    void release(void* p) { (void) cudaFreeHost(p); }
    cudaError_t submit(void* dst, const void* src, size_t n, cudaStream_t stream) {
        // Streamed K/V sources can be mapped host pools, not device allocations.
        return cudaMemcpyAsync(dst, src, n, cudaMemcpyDefault, stream);
    }
    cudaError_t synchronize(cudaStream_t stream) { return cudaStreamSynchronize(stream); }
    void progress() { core::progress_beat(); }
};

// One bounded, privately owned pinned buffer. The final destination is touched only after checked stream
// completion. Ops is injectable in tests; production has no failure-injection environment or mutable hooks.
// This does not order producers on other streams: callers must establish that dependency before capture.
template<class Ops = ReadbackOps> class PinnedReadback {
public:
    static constexpr size_t piece_bytes = 4 * 1024 * 1024;
    Ops ops;
    PinnedReadback() = default;
    PinnedReadback(const PinnedReadback&) = delete;
    PinnedReadback& operator=(const PinnedReadback&) = delete;
    ~PinnedReadback() { if (stage_) ops.release(stage_); }

    cudaError_t copy(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
        if (!bytes) return cudaSuccess;
        if (!stage_) {
            const cudaError_t e = ops.allocate(&stage_, piece_bytes);
            if (e != cudaSuccess) { stage_ = nullptr; return e; }
        }
        auto* out = static_cast<uint8_t*>(dst);
        const auto* in = static_cast<const uint8_t*>(src);
        for (size_t at = 0; at < bytes;) {
            const size_t n = std::min(piece_bytes, bytes - at);
            const cudaError_t submitted = ops.submit(stage_, in + at, n, stream);
            // Also drain after a failed submission: do not assume an error means no work was queued.
            const cudaError_t completed = ops.synchronize(stream);
            if (completed != cudaSuccess) {
                // Completion is unknown. Never free/reuse memory that the GPU might still reference.
                // Quarantine until process exit; caller must reject this capture and handle the GPU error.
                stage_ = nullptr;
                unknown_completion_ = true;
                return completed;
            }
            if (submitted != cudaSuccess) return submitted;
            std::memcpy(out + at, stage_, n);
            ops.progress();
            at += n;
        }
        return cudaSuccess;
    }
    bool unknown_completion() const { return unknown_completion_; }
private:
    void* stage_ = nullptr;
    bool unknown_completion_ = false;
};

// HIP snapshot readback avoids the pageable destination path implicated by the concurrent QSA reproducer.
// CUDA keeps its existing synchronous private-stream path; async CUDA staging is audited separately.
inline cudaError_t readback_host(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
#if defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906)
    thread_local PinnedReadback<> staging;
    const auto e = staging.copy(dst, src, bytes, stream);
    if (staging.unknown_completion())
        core::transfer_terminal("conversation readback completion", cudaGetErrorString(e));
    return e;
#else
    cudaError_t e = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, stream);
    return finish_copy(e, stream);
#endif
}

} // namespace strata::core::conversation_detail
