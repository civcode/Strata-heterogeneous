#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdio>
#include <initializer_list>

namespace strata::prefill::detail {
// Cleanup may destroy several independent staging owners. Every participating stream must drain;
// a failed wait is never permission to run those owners' destructors.
inline bool drain_prefill_streams(cudaStream_t compute, std::initializer_list<cudaStream_t> streams = {}) {
    // The compute stream may be the valid default (null) stream. Only optional streams are skipped.
    bool ok = cudaStreamSynchronize(compute) == cudaSuccess;
    for (cudaStream_t s : streams)
        if (s && cudaStreamSynchronize(s) != cudaSuccess) ok = false;
    return ok;
}

// Single run-thread completion status, observed by callers only after pipeline joins. Cleanup
// cannot throw: latch its error even if a later wait succeeds or the body had already returned true.
struct PrefillCompletion {
    cudaError_t error = cudaSuccess;
    cudaError_t observe(cudaError_t status) {
        if (error == cudaSuccess && status != cudaSuccess) error = status;
        return status;
    }
    cudaError_t stream(cudaStream_t s) { return observe(cudaStreamSynchronize(s)); }
    cudaError_t event(cudaEvent_t e) { return observe(cudaEventSynchronize(e)); }
    bool failed() const { return error != cudaSuccess; }
};
struct PrefillStreamDrain {
    PrefillCompletion& completion;
    cudaStream_t stream;
    ~PrefillStreamDrain() { if (stream) (void) completion.stream(stream); }
};

// Early return/cancellation must settle the borrowed compute/copy operands before refill.
// The normal path already waits explicitly, so it disarms this guard without another fence.
struct PrefillRunDrain {
    PrefillCompletion& completion;
    cudaStream_t compute;
    cudaStream_t copy;
    cudaStream_t kv = nullptr;
    cudaStream_t ple = nullptr;
    bool completed = false;
    ~PrefillRunDrain() {
        if (completed) return;
        (void) completion.stream(compute); // null is a valid compute stream
        if (copy) (void) completion.stream(copy);
        if (kv) (void) completion.stream(kv);
        if (ple) (void) completion.stream(ple);
    }
};

// The host grouping gate. Both the mapped-ID producer and an optional activation download must
// complete before callbacks/grouping may read their host storage. A failed gate is terminal for this
// Prefill instance; its GPU-visible grouping buffers must be retained, not recycled during cleanup.
struct HostGroupReadback {
    cudaError_t error = cudaSuccess;
    bool finish(cudaStream_t stream, cudaError_t submitted = cudaSuccess) {
        if (error != cudaSuccess) return false;
        const cudaError_t completed = cudaStreamSynchronize(stream);
        error = completed == cudaSuccess ? submitted : completed;
        return error == cudaSuccess;
    }
    bool failed() const { return error != cudaSuccess; }
    size_t retained_bytes = 0;
    template<class T> void release_host(T*& p, size_t bytes) {
        if (!p) return;
        // Even a later successful cleanup fence must not silently restart a failed grouping instance.
        // Retention here is conservative; the terminal gate prevents replacement/retry allocations.
        if (failed()) {
            retained_bytes += bytes;
            std::fprintf(stderr, "strata: host grouping: retaining %zu bytes until process exit\n", bytes);
        } else {
            const cudaError_t e = cudaFreeHost(p);
            if (e != cudaSuccess) { error = e; retained_bytes += bytes; }
        }
        p = nullptr; // never retry a free whose result was an error
    }
};
} // namespace strata::prefill::detail
