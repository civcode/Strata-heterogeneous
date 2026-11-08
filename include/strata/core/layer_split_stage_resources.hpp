#pragma once

#include "strata/core/transfer_terminal.hpp"

#include <cuda_runtime.h>

namespace strata::core {

// Declared BEFORE dependent stage members, so this destructor runs AFTER their
// destructors. LayerSplitStageDelete selects/fences the owner GPU before delete.
// The resources are stage-owned, not aliases of NativeDense/Verifier allocations.
struct LayerSplitStageResources {
    void* weight_arena = nullptr;
    void* session_arena = nullptr;
    void* residency = nullptr;
    void* mrope = nullptr;
    cudaStream_t stream = nullptr;
    cudaStream_t adapt_stream = nullptr;
    cudaEvent_t adapt_event = nullptr;

    LayerSplitStageResources() = default;
    LayerSplitStageResources(const LayerSplitStageResources&) = delete;
    LayerSplitStageResources& operator=(const LayerSplitStageResources&) = delete;
    LayerSplitStageResources(LayerSplitStageResources&&) = delete;
    LayerSplitStageResources& operator=(LayerSplitStageResources&&) = delete;

    ~LayerSplitStageResources() noexcept {
        // Streams/events outlive dependent objects; buffers outlive the streams.
        // If release fails, stop without running further uncertain cleanup.
        if (adapt_event) check(cudaEventDestroy(adapt_event), "stage adapt event");
        if (adapt_stream) check(cudaStreamDestroy(adapt_stream), "stage adapt stream");
        if (stream) check(cudaStreamDestroy(stream), "stage stream");
        if (residency) check(cudaFree(residency), "stage residency table");
        if (mrope) check(cudaFree(mrope), "stage vision table");
        if (session_arena) check(cudaFree(session_arena), "stage session arena");
        if (weight_arena) check(cudaFree(weight_arena), "stage weight arena");
    }

private:
    static void check(cudaError_t e, const char* what) {
        if (e != cudaSuccess) transfer_terminal(what, cudaGetErrorString(e));
    }
};

} // namespace strata::core
