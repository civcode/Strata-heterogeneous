#pragma once

#include "strata/core/transfer_terminal.hpp"
#include <cuda_runtime.h>

namespace strata::core {

// Created at engine entry (after the cvec shutdown guard). On return it is
// destroyed after later GPU-owning locals, but before cvec process tables.
// LayerSplitStageDelete selects and synchronizes 'dev' before invoking this
// destructor. These pointers are owned here, not by stage.raw or Verifier.
struct PrimaryEarlyResources {
    int dev = -1;
    bool cleanup_armed = false;
    void* weight_arena = nullptr;
    void* vision_table = nullptr;
    void* ple_embedding = nullptr;
    void* ple_scratch = nullptr;
    void* expert_partials = nullptr;
    cudaStream_t main_stream = nullptr;

    PrimaryEarlyResources() = default;
    PrimaryEarlyResources(const PrimaryEarlyResources&) = delete;
    PrimaryEarlyResources& operator=(const PrimaryEarlyResources&) = delete;

    ~PrimaryEarlyResources() noexcept {
        // All dependent GPU owners and their captured graphs have unwound.
        if (main_stream) check(cudaStreamDestroy(main_stream), "primary main stream");
        if (expert_partials) check(cudaFree(expert_partials), "primary expert partials");
        if (ple_scratch) check(cudaFree(ple_scratch), "primary PLE scratch");
        if (ple_embedding) check(cudaFree(ple_embedding), "primary PLE embedding");
        if (vision_table) check(cudaFree(vision_table), "primary vision table");
        if (weight_arena) check(cudaFree(weight_arena), "primary weight arena");
    }

private:
    static void check(cudaError_t status, const char* context) noexcept {
        if (status != cudaSuccess)
            transfer_terminal(context, cudaGetErrorString(status));
    }
};

} // namespace strata::core
