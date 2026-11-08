#pragma once
#include "strata/core/transfer_terminal.hpp"
#include <cuda_runtime.h>

namespace strata::core {

// Constructed before stage and verifier locals; destroyed after they unwind.
// The LayerSplitStageDelete policy selects and fences the owning GPU before
// this destructor runs. All pointers here are borrowed by consumers, owned here.
struct PrimaryServingResources {
    int dev = -1;
    bool cleanup_armed = false;
    void* session_arena = nullptr;
    void* logits = nullptr;
    void* hit_scratch = nullptr;
    void* hit_slot = nullptr;
    void* hit_destination = nullptr;
    void* hit_q8 = nullptr;
    void* hit_q8_scale = nullptr;
    void* hit_output = nullptr;
    cudaEvent_t hit_done = nullptr;
    void* embedding = nullptr;
    void* next_token = nullptr;
    void* residency = nullptr;
    void* hit_count = nullptr;

    PrimaryServingResources() = default;
    PrimaryServingResources(const PrimaryServingResources&) = delete;
    PrimaryServingResources& operator=(const PrimaryServingResources&) = delete;

    ~PrimaryServingResources() noexcept {
        if (hit_done) check(cudaEventDestroy(hit_done), "primary hit completion event");
        if (hit_count) check(cudaFree(hit_count), "primary hit count");
        if (residency) check(cudaFree(residency), "primary residency");
        if (next_token) check(cudaFree(next_token), "primary next token");
        if (embedding) check(cudaFree(embedding), "primary embedding");
        if (hit_output) check(cudaFree(hit_output), "primary hit output");
        if (hit_q8_scale) check(cudaFree(hit_q8_scale), "primary hit scale");
        if (hit_q8) check(cudaFree(hit_q8), "primary hit q8");
        if (hit_destination) check(cudaFree(hit_destination), "primary hit destination");
        if (hit_slot) check(cudaFree(hit_slot), "primary hit slot");
        if (hit_scratch) check(cudaFree(hit_scratch), "primary hit scratch");
        if (logits) check(cudaFree(logits), "primary logits");
        if (session_arena) check(cudaFree(session_arena), "primary session arena");
    }
private:
    static void check(cudaError_t status, const char* context) noexcept {
        if (status != cudaSuccess)
            transfer_terminal(context, cudaGetErrorString(status));
    }
};

} // namespace strata::core
