#pragma once

// CUDA-worker-only, opt-in expert provider for the Q2_0 canonical pack.
// It owns real FileExpertSource, ExpertCache and ExpertPool resources; the
// verifier must die before the parent CudaStageResources (which owns this).
#include "strata_heterogeneous/cuda_stage_resources.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace strata::heterogeneous {

struct CudaFileExpertOptions {
    std::string pack_dir;
    // Exactly one source of ranked pairs: trusted STRP/v1 profile file or
    // explicitly supplied (layer, expert) pairs. Empty is refused rather than
    // creating a fake/empty cache that Verifier::init cannot safely consume.
    std::string profile_path;
    std::vector<std::pair<int32_t, int32_t>> ranked_pairs;
    // 0 = fill all in-stage pairs; otherwise take up to this many in rank
    // order. A positive count equal to stage capacity requires every stage
    // expert; an incomplete profile is refused, never silently downgraded.
    int64_t max_cache_slots = 0;
    int cpu_workers = 1; // Explicit, bounded: avoid oversubscribing CPU cores.
    // GPU space left free *after* this cache and the residency table, for
    // verifier graphs, logits, transfers and other stage allocations.
    uint64_t free_reserve_bytes = 512ull << 20;
};

// A real canonical-pack expert tier (not a mock or zero callback). Opened only
// inside a supervised CUDA-only process after CudaStageResources::load.
// No CUDA/HIP pointers are exchanged over AF_UNIX or between backends.
class CudaFileExperts final : public CudaStageExperts {
public:
    CudaFileExperts() = default;
    ~CudaFileExperts() noexcept override { shutdown(); }
    CudaFileExperts(const CudaFileExperts&) = delete;
    CudaFileExperts& operator=(const CudaFileExperts&) = delete;

    bool open(const Plan& plan, size_t stage_index,
              const core::ModelGeometry& geometry,
              const CudaFileExpertOptions& options, std::string& error);
    core::VerifyHits hits() const override;
    void run(const float* x_f, const int32_t* ids,
             int64_t n_tok, int64_t k, float* out, int64_t layer) override;
    void shutdown() noexcept;

    bool ready() const noexcept { return state_ == State::ready; }
    bool poisoned() const noexcept { return state_ == State::dead; }
    size_t filled_slots() const noexcept { return filled_slots_; }

private:
    enum class State { fresh, initializing, ready, dead };
    State state_ = State::fresh;
    int device_ = -1;
    int64_t first_ = 0, end_ = 0, n_expert_ = 0, blob_bytes_ = 0;
    size_t filled_slots_ = 0;
    bool full_residency_ = false;
    uint64_t host_callbacks_ = 0, host_routes_ = 0, host_misses_ = 0;
    std::vector<int32_t> host_res_;
    int32_t* device_res_ = nullptr;
    core::FileExpertSource source_;
    core::ExpertCache cache_;
    std::unique_ptr<strata::kernels::cpu::ExpertPool> pool_;
    core::ExpertDispatch dispatch_{};
};

} // namespace strata::heterogeneous
