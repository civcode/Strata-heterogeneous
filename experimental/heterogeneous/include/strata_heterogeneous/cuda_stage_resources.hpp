#pragma once

// CUDA-worker-only, opt-in owner of a stage's native pack arena and carved
// sequence session. Never include/link in a HIP or coordinator process.
#include "strata_heterogeneous/cuda_verifier.hpp"
#include "strata_heterogeneous/stage_weights.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

namespace strata::heterogeneous {
class CudaNativeStage;

// This interface is supplied by a *real* worker expert-source/cache builder.
// A callback with invented/zero expert values is NOT a working inference pool.
// The resource owner owns and destroys the provider before GPU weights/state.
class CudaStageExperts {
public:
    virtual ~CudaStageExperts() = default;
    virtual core::VerifyHits hits() const = 0;
    virtual void run(const float* x_f, const int32_t* ids,
                     int64_t n_tok, int64_t k, float* out, int64_t layer) = 0;
};

struct CudaStageLoadOptions {
    std::string pack_dir;
    // M1.17 real worker requires NativeDense and PLE (where layer 1 is owned)
    // before make_bindings. Kept opt-in for existing M1.14 contract fixtures.
    bool require_native_stage = false;
    int64_t max_cells = 0;
    int64_t experts_per_token = 0;
    int64_t initial_position = 0; // ONLY fresh zero-state supported in M1.14
    int max_t = 8;
    int timeout_ms = 30000;
    // Additional free device memory to preserve for head/graphs/experts.
    uint64_t free_reserve_bytes = 512ull << 20;
    // Optional limit on weight + session arenas, zero means no extra cap.
    uint64_t arena_budget_bytes = 0;
};

// One-shot lifetime: load -> attach_experts -> make_bindings -> stop verifier
// -> shutdown. Any CUDA/load failure poisons the owner. This class does NOT
// construct a model geometry from GGUF, prefill prompts, create an expert
// source, manage an active Verifier, or prove model/fingerprint authenticity.
class CudaStageResources final {
public:
    CudaStageResources(core::ModelGeometry geometry, uint64_t model_fingerprint)
        : geometry_(std::move(geometry)), model_fingerprint_(model_fingerprint) {}
    ~CudaStageResources() noexcept { shutdown(); }
    CudaStageResources(const CudaStageResources&) = delete;
    CudaStageResources& operator=(const CudaStageResources&) = delete;

    bool load(const Plan& plan, size_t stage_index,
              const CudaStageLoadOptions& options, std::string& error);
    bool attach_experts(std::unique_ptr<CudaStageExperts> experts, std::string& error);
    // Borrowed references valid only while this owner and its expert provider
    // live. Destroy CudaVerifierDriver / CudaFinalVerifierExecutor FIRST.
    CudaVerifierBindings make_bindings();
    void shutdown() noexcept;

    bool loaded() const noexcept { return state_ == State::loaded; }
    bool ready() const noexcept { return loaded() && experts_ != nullptr &&
         (!native_required_ || native_ready_); }
    bool native_ready() const noexcept { return native_ready_; }
    bool poisoned() const noexcept { return state_ == State::dead; }
    uint64_t weight_bytes() const noexcept { return weight_bytes_; }
    uint64_t session_bytes() const noexcept { return session_bytes_; }
    const StageWeightPlan& weight_plan() const noexcept { return selected_; }
    const core::SessionState& session() const noexcept { return session_; }
private:
    friend class CudaNativeStage; // grants access to the live WeightTable and SessionState
    enum class State { fresh, initializing, loaded, dead };
    static void pool_dispatch(void* ctx, const float* x_f, const int32_t* ids,
                              int64_t n_tok, int64_t k, float* out, int64_t layer);
    core::ModelGeometry geometry_;
    uint64_t model_fingerprint_ = 0;
    State state_ = State::fresh;
    StageWeightPlan selected_;
    std::string pack_dir_;
    bool native_required_ = false;
    bool native_attached_ = false; // owner shutdown before native teardown is fatal
    bool native_ready_ = false;
    core::WeightTable weights_;
    core::SessionState session_{};
    std::unique_ptr<CudaStageExperts> experts_;
    void* weight_arena_ = nullptr;
    void* session_arena_ = nullptr;
    uint64_t weight_bytes_ = 0;
    uint64_t session_bytes_ = 0;
    int device_ = -1;
    int max_t_ = 0;
    int timeout_ms_ = 0;
    int64_t initial_position_ = 0;
};

} // namespace strata::heterogeneous
