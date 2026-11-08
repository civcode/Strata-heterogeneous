#include "strata_heterogeneous/cuda_stage_resources.hpp"
#if defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906) || defined(STRATA_GPU_METAL) || defined(STRATA_ENABLE_SYCL)
#error "CudaStageResources requires a CUDA-only worker"
#endif
#include "strata/core/session.hpp"
#include "strata/kernels/mrope.hpp"
#include <cuda_runtime.h>

#include <climits>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <utility>

namespace strata::heterogeneous {
namespace {
bool check_cuda(cudaError_t code, const char* operation, std::string& error) {
    if (code == cudaSuccess) return true;
    error = std::string("CUDA stage resources: ") + operation + ": " + cudaGetErrorString(code);
    return false;
}
bool sum_bounded(uint64_t x, uint64_t y, uint64_t& sum) {
    if (y > std::numeric_limits<uint64_t>::max() - x) return false;
    sum = x + y;
    return true;
}
} // namespace

void CudaStageResources::pool_dispatch(void* ctx, const float* x_f,
                                        const int32_t* ids, int64_t n_tok,
                                        int64_t k, float* out, int64_t layer) {
    // The real expert provider is owned, never borrowed from a stack frame.
    static_cast<CudaStageExperts*>(ctx)->run(x_f, ids, n_tok, k, out, layer);
}

bool CudaStageResources::load(const Plan& plan, size_t index,
                              const CudaStageLoadOptions& options, std::string& error) {
    if (state_ != State::fresh) {
        error = "CUDA stage resource owner is already used or poisoned"; return false;
    }
    try {
        plan.validate();
        if (index >= plan.stages.size()) { error = "stage index outside plan"; return false; }
        const Stage& stage = plan.stages[index];
        if (stage.backend != Backend::cuda || stage.ordinal > INT_MAX ||
            !model_fingerprint_ || plan.model_fingerprint != model_fingerprint_ ||
            geometry_.n_layers != plan.total_layers ||
            geometry_.hc != plan.geometry.hc || geometry_.n_embd != plan.geometry.n_embd ||
            geometry_.n_vocab <= 0 || geometry_.n_vocab > INT32_MAX ||
            options.pack_dir.empty() || options.max_cells <= 0 || options.max_cells > INT_MAX ||
            options.experts_per_token <= 0 || options.experts_per_token > INT_MAX ||
            options.initial_position != 0 || // no prefill or restored session in this increment
            options.max_t < 1 || options.max_t > strata::kernels::kVerifyMaxT ||
            options.timeout_ms <= 0) {
            error = "CUDA stage resource geometry, fingerprint, backend or load options are invalid";
            return false;
        }
        StageWeightPlan selected;
        if (!plan_stage_weights(plan, index, options.pack_dir + "/index.txt", selected, error)) return false;
        uint64_t native_bytes = 0;
        if (!core::WeightTable::pool_bytes(options.pack_dir, native_bytes, error, &selected.skip)) return false;
        if (native_bytes != selected.selected_pool_bytes) {
            error = "stage weight planner and native loader disagree on arena size"; return false;
        }
        const uint64_t session_bytes = core::session_bytes(geometry_, options.max_cells,
            options.experts_per_token, stage.first_layer, stage.end_layer);
        uint64_t total = 0;
        if (!native_bytes || !session_bytes ||
            native_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
            session_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
            !sum_bounded(native_bytes, session_bytes, total) ||
            (options.arena_budget_bytes && total > options.arena_budget_bytes)) {
            error = "CUDA stage weight/session memory exceeds bounds or configured budget";
            return false;
        }
        // Only now touch CUDA; all index and shape failures are CPU-only.
        state_ = State::initializing;
        const int requested_device = static_cast<int>(stage.ordinal);
        if (!check_cuda(cudaSetDevice(requested_device), "cudaSetDevice", error)) {
            state_ = State::dead;
            return false;
        }
        device_ = requested_device;
        size_t free_bytes = 0, total_bytes = 0;
        if (!check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo", error) ||
            total > free_bytes || options.free_reserve_bytes > free_bytes - total) {
            if (error.empty()) error = "insufficient free CUDA memory for arenas and safety reserve";
            shutdown(); return false;
        }
        weight_bytes_ = native_bytes;
        session_bytes_ = session_bytes;
        if (!check_cuda(cudaMalloc(&weight_arena_, static_cast<size_t>(weight_bytes_)),
                        "weight arena allocation", error) ||
            !weights_.load(options.pack_dir, weight_arena_, weight_bytes_, error, &selected.skip) ||
            !check_cuda(cudaMalloc(&session_arena_, static_cast<size_t>(session_bytes_)),
                        "session arena allocation", error)) {
            shutdown(); return false;
        }
        // session_init() creates host-owned QSA descriptors and device-side RoPE
        // resources. The partial-initialization cleanup handles both.
        const uint64_t used = core::session_init(geometry_, options.max_cells,
            options.experts_per_token, session_arena_, session_,
            stage.first_layer, stage.end_layer);
        if (!used || used > session_bytes_ ||
            session_.layer_lo != stage.first_layer ||
            session_.layer_hi != stage.end_layer || session_.max_cells != options.max_cells) {
            error = "stage-local session initialization failed or exceeded arena";
            shutdown(); return false;
        }
        core::session_zero(session_, geometry_, nullptr, nullptr);
        if (!check_cuda(cudaDeviceSynchronize(), "session initialization sync", error)) {
            shutdown(); return false;
        }
        selected_ = std::move(selected);
        pack_dir_ = options.pack_dir;
        native_required_ = options.require_native_stage;
        max_t_ = options.max_t;
        timeout_ms_ = options.timeout_ms;
        initial_position_ = options.initial_position;
        state_ = State::loaded;
        error.clear();
        return true;
    } catch (const std::exception& ex) {
        error = ex.what();
        if (state_ == State::initializing) shutdown();
        return false;
    }
}

bool CudaStageResources::attach_experts(std::unique_ptr<CudaStageExperts> experts,
                                        std::string& error) {
    if (state_ != State::loaded || experts_ || !experts) {
        error = "expert provider must be attached once to a loaded stage";
        return false;
    }
    experts_ = std::move(experts);
    error.clear();
    return true;
}

CudaVerifierBindings CudaStageResources::make_bindings() {
    if (!ready()) throw std::logic_error("CUDA stage has no live expert provider or resources");
    return CudaVerifierBindings{weights_, geometry_, session_,
                                experts_->hits(), &pool_dispatch, experts_.get(),
                                model_fingerprint_, max_t_, timeout_ms_, initial_position_};
}

void CudaStageResources::shutdown() noexcept {
    if (state_ == State::dead) return;
    // This process owns its GPU runtime. On a driver failure we must never
    // free allocations that may still be referenced by a hung kernel.
    state_ = State::dead;
    // NativeDense and PLE weights/table are still referenced by verifier
    // graphs. The independently owned CUDA native stage must die FIRST.
    if (native_attached_) std::_Exit(70);
    if (device_ >= 0 && (cudaSetDevice(device_) != cudaSuccess ||
                         cudaDeviceSynchronize() != cudaSuccess)) std::_Exit(70);
    experts_.reset(); // expert callbacks and cache references go away first
    if (session_.qsa_states != nullptr) {
        // The upstream session_release() only checks the first qsa_alloc
        // array entries; a carved stage can start at a nonzero qsa_ord0.
        // Release registrations by GLOBAL ordinal before calling it.
        for (int64_t j = 0; j < session_.qsa_alloc; ++j) {
            auto& q = session_.qsa_states[session_.qsa_ord0 + j];
            if (q.owns_rope) {
                strata::kernels::rope_table_release(q.cos_tab);
                q.owns_rope = false;
            }
        }
        core::session_release(session_);
        delete[] session_.qsa_states;
        session_.qsa_states = nullptr;
    }
    if (session_arena_) {
        if (cudaFree(session_arena_) != cudaSuccess) std::_Exit(70);
        session_arena_ = nullptr;
    }
    if (weight_arena_) {
        if (cudaFree(weight_arena_) != cudaSuccess) std::_Exit(70);
        weight_arena_ = nullptr;
    }
    session_ = core::SessionState{};
    weights_ = core::WeightTable{};
    selected_ = StageWeightPlan{};
    pack_dir_.clear();
    native_required_ = native_ready_ = false;
    session_bytes_ = weight_bytes_ = 0;
    device_ = -1;
}
} // namespace strata::heterogeneous
