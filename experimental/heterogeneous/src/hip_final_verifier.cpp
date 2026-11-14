#include "strata_heterogeneous/hip_final_verifier.hpp"
#include "strata_heterogeneous/boundary_capture.hpp"
#if !defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906) || defined(STRATA_GPU_METAL)
#error "HipFinalVerifierExecutor requires an isolated wave32 HIP-only build"
#endif

#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

namespace strata::heterogeneous {
bool HipFinalVerifierExecutor::check(cudaError_t result, std::string& error) {
    if (result == cudaSuccess) return true;
    error = std::string("HIP final verifier: ") + cudaGetErrorString(result);
    return false;
}
bool HipFinalVerifierExecutor::select_device(std::string& error) {
    return device_ < 0 || check(cudaSetDevice(device_), error);
}
bool HipFinalVerifierExecutor::fence(std::string& error) {
    return check(cudaEventRecord(fence_, verifier_->stream()), error);
}
bool HipFinalVerifierExecutor::wait_fence(std::string& error) {
    const auto end = std::chrono::steady_clock::now() +
                     std::chrono::milliseconds(bindings_.timeout_ms);
    for (;;) {
        const cudaError_t status = cudaEventQuery(fence_);
        if (status != cudaErrorNotReady) return check(status, error);
        if (std::chrono::steady_clock::now() >= end) {
            error = "HIP final verifier stream fence timed out";
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
bool HipFinalVerifierExecutor::prepare(const Plan& plan, size_t index,
                                         std::string& error) {
    if (state_ != State::fresh) { error = "HIP final verifier already prepared or poisoned"; return false; }
    try { plan.validate(); }
    catch (const std::exception& e) { error = e.what(); return false; }
    const auto& g = bindings_.geometry;
    const auto& ss = bindings_.session;
    if (index >= plan.stages.size() || index + 1 != plan.stages.size()) {
        error = "HIP final verifier must own the final stage";
        return false;
    }
    const Stage& stage = plan.stages[index];
    if (stage.backend != Backend::hip || stage.ordinal > INT_MAX ||
        stage.end_layer != plan.total_layers ||
        plan.total_layers != g.n_layers ||
        plan.geometry.hc != g.hc || plan.geometry.n_embd != g.n_embd ||
        plan.model_fingerprint != bindings_.model_fingerprint ||
        ss.layer_lo != stage.first_layer || ss.layer_hi != stage.end_layer ||
        ss.max_cells <= 0 || ss.max_cells > INT_MAX ||
        g.n_vocab <= 0 || g.n_vocab > INT32_MAX ||
        bindings_.initial_position < 0 || bindings_.initial_position > ss.max_cells ||
        bindings_.pool == nullptr || bindings_.timeout_ms <= 0 ||
        bindings_.max_t < 1 || bindings_.max_t > strata::kernels::kVerifyMaxT ||
        (!options_.include_logits && !core::Verifier::nan_guard_on())) {
        error = "HIP final verifier requires loaded final HIP stage, valid session and pool";
        return false;
    }
    const uint64_t stride = plan.geometry.floats_per_row();
    if (stride > max_payload_bytes / (sizeof(float) * static_cast<uint64_t>(bindings_.max_t))) {
        error = "HIP final verifier input exceeds activation cap";
        return false;
    }
    capacity_ = static_cast<size_t>(stride) * sizeof(float) *
                static_cast<size_t>(bindings_.max_t);
    plan_ = plan;
    index_ = index;
    device_ = static_cast<int>(stage.ordinal);
    state_ = State::initializing; // cleanup also handles partial initialization
    // An unsplit [0,n_layers) reference embeds tokens inside Verifier and
    // has no prior activation handoff; never upload the dummy input.
    if (!select_device(error) ||
        (stage.first_layer != 0 &&
         (!check(cudaMalloc(reinterpret_cast<void**>(&input_), capacity_), error) ||
          !check(cudaMallocHost(&staging_, capacity_), error))) ||
        !check(cudaEventCreateWithFlags(&fence_, cudaEventDisableTiming), error))
        return false;
    verifier_ = std::make_unique<core::Verifier>();
    verifier_->set_stage(stage.first_layer, stage.end_layer,
                         stage.first_layer ? input_ : nullptr, nullptr);
    verifier_->set_handoff_rows(bindings_.max_t);
    verifier_->set_next(nullptr, nullptr);
    verifier_->set_handoff_flags(nullptr, nullptr, nullptr, nullptr);
    verifier_->set_always_publish(true); // prevent T=1 self-commit
    verifier_->set_head_sampling(true); // predictions are required
    if (!taps_.configure(*verifier_, plan, index, error)) return false;
    if (!layer_zero_.configure(*verifier_, plan, index, error)) return false;
    if (!stage0_phases_.configure(*verifier_, plan, index, error)) return false;
    if (!branch_probe_.configure(*verifier_, plan, index, error)) return false;
    if (!micro_probe_.configure(*verifier_, plan, index, error)) return false;
    if (!verifier_->init(bindings_.weights, g, bindings_.session, bindings_.hits,
                         nullptr, bindings_.max_t, error)) return false;
    if (verifier_->vocab() != g.n_vocab) {
        error = "HIP final verifier vocabulary differs from loaded model";
        return false;
    }
    next_position_ = bindings_.initial_position;
    state_ = State::ready;
    return true;
}
bool HipFinalVerifierExecutor::begin_window(const WindowSpec& spec,
                                              uint64_t request, uint64_t window,
                                              std::string& error) {
    if (state_ != State::ready || !select_device(error)) {
        if (error.empty()) error = "HIP final verifier has a pending or failed window";
        return false;
    }
    try { spec.validate(static_cast<uint32_t>(spec.token_ids.size())); }
    catch (const std::exception& e) { error = e.what(); return false; }
    if (!request || !window || (request_ && request != request_) ||
        (window_ && window <= window_) || spec.pos0 != next_position_ ||
        spec.token_ids.size() > static_cast<size_t>(bindings_.max_t) ||
        spec.pos0 > bindings_.session.max_cells - static_cast<int64_t>(spec.token_ids.size())) {
        error = "HIP final verifier window IDs, position or context do not match";
        return false;
    }
    for (const int32_t token : spec.token_ids)
        if (token >= bindings_.geometry.n_vocab) {
            error = "HIP final verifier token outside model vocabulary";
            return false;
        }
    // Count entire STFR packet, including variable optional logits and
    // protocol overhead, before executing any device work.
    const uint64_t rows = spec.token_ids.size();
    const uint64_t vocab = static_cast<uint64_t>(bindings_.geometry.n_vocab);
    const uint64_t max_floats = final_result_max_payload_bytes / sizeof(float);
    if (options_.include_logits && (rows > max_floats / vocab ||
        rows * vocab > max_floats - (2 * rows + 1))) {
        error = "HIP final verifier logits exceed STFR packet cap";
        return false;
    }
    unsafe_result_ = false;
    spec_ = spec;
    request_ = request;
    window_ = window;
    state_ = State::begun;
    return true;
}
bool HipFinalVerifierExecutor::run_final_window(const Frame& metadata,
                                                  ActivationView input,
                                                  FinalStageResult& result,
                                                  std::string& error) {
    if (state_ != State::begun || !select_device(error)) {
        if (error.empty()) error = "HIP final verifier window not admitted";
        return false;
    }
    if (metadata.kind != Kind::data || metadata.stage_index != index_ ||
        metadata.request_id != request_ || metadata.window_id != window_ ||
        metadata.plan_fingerprint != plan_.fingerprint() ||
        metadata.rows != spec_.token_ids.size() ||
        metadata.geometry.hc != plan_.geometry.hc ||
        metadata.geometry.n_embd != plan_.geometry.n_embd ||
        !metadata.payload.empty() || metadata.shared_checksum ||
        !metadata.error.empty() || metadata.window || metadata.keep_count ||
        metadata.identity != std::array<uint32_t,4>{} ||
        input.rows() != metadata.rows ||
        input.geometry().hc != metadata.geometry.hc ||
        input.geometry().n_embd != metadata.geometry.n_embd ||
        input.bytes() != spec_.token_ids.size() * plan_.geometry.floats_per_row() * sizeof(float)) {
        error = "HIP final verifier stage/activation identity mismatch";
        return false;
    }
    if (plan_.stages[index_].first_layer != 0) {
        const size_t bytes = input.bytes();
        std::memcpy(staging_, input.data(), bytes); // raw memfd bytes, not float[]
        if (!check(cudaMemcpyAsync(input_, staging_, bytes, cudaMemcpyHostToDevice,
                                   verifier_->stream()), error) ||
            !fence(error) || !wait_fence(error) ||
            !capture_device_handoff(input_, staging_, bytes, index_, request_,
                                    window_, "in", verifier_->stream(),
                                    [this](std::string& e) { return fence(e) && wait_fence(e); },
                                    error)) return false;
    } // layer-zero native Verifier uses token embeddings, not handoff bytes

    std::vector<int32_t> predictions(spec_.token_ids.size());
    if (!verifier_->run(static_cast<int>(predictions.size()), spec_.token_ids.data(),
                        spec_.pos0, bindings_.pool, bindings_.pool_user,
                        predictions.data(), error) ||
        !fence(error) || !wait_fence(error)) return false;
    if (!taps_.record(*verifier_, plan_, spec_, request_, window_, error)) return false;
    if (!layer_zero_.record(*verifier_, plan_, spec_, request_, window_, error)) return false;
    if (!stage0_phases_.record(*verifier_, plan_, spec_, request_, window_, error)) return false;
    if (!branch_probe_.record(*verifier_, plan_, spec_, request_, window_, error)) return false;
    if (!micro_probe_.record(*verifier_, plan_, spec_, request_, window_, error)) return false;
    FinalStageResult pending;
    pending.stage_index = static_cast<uint32_t>(index_);
    pending.request_id = request_;
    pending.window_id = window_;
    pending.plan_fingerprint = plan_.fingerprint();
    pending.rows = metadata.rows;
    pending.vocab_size = static_cast<uint32_t>(verifier_->vocab());
    pending.predicted_tokens = std::move(predictions);
    for (uint32_t t = 0; t < pending.rows; ++t)
        if (verifier_->nonfinite_in(static_cast<int>(t)))
            pending.nonfinite_rows.push_back(t);
    if (options_.include_logits) {
        std::vector<float> logits(static_cast<size_t>(pending.rows) * pending.vocab_size);
        for (uint32_t t = 0; t < pending.rows; ++t) {
            if (!verifier_->copy_logits(static_cast<int>(t),
                    logits.data() + static_cast<size_t>(t) * pending.vocab_size)) {
                error = "HIP final verifier failed to download head logits";
                return false;
            }
        }
        pending.logits = std::move(logits);
    }
    try { validate_final_result(pending); }
    catch (const std::exception& e) { error = e.what(); return false; }
    unsafe_result_ = pending.has_nonfinite();
    result = std::move(pending);
    state_ = State::provisional;
    return true;
}
bool HipFinalVerifierExecutor::commit_prefix(uint64_t request, uint64_t window,
                                              uint32_t n_keep, std::string& error) {
    if (state_ != State::provisional || !select_device(error)) {
        if (error.empty()) error = "HIP final verifier has no provisional window";
        return false;
    }
    if (request != request_ || window != window_ ||
        n_keep == 0 || n_keep > spec_.token_ids.size() || unsafe_result_) {
        error = "HIP final verifier requires a matching positive accepted prefix; zero needs restoration";
        return false;
    }
    keep_ = n_keep;
    if (!verifier_->commit(static_cast<int>(n_keep), error) ||
        !verifier_->wait_commit(error) ||
        !fence(error) || !wait_fence(error)) return false;
    next_position_ += keep_;
    state_ = State::ready;
    unsafe_result_ = false;
    spec_ = {};
    return true;
}
bool HipFinalVerifierExecutor::run_window(const Frame&, Frame&, std::string& error) {
    error = "final HIP verifier never returns activation frames";
    return false;
}
bool HipFinalVerifierExecutor::commit_window(uint64_t, uint64_t, std::string& error) {
    error = "final HIP verifier requires accepted-prefix commit";
    return false;
}
void HipFinalVerifierExecutor::abort_once() noexcept {
    if (aborted_) return;
    aborted_ = true;
    state_ = State::dead;
    if (device_ >= 0 && cudaSetDevice(device_) != cudaSuccess) std::_Exit(70);
    // An aborted worker is not reusable. Releasing waits must precede freeing
    // captured device allocations; a hung GPU requires supervisor SIGKILL.
    if (verifier_ && (!verifier_->release_gpu_waits(bindings_.timeout_ms) ||
                      cudaStreamQuery(verifier_->stream()) != cudaSuccess)) std::_Exit(70);
}
void HipFinalVerifierExecutor::abandon_window() noexcept { abort_once(); }
void HipFinalVerifierExecutor::shutdown() noexcept {
    if (state_ != State::fresh && state_ != State::ready && state_ != State::dead)
        abort_once();
    if (device_ >= 0 && cudaSetDevice(device_) != cudaSuccess) std::_Exit(70);
    verifier_.reset();  // captured graphs/streams released before input buffer
    if (fence_) { if (cudaEventDestroy(fence_) != cudaSuccess) std::_Exit(70); fence_ = nullptr; }
    if (input_) { if (cudaFree(input_) != cudaSuccess) std::_Exit(70); input_ = nullptr; }
    if (staging_) { if (cudaFreeHost(staging_) != cudaSuccess) std::_Exit(70); staging_ = nullptr; }
    device_ = -1;
    state_ = State::dead;
}
} // namespace strata::heterogeneous
