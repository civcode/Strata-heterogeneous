#include "strata_heterogeneous/hip_verifier.hpp"
#if !defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906) || defined(STRATA_GPU_METAL)
#error "HipVerifierDriver requires an isolated wave32 HIP-only build"
#endif
#include <chrono>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace strata::heterogeneous {
bool HipVerifierDriver::check(cudaError_t code, std::string& error) {
    if (code == cudaSuccess) return true;
    error = std::string("HIP verifier: ") + cudaGetErrorString(code);
    return false;
}
bool HipVerifierDriver::expect(State state, std::string& error) {
    if (state_ != state) { error = "HIP verifier lifecycle mismatch"; return false; }
    return device_ < 0 || check(cudaSetDevice(device_), error);
}
bool HipVerifierDriver::record(std::string& error) {
    return check(cudaEventRecord(fence_, verifier_->stream()), error);
}
bool HipVerifierDriver::wait(std::string& error) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(bindings_.timeout_ms);
    for (;;) {
        const auto code = cudaEventQuery(fence_);
        if (code != cudaErrorNotReady) return check(code, error);
        if (std::chrono::steady_clock::now() >= deadline) {
            error = "HIP verifier fence timed out";
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
bool HipVerifierDriver::prepare(const Plan& plan, size_t index, std::string& error) {
    if (!expect(State::fresh, error)) return false;
    plan.validate();
    const auto& g = bindings_.geometry;
    const auto& ss = bindings_.session;
    if (index >= plan.stages.size()) { error = "HIP verifier stage index out of range"; return false; }
    const auto& stage = plan.stages[index];
    if (stage.backend != Backend::hip || stage.ordinal > INT_MAX ||
        stage.end_layer == plan.total_layers || plan.total_layers != g.n_layers ||
        plan.geometry.hc != g.hc || plan.geometry.n_embd != g.n_embd ||
        plan.model_fingerprint != bindings_.model_fingerprint ||
        ss.layer_lo != stage.first_layer || ss.layer_hi != stage.end_layer ||
        ss.max_cells <= 0 || ss.max_cells > INT_MAX || g.n_vocab <= 0 ||
        bindings_.initial_position < 0 || bindings_.initial_position > ss.max_cells ||
        bindings_.pool == nullptr || bindings_.timeout_ms <= 0 ||
        bindings_.max_t < 1 || bindings_.max_t > strata::kernels::kVerifyMaxT) {
        error = "HIP verifier requires matching loaded non-final stage, carved session, pool and limits";
        return false;
    }
    const uint64_t row_floats = plan.geometry.floats_per_row();
    if (row_floats > max_payload_bytes / (sizeof(float) * bindings_.max_t)) {
        error = "HIP verifier handoff exceeds protocol cap"; return false;
    }
    const size_t capacity = static_cast<size_t>(row_floats) * sizeof(float) * bindings_.max_t;
    plan_ = plan;
    index_ = index;
    device_ = static_cast<int>(stage.ordinal);
    // The layer-zero stage embeds tokens and never reads handoff_in.
    // Reserve an activation input only for later stages. Keep the output
    // and pinned host staging allocations for every non-final stage.
    if (!check(cudaSetDevice(device_), error) ||
        (stage.first_layer != 0 &&
         !check(cudaMalloc(reinterpret_cast<void**>(&input_), capacity), error)) ||
        !check(cudaMalloc(reinterpret_cast<void**>(&output_), capacity), error) ||
        !check(cudaMallocHost(&staging_, capacity), error) ||
        !check(cudaEventCreateWithFlags(&fence_, cudaEventDisableTiming), error)) return false;
    verifier_ = std::make_unique<core::Verifier>();
    verifier_->set_stage(stage.first_layer, stage.end_layer,
                         stage.first_layer ? input_ : nullptr, output_);
    verifier_->set_handoff_rows(bindings_.max_t);
    verifier_->set_next(nullptr, nullptr); // no in-process device chain
    verifier_->set_handoff_flags(nullptr, nullptr, nullptr, nullptr);
    // Prevent T=1 from committing GDN/indexer state inside run(). Also forces
    // pool publication, so an actual live pool is mandatory even if resident.
    verifier_->set_always_publish(true);
    verifier_->set_head_sampling(false);
    if (!taps_.configure(*verifier_, plan, index, error)) return false;
    if (!layer_zero_.configure(*verifier_, plan, index, error)) return false;
    if (!stage0_phases_.configure(*verifier_, plan, index, error)) return false;
    if (!branch_probe_.configure(*verifier_, plan, index, error)) return false;
    if (!verifier_->init(bindings_.weights, g, bindings_.session, bindings_.hits,
                          nullptr, bindings_.max_t, error)) return false;
    next_position_ = bindings_.initial_position;
    state_ = State::ready;
    return true;
}
bool HipVerifierDriver::begin_window(const WindowSpec& spec, uint64_t request,
                                      uint64_t window, std::string& error) {
    if (!expect(State::ready, error)) return false;
    spec.validate(static_cast<uint32_t>(spec.token_ids.size()));
    if (!request || !window || (request_ && request != request_) ||
        (window_ && window <= window_) || spec.pos0 != next_position_ ||
        spec.token_ids.size() > static_cast<size_t>(bindings_.max_t) ||
        spec.pos0 > bindings_.session.max_cells - static_cast<int64_t>(spec.token_ids.size())) {
        error = "HIP verifier window identity, position, context or row count is invalid"; return false;
    }
    for (auto token : spec.token_ids) if (token >= bindings_.geometry.n_vocab) {
        error = "HIP verifier token outside vocabulary"; return false;
    }
    spec_ = spec;
    request_ = request; window_ = window;
    bytes_ = spec.token_ids.size() * plan_.geometry.floats_per_row() * sizeof(float);
    state_ = State::begun;
    return true;
}
bool HipVerifierDriver::enqueue_upload(const uint8_t* host, size_t bytes, std::string& error) {
    if (!expect(State::begun, error)) return false;
    if (!host || bytes != bytes_) { error = "HIP verifier upload shape mismatch"; return false; }
    state_ = State::uploading;
    if (plan_.stages[index_].first_layer == 0) {
        // set_stage(..., nullptr, ...) uses embeddings for layer zero;
        // uploading the received activation would be discarded. Still
        // record a stream fence so wait_upload keeps its normal contract.
        return record(error);
    }
    std::memcpy(staging_, host, bytes);
    return check(cudaMemcpyAsync(input_, staging_, bytes, cudaMemcpyHostToDevice, verifier_->stream()), error) && record(error);
}
bool HipVerifierDriver::wait_upload(std::string& error) {
    if (!expect(State::uploading, error) || !wait(error)) return false;
    state_ = State::uploaded; return true;
}
bool HipVerifierDriver::enqueue_execute(const Frame& frame, std::string& error) {
    if (!expect(State::uploaded, error)) return false;
    if (frame.kind != Kind::data || frame.stage_index != index_ ||
        frame.request_id != request_ || frame.window_id != window_ ||
        frame.plan_fingerprint != plan_.fingerprint() || frame.rows != spec_.token_ids.size() ||
        frame.geometry.hc != plan_.geometry.hc || frame.geometry.n_embd != plan_.geometry.n_embd) {
        error = "HIP verifier execution metadata mismatch"; return false;
    }
    state_ = State::executing;
    int32_t ignored[strata::kernels::kVerifyMaxT] = {};
    return verifier_->run(static_cast<int>(spec_.token_ids.size()), spec_.token_ids.data(),
                          spec_.pos0, bindings_.pool, bindings_.pool_user, ignored, error) && record(error);
}
bool HipVerifierDriver::wait_execute(std::string& error) {
    if (!expect(State::executing, error) || !wait(error)) return false;
    // The compute stream completed. Gather post-layer diagnostics before
    // an accepted-prefix commit can overwrite the verifier's tap buffers.
    if (!taps_.record(*verifier_, plan_, spec_, request_, window_, error)) return false;
    if (!layer_zero_.record(*verifier_, plan_, spec_, request_, window_, error)) return false;
    if (!stage0_phases_.record(*verifier_, plan_, spec_, request_, window_, error)) return false;
    if (!branch_probe_.record(*verifier_, plan_, spec_, request_, window_, error)) return false;
    state_ = State::executed; return true;
}
bool HipVerifierDriver::enqueue_download(uint8_t* host, size_t bytes, std::string& error) {
    if (!expect(State::executed, error)) return false;
    if (!host || bytes != bytes_) { error = "HIP verifier download shape mismatch"; return false; }
    download_host_ = host;
    state_ = State::downloading;
    return check(cudaMemcpyAsync(staging_, output_, bytes, cudaMemcpyDeviceToHost, verifier_->stream()), error) && record(error);
}
bool HipVerifierDriver::wait_download(std::string& error) {
    if (!expect(State::downloading, error) || !wait(error)) return false;
    std::memcpy(download_host_, staging_, bytes_);
    download_host_ = nullptr;
    state_ = State::downloaded; return true;
}
bool HipVerifierDriver::enqueue_commit(uint64_t, uint64_t, std::string& error) {
    error = "HIP verifier requires explicit v4 tokens and prefix commit"; return false;
}
bool HipVerifierDriver::enqueue_commit_prefix(uint64_t request, uint64_t window,
                                               uint32_t keep, std::string& error) {
    if (!expect(State::downloaded, error)) return false;
    if (request != request_ || window != window_ || !keep || keep > spec_.token_ids.size()) {
        // commit(0) is NOT a no-op: run has overwritten the indexer key tail.
        error = "HIP verifier requires matching positive accepted prefix; zero needs state restoration";
        return false;
    }
    keep_ = keep;
    state_ = State::committing;
    return verifier_->commit(static_cast<int>(keep), error) && record(error);
}
bool HipVerifierDriver::wait_commit(std::string& error) {
    if (!expect(State::committing, error) || !wait(error)) return false;
    next_position_ += keep_;
    state_ = State::ready; spec_ = {}; return true;
}
void HipVerifierDriver::abort() noexcept {
    if (state_ == State::dead) return;
    state_ = State::dead;
    download_host_ = nullptr;
    // No CUDA operation references the borrowed IPC slot. Never free our own
    // pinned/device allocations while the verifier's work might still use them.
    if (device_ >= 0 && cudaSetDevice(device_) != cudaSuccess) std::_Exit(70);
    if (verifier_ && (!verifier_->release_gpu_waits(bindings_.timeout_ms) ||
                      cudaStreamQuery(verifier_->stream()) != cudaSuccess)) std::_Exit(70);
}
void HipVerifierDriver::shutdown() noexcept {
    if (state_ != State::fresh && state_ != State::ready && state_ != State::dead) abort();
    if (device_ >= 0 && cudaSetDevice(device_) != cudaSuccess) std::_Exit(70);
    verifier_.reset(); // graphs/streams must die BEFORE their captured pointers
    if (fence_) { if (cudaEventDestroy(fence_) != cudaSuccess) std::_Exit(70); fence_ = nullptr; }
    if (input_) { if (cudaFree(input_) != cudaSuccess) std::_Exit(70); input_ = nullptr; }
    if (output_) { if (cudaFree(output_) != cudaSuccess) std::_Exit(70); output_ = nullptr; }
    if (staging_) { if (cudaFreeHost(staging_) != cudaSuccess) std::_Exit(70); staging_ = nullptr; }
    device_ = -1;
    state_ = State::dead;
}
} // namespace strata::heterogeneous
