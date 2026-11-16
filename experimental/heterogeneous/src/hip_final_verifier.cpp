#include "strata_heterogeneous/hip_final_verifier.hpp"
#include "strata_heterogeneous/boundary_capture.hpp"
#include "strata_heterogeneous/performance.hpp"
#include "strata_heterogeneous/receiver_publish.hpp"
#include <atomic>
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
        std::this_thread::sleep_for(fence_poll_interval());
    }
}
bool HipFinalVerifierExecutor::bind_direct_input(DirectVerifiedBinding b, std::string& error) {
    if (state_ != State::fresh || direct_ || options_.receiver_arm || !b || b.guarded != options_.guarded_direct) {
        error = "direct HIP input binding refused"; return false;
    }
    direct_ = b;
    return true;
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
    if (observer_ && (chain_stages_.size()!=2 || !options_.include_logits || bindings_.initial_position!=0)) {
        error = "terminal observer requires grouped native HIP and full logits from fresh state"; return false;
    }
    if (options_.native_split_overlap && chain_stages_.empty()) {
        error = "native split overlap requires grouped HIP"; return false;
    }
    input_index_ = index;
    if (!chain_stages_.empty()) {
        const auto domain = worker_domains(plan, true).back();
        if (index != domain.terminal_stage() || chain_stages_.size() != domain.size()) {
            error = "HIP group does not own complete terminal domain"; return false;
        }
        input_index_ = domain.first_stage;
    }
    const Stage& stage = plan.stages[input_index_];
    if (options_.receiver_arm && (!options_.native_split_overlap || chain_stages_.size()!=2 ||
                                  input_index_!=1 || stage.first_layer<=1 || plan.stages[0].backend!=Backend::cuda)) {
        error = "external receiver arm requires CUDA then native HIP pair"; return false;
    }
    if (stage.backend != Backend::hip || stage.ordinal > INT_MAX ||
        plan.stages[index].end_layer != plan.total_layers ||
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
    if (!select_device(error)) return false;
    if (options_.guarded_direct && !direct_) { error = "guarded direct requires registered input"; return false; }
    if (direct_) {
        if (!direct_verified_plan(plan) || !options_.native_split_overlap || observer_ ||
            !options_.include_logits || chain_stages_.size() != 2 || bindings_.max_t != 8 ||
            direct_.capacity != capacity_) {
            error = "direct HIP input capacity/plan/mode mismatch"; return false;
        }
        input_ = direct_.device_payload;
        input_flag_host_ = direct_.host_ready;
        input_flag_device_ = direct_.device_ready;
    } else if (options_.receiver_arm) {
        // This mapping belongs ONLY to HIP. The existing socket/memfd is
        // CPU-visible transport, not registered with either runtime.
        const unsigned flags = cudaHostAllocMapped | cudaHostAllocPortable | hipHostMallocCoherent;
        if (!check(cudaHostAlloc(&staging_, capacity_, flags), error) ||
            !check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&input_), staging_, 0), error) ||
            !check(cudaHostAlloc(reinterpret_cast<void**>(&input_flag_host_), 64, flags), error) ||
            !check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&input_flag_device_), input_flag_host_, 0), error))
            return false;
        std::memset(staging_, 0, capacity_);
        std::memset(input_flag_host_, 0, 64);
    } else if (stage.first_layer != 0 &&
               (!check(cudaMalloc(reinterpret_cast<void**>(&input_), capacity_), error) ||
                !check(cudaMallocHost(&staging_, capacity_), error))) return false;
    if (!check(cudaEventCreateWithFlags(&fence_, cudaEventDisableTiming), error)) return false;
    if (!chain_stages_.empty()) {
        if (!chain_.prepare(plan, chain_stages_, input_, nullptr, error, options_.native_split_overlap,
                            input_flag_host_, input_flag_device_,options_.guarded_direct)) return false;
        verifier_ = &chain_.first();
    } else {
    owned_verifier_ = std::make_unique<core::Verifier>();
    verifier_ = owned_verifier_.get();
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
    }
    if (verifier_->vocab() != g.n_vocab) {
        error = "HIP final verifier vocabulary differs from loaded model";
        return false;
    }
    if (observer_ && (!check(cudaSetDevice(chain_.last().device()), error) ||
        !observer_->prepare(chain_stages_.back().bindings,chain_.last(),error) || !select_device(error))) return false;
    next_position_ = bindings_.initial_position;
    state_ = State::ready;
    return true;
}
bool HipFinalVerifierExecutor::reset_request(uint64_t request, std::string& error) {
    PerformanceSpan perf("worker", "reset_local", index_, request, 0, 1);
    if (observer_ && (!observer_->resident_relay_enabled() || chain_stages_.empty())) { error="one-shot terminal observer cannot reset"; return false; }
    if (state_ != State::ready || !select_device(error) ||
        !request || request <= request_ || bindings_.initial_position != 0 ||
        !bindings_.zero_session || !verifier_) {
        if (error.empty()) error = "resident reset refused by native verifier";
        return false;
    }
    const bool overlap_reset = mtp_service_enabled() && performance_flag("STRATA_HETERO_MTP_RESET_OVERLAP");
    // All prior commits were fenced. Reuse native session_zero, including
    // stage-local GDN/QSA/PLE state, on the same verifier stream. Weights,
    // experts, device allocations and graphs remain resident.
    if (chain_stages_.empty()) {
        bindings_.zero_session(bindings_.session, bindings_.geometry, verifier_->stream());
    } else {
        // Reset each native session on its owning stream. Never reset a
        // partial group and acknowledge it as a fresh request.
        for (size_t j = 0; j < chain_stages_.size(); ++j) {
            const auto& b = chain_stages_[j].bindings;
            auto& v = chain_.at(j);
            if (!b.zero_session || !check(cudaSetDevice(v.device()), error)) return false;
            {
                PerformanceSpan enqueue("worker", "reset_enqueue", input_index_+j, request, 0, 1);
                b.zero_session(b.session, b.geometry, v.stream());
                if (overlap_reset && !check(cudaGetLastError(), error)) return false;
            }
            if (!overlap_reset && !check(cudaStreamSynchronize(v.stream()), error)) return false;
        }
        // The two stage-local sessions are independent. Let vanilla's async
        // zeroing run on both existing verifier streams before joining them.
        // Do not reset the terminal observer or ACK until every owner drains.
        if (overlap_reset) {
            for (size_t j = 0; j < chain_stages_.size(); ++j) {
                auto& v = chain_.at(j);
                PerformanceSpan join("worker", "reset_join", input_index_+j, request, 0, 1);
                if (!check(cudaSetDevice(v.device()), error) ||
                    !check(cudaStreamSynchronize(v.stream()), error)) return false;
            }
        }
        if (!select_device(error)) return false;
    }
    if (!overlap_reset && (!fence(error) || !wait_fence(error))) return false;
    if (observer_ && (!check(cudaSetDevice(chain_.last().device()),error) ||
        !observer_->reset_request(request,error) || !select_device(error))) {abort_once();return false;}
    observer_predictions_.clear();
    request_ = request;
    window_ = 0;
    next_position_ = 0;
    keep_ = 0;
    spec_ = {};
    unsafe_result_ = false;
    return true;
}
bool HipFinalVerifierExecutor::adopt_native_prefill(uint64_t request,uint32_t rows,std::string& error) {
    // The native prompt engine has fully drained the same stage-local session.
    if (receiver_arm_enabled() || state_!=State::ready || request_!=request || !request || window_ || next_position_ ||
        !rows || rows>=bindings_.session.max_cells || chain_stages_.empty()) {
        error="native prefill position adoption refused"; return false;
    }
    if (observer_ && (!observer_->service_enabled() || !observer_->finish_bulk_prompt(request,rows,error) || !select_device(error))) {
        if(error.empty()) error="native MTP bulk prompt adoption refused";
        abort_once();return false;
    }
    next_position_=rows; window_=1; return true;
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
    // Complete draft KV pairing before the next run can overwrite a borrowed
    // tail residual or arm either native HIP graph.
    if (observer_ && (!check(cudaSetDevice(chain_.last().device()),error) ||
        !observer_->begin(spec,request,window,error) || !select_device(error))) { abort_once(); return false; }
    unsafe_result_ = false;
    spec_ = spec;
    request_ = request;
    window_ = window;
    state_ = State::begun;
    return true;
}
bool HipFinalVerifierExecutor::arm_window(std::string& error) {
    if (!receiver_arm_enabled() || state_ != State::begun || armed_ || !input_flag_host_ ||
        !select_device(error)) {
        if (error.empty()) error = "HIP receiver arm not admitted";
        return false;
    }
    // Prior accepted-prefix commit was fully fenced. Do not clear this
    // word again until that window has drained and the next is admitted.
    *(volatile uint32_t*)input_flag_host_ = 0;
    std::atomic_thread_fence(std::memory_order_seq_cst);
    strata_store_fence();
    if (!verifier_->arm_external_handoff(static_cast<int>(spec_.token_ids.size()),
                                         spec_.token_ids.data(), spec_.pos0, error)) return false;
    armed_ = true;
    if (performance_flag("STRATA_HETERO_OVERLAP_TRACE"))
        std::fprintf(stderr, "HETERO_RECEIVER_ARMED request=%llu window=%llu\n",
                     static_cast<unsigned long long>(request_), static_cast<unsigned long long>(window_));
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
    if (direct_) {
        // Checked device publication can release the already-armed HIP graph.
        // CPU metadata, checksum and finite checks still gate result/commit.
        // The old direct mode launches only after these host checks.
        strata_load_fence();
        if ((options_.guarded_direct && !armed_) ||
            !direct_.matches(input.data(), input.bytes()) || !direct_.published()) {
            error = "direct HIP input publication or slot mismatch"; return false;
        }
    } else if (options_.receiver_arm) {
        if (!armed_ || !input_flag_host_ || *(volatile uint32_t*)input_flag_host_ != 0) {
            error = "HIP receiver not armed for this input"; return false;
        }
        // The transport already validated identity, size and shared checksum.
        // Copy to a HIP-owned coherent allocation and fence BEFORE ready.
        PerformanceSpan perf("worker", "receiver_publish", index_, request_, window_, metadata.rows);
        if (!publish_receiver_input(staging_, capacity_, input_flag_host_, input.data(), input.bytes())) {
            error = "HIP mapped receiver publication refused"; return false;
        }
    } else if (plan_.stages[input_index_].first_layer != 0) {
        PerformanceSpan perf("worker", "final_upload", index_, request_, window_, spec_.token_ids.size());
        const size_t bytes = input.bytes();
        {
            PerformanceSpan copy("worker", "upload_host_copy", index_, request_, window_, spec_.token_ids.size());
            std::memcpy(staging_, input.data(), bytes); // raw memfd bytes, not float[]
        }
        if (!check(cudaMemcpyAsync(input_, staging_, bytes, cudaMemcpyHostToDevice,
                                   verifier_->stream()), error) ||
            !fence(error) || !wait_fence(error) ||
            !capture_device_handoff(input_, staging_, bytes, input_index_, request_,
                                    window_, "in", verifier_->stream(),
                                    [this](std::string& e) { return fence(e) && wait_fence(e); },
                                    error)) return false;
    } // layer-zero native Verifier uses token embeddings, not handoff bytes

    std::vector<int32_t> predictions(spec_.token_ids.size());
    {
        PerformanceSpan perf("worker", "final_execute", index_, request_, window_, predictions.size());
        if (!verifier_->run(static_cast<int>(predictions.size()), spec_.token_ids.data(),
                            spec_.pos0, bindings_.pool, bindings_.pool_user,
                            predictions.data(), error) ||
            (!native_mtp_completion() && (!fence(error) || !wait_fence(error)))) return false;
    }
    armed_ = false; // run consumed the prelaunch; all stages have drained
    if (options_.native_split_overlap && !chain_stages_.empty() &&
        performance_flag("STRATA_HETERO_OVERLAP_TRACE")) {
        std::fprintf(stderr, "HETERO_HIP_OVERLAP_TRACE request=%llu window=%llu handoff_flag=%u\n",
                     static_cast<unsigned long long>(request_), static_cast<unsigned long long>(window_),
                     chain_.trace_handoff_flag());
        chain_.last().trace_dump(stderr);
    }
    if (chain_stages_.empty()) {
        if (!taps_.record(*verifier_, plan_, spec_, request_, window_, error)) return false;
        if (!layer_zero_.record(*verifier_, plan_, spec_, request_, window_, error)) return false;
        if (!stage0_phases_.record(*verifier_, plan_, spec_, request_, window_, error)) return false;
        if (!branch_probe_.record(*verifier_, plan_, spec_, request_, window_, error)) return false;
        if (!micro_probe_.record(*verifier_, plan_, spec_, request_, window_, error)) return false;
    }
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
        // The recursive native head lives on the last HIP card.
        if (!chain_stages_.empty() && !check(cudaSetDevice(chain_.last().device()), error)) return false;
        PerformanceSpan perf("worker", "final_logits_download", index_, request_, window_, pending.rows);
        std::vector<float> logits(static_cast<size_t>(pending.rows) * pending.vocab_size);
        for (uint32_t t = 0; t < pending.rows; ++t) {
            if (!verifier_->copy_logits(static_cast<int>(t),
                    logits.data() + static_cast<size_t>(t) * pending.vocab_size)) {
                error = "HIP final verifier failed to download head logits";
                return false;
            }
        }
        pending.logits = std::move(logits);
        if (!select_device(error)) return false;
    }
    try { validate_final_result(pending); }
    catch (const std::exception& e) { error = e.what(); return false; }
    unsafe_result_ = pending.has_nonfinite();
    if (observer_) observer_predictions_ = pending.predicted_tokens;
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
    PerformanceSpan perf("worker", "final_commit", index_, request_, window_, spec_.token_ids.size());
    keep_ = n_keep;
    if (!verifier_->commit(static_cast<int>(n_keep), error) ||
        !verifier_->wait_commit(error) ||
        (!native_mtp_completion() && (!fence(error) || !wait_fence(error)))) return false;
    // Native verifier commit is drained. The terminal observer runs before
    // the domain ACK; failure poisons the session, never retries a commit.
    if (observer_ && (!check(cudaSetDevice(chain_.last().device()),error) ||
        !observer_->committed(spec_,request,window,n_keep,chain_.last(),observer_predictions_,error) ||
        !select_device(error))) { abort_once(); return false; }
    observer_predictions_.clear();
    if (window_ <= 3) {
        bool sampled = false;
        for (const auto& stage : chain_stages_) {
            if (!stage.bindings.capacity_probe) continue;
            sampled = true;
            if (!stage.bindings.capacity_probe("early_commit",error)) return false;
        }
        if (sampled && !select_device(error)) return false;
    }
    next_position_ += keep_;
    state_ = State::ready;
    unsafe_result_ = false;
    spec_ = {};
    return true;
}
bool HipFinalVerifierExecutor::configure_mtp_request(const Frame& meta,std::string& error) {
    if (!mtp_service_enabled() || state_!=State::ready || chain_stages_.empty() ||
        meta.request_id!=request_ || window_ || next_position_ ||
        !check(cudaSetDevice(chain_.last().device()),error) ||
        !observer_->configure_request(meta,error) || !select_device(error)) {
        if (error.empty()) error="terminal MTP request configuration refused";
        abort_once(); return false;
    }
    return true;
}
bool HipFinalVerifierExecutor::propose_mtp(const WindowSpec& anchor, WindowSpec& offer, std::string& error) {
    if (!observer_ || state_!=State::ready || chain_stages_.empty() ||
        anchor.pos0!=next_position_ || anchor.token_ids.size()!=1 ||
        !check(cudaSetDevice(chain_.last().device()),error) ||
        !observer_->propose(anchor,offer,error) || !select_device(error)) {
        if(error.empty()) error="terminal native MTP relay refused";
        abort_once(); return false;
    }
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
    if (!chain_stages_.empty()) chain_.abort(bindings_.timeout_ms);
    else if (verifier_ && (!verifier_->release_gpu_waits(bindings_.timeout_ms) ||
                      cudaStreamQuery(verifier_->stream()) != cudaSuccess)) std::_Exit(70);
}
void HipFinalVerifierExecutor::abandon_window() noexcept { abort_once(); }
void HipFinalVerifierExecutor::shutdown() noexcept {
    if (state_ != State::fresh && state_ != State::ready && state_ != State::dead)
        abort_once();
    if (device_ >= 0 && cudaSetDevice(device_) != cudaSuccess) std::_Exit(70);
    if (observer_) { observer_->shutdown(); observer_ = nullptr; }
    chain_.shutdown();
    if (device_ >= 0 && cudaSetDevice(device_) != cudaSuccess) std::_Exit(70);
    owned_verifier_.reset(); // captured graphs/streams released before input buffer
    verifier_ = nullptr;
    if (fence_) { if (cudaEventDestroy(fence_) != cudaSuccess) std::_Exit(70); fence_ = nullptr; }
    if (input_) {
        if (!options_.receiver_arm && !direct_ && cudaFree(input_) != cudaSuccess) std::_Exit(70);
        input_ = nullptr; // mapped alias is released with staging_, not cudaFree
    }
    if (input_flag_host_) {
        if (!direct_ && cudaFreeHost(input_flag_host_) != cudaSuccess) std::_Exit(70);
        input_flag_host_ = input_flag_device_ = nullptr;
    }
    if (staging_) { if (cudaFreeHost(staging_) != cudaSuccess) std::_Exit(70); staging_ = nullptr; }
    device_ = -1;
    state_ = State::dead;
}
} // namespace strata::heterogeneous
