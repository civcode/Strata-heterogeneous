#include "strata_heterogeneous/final_pipeline.hpp"
#include "strata_heterogeneous/backend_domains.hpp"
#include "strata_heterogeneous/shared.hpp"
#include "strata_heterogeneous/performance.hpp"
#include "strata_heterogeneous/window_admission.hpp"
#include "strata_heterogeneous/speculation.hpp"
#include "strata_heterogeneous/mtp_resident.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <sys/socket.h>
#include <utility>

namespace strata::heterogeneous {
namespace {
Frame control(const Plan& plan, size_t index, Kind kind, uint64_t request = 1,
              uint64_t window = 1, uint32_t rows = 1) {
    Frame f;
    f.kind = kind;
    f.stage_index = static_cast<uint32_t>(index);
    f.request_id = request;
    f.window_id = window;
    f.rows = rows;
    f.plan_fingerprint = plan.fingerprint();
    f.geometry = plan.geometry;
    if (kind == Kind::hello_final || kind == Kind::hello_shared_final || kind == Kind::hello_direct_shared_final || kind == Kind::hello_guarded_direct_final || kind == Kind::hello_mtp_service_final) {
        const Stage& s = plan.stages.at(index);
        f.identity = {static_cast<uint32_t>(s.backend), s.ordinal, s.first_layer, s.end_layer};
    }
    return f;
}
void match(const Frame& got, const Frame& expected, Kind kind) {
    if (got.kind != kind || got.stage_index != expected.stage_index ||
        got.request_id != expected.request_id || got.window_id != expected.window_id ||
        got.plan_fingerprint != expected.plan_fingerprint || got.rows != expected.rows ||
        got.geometry.hc != expected.geometry.hc || got.geometry.n_embd != expected.geometry.n_embd)
        throw std::runtime_error("final pipeline frame kind, identity, plan or geometry mismatch");
}
void trace_window(const WindowAdmission* gate, const char* phase) noexcept {
    if (!gate) return;
    const char* opt = std::getenv("STRATA_HETERO_WINDOW_TRACE");
    if (!opt || std::strcmp(opt, "1") != 0) return;
    try { std::fprintf(stderr, "%s\n", gate->diagnostic(phase).c_str()); }
    catch (...) {}
}

void failure(int fd, const Frame& frame, std::string error, int timeout_ms) noexcept {
    try {
        Frame f = frame;
        f.kind = Kind::failure;
        f.identity = {};
        f.payload.clear();
        f.window.reset();
        f.keep_count = 0;
        f.prefill_chunk = 0;
        f.mtp_max_new = 0; // request metadata is not valid on a failure frame
        f.shared_checksum = 0;
        f.error = error.empty() ? "final stage failed" : std::move(error);
        if (f.error.size() > 1024) f.error.resize(1024);
        send_frame(fd, f, timeout_ms);
    } catch (...) {} // a disconnected peer is expected after fail-stop
}
} // namespace

void serve_final_stage_worker(int fd, const Plan& plan, size_t index,
                              FinalStageExecutor& executor, int timeout_ms,
                              bool shared_memory, const WindowLimits* limits, int startup_wait_ms, bool group_hip,
                              NativePrefillHandler prefill, DirectSharedBinder direct_bind, bool guarded_direct) {
    plan.validate();
    if (index + 1 != plan.stages.size() || timeout_ms < 0)
        throw std::invalid_argument("final worker must own the last stage and a valid deadline");
    if (guarded_direct && !direct_bind) throw std::invalid_argument("guarded consumer needs registered slot");
    const bool mtp_service = executor.mtp_service_enabled();
    Frame hello = control(plan, index,
        direct_bind ? (guarded_direct ? Kind::hello_guarded_direct_final : Kind::hello_direct_shared_final) :
        executor.mtp_service_enabled() ? Kind::hello_mtp_service_final :
        shared_memory ? Kind::hello_shared_final : Kind::hello_final);
    if (group_hip) {
        const auto d = worker_domains(plan, true).back();
        const auto& front = plan.stages[d.first_stage];
        hello.identity = {static_cast<uint32_t>(front.backend), front.ordinal, front.first_layer,
                          plan.stages[d.terminal_stage()].end_layer};
    }
    std::unique_ptr<WindowAdmission> gate;
    if (limits) gate = std::make_unique<WindowAdmission>(plan, index, *limits);
    std::string error;
    SharedActivation shared;
    if (direct_bind) {
        if (!shared_memory || !limits || !group_hip || !direct_verified_plan(plan) || prefill ||
            executor.receiver_arm_enabled() != guarded_direct || executor.mtp_relay_enabled())
            throw std::invalid_argument("direct consumer requires bounded grouped HIP and matching checked prearm capability");
        shared = SharedActivation::receive(fd, timeout_ms);
        if (!direct_bind(shared, error)) throw std::runtime_error(error);
    }
    if (!executor.prepare(plan, index, error)) {
        failure(fd, hello, error.empty() ? "final stage preparation failed" : error, timeout_ms);
        executor.shutdown();
        if (mtp_service) throw std::runtime_error(error.empty() ? "MTP service preparation failed" : error);
        return;
    }
    try {
        send_frame(fd, hello, timeout_ms);
        auto ready = receive_startup_frame(fd, timeout_ms, startup_wait_ms);
        if (!ready) { executor.shutdown(); return; }
        const bool resident = ready->kind == Kind::ready_resident;
        if (resident && !gate)
            throw std::runtime_error("resident worker requires native window limits");
        match(*ready, hello, resident ? Kind::ready_resident : Kind::ready);
        uint64_t active_request = 0;
        if (shared_memory && !direct_bind) shared = SharedActivation::receive(fd, timeout_ms);
        if (executor.mtp_resident_enabled() && (!resident || !executor.mtp_relay_enabled() || !executor.mtp_anchor_only_enabled()))
            throw std::runtime_error("resident MTP requires explicit resident relay");
        if (executor.mtp_relay_enabled() && ((resident && !executor.mtp_resident_enabled()) || !group_hip || !gate || executor.receiver_arm_enabled()))
            throw std::runtime_error("MTP relay requires explicitly admitted grouped HIP without external arm");
        if (executor.mtp_service_enabled() && (!resident || !shared_memory || !executor.mtp_resident_enabled()))
            throw std::runtime_error("MTP service requires shared resident relay");
        bool mtp_service_configured = false;
        std::optional<Frame> last_mtp_commit;
        bool mtp_requested = false;
        while (auto request = resident ? receive_idle_frame(fd, timeout_ms)
                                       : receive_frame(fd, timeout_ms)) {
            try {
                if (request->kind == Kind::begin_request) {
                    if (!resident || request->request_id <= active_request || (executor.mtp_resident_enabled() && !executor.mtp_service_enabled() && !mtp_resident_next_request(active_request,request->request_id)) ||
                        (executor.mtp_service_enabled() && request->request_id != active_request+1))
                        throw std::runtime_error("unexpected or replayed resident request");
                    const auto expected = control(plan, index, Kind::begin_request,
                                                  request->request_id, 0, 1);
                    match(*request, expected, Kind::begin_request);
                    if (!gate->reset_request(request->request_id, error) ||
                        !executor.reset_request(request->request_id, error))
                        throw std::runtime_error(error.empty() ? "resident reset failed" : error);
                    active_request = request->request_id;
                    last_mtp_commit.reset();
                    mtp_requested=false;
                    mtp_service_configured=false;
                    trace_window(gate.get(), "request_ready");
                    Frame ack = *request;
                    ack.kind = Kind::request_ready;
                    send_frame(fd, ack, timeout_ms);
                    continue;
                }
                if (resident && (!active_request || request->request_id != active_request))
                    throw std::runtime_error("resident window lacks matching request reset");
                if (request->kind == Kind::mtp_service_request) {
                    if (!executor.mtp_service_enabled() || mtp_service_configured || last_mtp_commit ||
                        !request->window || !gate)
                        throw std::runtime_error("unexpected MTP service request metadata");
                    match(*request,control(plan,index,Kind::mtp_service_request,active_request,0,request->rows),Kind::mtp_service_request);
                    if (!executor.configure_mtp_request(*request,error))
                        throw std::runtime_error(error.empty()?"MTP service metadata refused":error);
                    mtp_service_configured=true;
                    request->kind=Kind::mtp_service_ready;
                    send_frame(fd,*request,timeout_ms);
                    continue;
                }
                if (executor.mtp_service_enabled() && !mtp_service_configured)
                    throw std::runtime_error("MTP service requires request-local metadata before execution");
                if (request->kind == Kind::prefill_spec) {
                    if ((executor.mtp_relay_enabled() && !executor.mtp_service_enabled()) || executor.receiver_arm_enabled() || !resident || !gate || !prefill ||
                        !gate->begin_prefill(*request,error) ||
                        !prefill(*request,fd,error) ||
                        !gate->finished_prefill(error))
                        throw std::runtime_error(error.empty() ? "native prefill unavailable or failed" : error);
                    Frame ack;
                    ack.kind=Kind::prefill_done; ack.stage_index=request->stage_index;
                    ack.request_id=request->request_id; ack.window_id=request->window_id;
                    ack.rows=request->rows; ack.plan_fingerprint=request->plan_fingerprint;
                    ack.geometry=request->geometry;
                    send_frame(fd,ack,timeout_ms);
                    continue;
                }
                if (request->kind == Kind::mtp_proposal_request) {
                    if (!executor.mtp_relay_enabled() || !last_mtp_commit || mtp_requested)
                        throw std::runtime_error("MTP proposal request lacks a fresh committed prefix");
                    match(*request,*last_mtp_commit,Kind::mtp_proposal_request);
                    if (!request->window || request->window->pos0 != last_mtp_commit->window->pos0 ||
                        request->window->token_ids != last_mtp_commit->window->token_ids)
                        throw std::runtime_error("MTP proposal anchor differs from committed target");
                    mtp_requested = true;
                    WindowSpec offer;
                    if (!executor.propose_mtp(*request->window,offer,error))
                        throw std::runtime_error(error.empty()?"native MTP proposal failed":error);
                    if (offer.pos0 != request->window->pos0 || offer.token_ids.size()<(executor.mtp_anchor_only_enabled()?1u:2u) ||
                        offer.token_ids.size()>5 || offer.token_ids.front()!=request->window->token_ids.front())
                        throw std::runtime_error("invalid native MTP offer");
                    Frame response=*request;
                    response.kind=Kind::mtp_proposal;
                    response.rows=static_cast<uint32_t>(offer.token_ids.size());
                    response.window=std::move(offer);
                    send_frame(fd,response,timeout_ms);
                    continue;
                }
                last_mtp_commit.reset();
                mtp_requested=false;
                if (request->kind != Kind::window_spec || !request->window)
                    throw std::runtime_error("final worker requires verifier window metadata");
                if (request->stage_index != index ||
                    request->plan_fingerprint != hello.plan_fingerprint ||
                    request->geometry.hc != plan.geometry.hc ||
                    request->geometry.n_embd != plan.geometry.n_embd)
                    throw std::runtime_error("final worker window plan mismatch");
                if (gate && !gate->begin(*request->window, request->request_id,
                                         request->window_id, error))
                    throw std::runtime_error(error);
                trace_window(gate.get(), "admitted");
                if (!executor.begin_window(*request->window,
                                           request->request_id, request->window_id, error))
                    throw std::runtime_error(error.empty() ? "final window admission failed" : error);
                if (executor.receiver_arm_enabled()) {
                    if (!group_hip || !gate || !executor.arm_window(error))
                        throw std::runtime_error(error.empty() ? "receiver arm admission failed" : error);
                    trace_window(gate.get(), "receiver_armed");
                    send_frame(fd, control(plan, index, Kind::window_armed,
                                           request->request_id, request->window_id, request->rows), timeout_ms);
                }
                auto data = receive_frame(fd, timeout_ms);
                if (!data) throw std::runtime_error("final worker closed after window metadata");
                match(*data, *request, shared_memory ? Kind::shared_data : Kind::data);
                if (gate && !gate->validate_input(*data, shared_memory, error))
                    throw std::runtime_error(error);
                const size_t floats = static_cast<size_t>(request->rows) * plan.geometry.floats_per_row();
                const size_t bytes = floats * sizeof(float);
                // encode()/decode() bound the activation window to the 16MiB cap.
                // The decoded data frame is already validated. Borrow its
                // activation below; copy only the identity fields, not the payload.
                Frame metadata;
                metadata.kind = Kind::data;
                metadata.stage_index = data->stage_index;
                metadata.request_id = data->request_id;
                metadata.window_id = data->window_id;
                metadata.plan_fingerprint = data->plan_fingerprint;
                metadata.rows = data->rows;
                metadata.geometry = data->geometry;
                FinalStageResult result;
                if (shared_memory) {
                    if (shared.checksum(bytes) != data->shared_checksum)
                        throw std::runtime_error("final worker shared input checksum mismatch");
                    if (gate && !gate->validate_bytes(shared.data(), bytes, error))
                        throw std::runtime_error(error);
                    ActivationView view(shared.data(), bytes, metadata.geometry, metadata.rows);
                    if (!executor.run_final_window(metadata, view, result, error))
                        throw std::runtime_error(error.empty() ? "final stage execution failed" : error);
                } else {
                    ActivationView view(reinterpret_cast<uint8_t*>(data->payload.data()),
                                        bytes, metadata.geometry, metadata.rows);
                    if (!executor.run_final_window(metadata, view, result, error))
                        throw std::runtime_error(error.empty() ? "final stage execution failed" : error);
                }
                require_final_result_identity(result, static_cast<uint32_t>(index),
                                              request->request_id, request->window_id,
                                              request->plan_fingerprint, request->rows);
                validate_final_result(result);
                if (gate && !gate->provisional_final(error))
                    throw std::runtime_error(error);
                trace_window(gate.get(), "provisional");
                {
                    PerformanceSpan perf("worker", "final_result_send", index,
                                         request->request_id, request->window_id, request->rows);
                    send_final_result(fd, result, timeout_ms);
                }
                auto commit = receive_frame(fd, timeout_ms);
                if (!commit) {
                    if (mtp_service) throw std::runtime_error("MTP service peer closed before prefix commit");
                    executor.abandon_window(); break;
                }
                match(*commit, *request, Kind::commit_prefix);
                if (gate && !gate->check_commit(*commit, error))
                    throw std::runtime_error(error);
                if (!executor.commit_prefix(request->request_id, request->window_id,
                                            commit->keep_count, error))
                    throw std::runtime_error(error.empty() ? "final stage prefix commit failed" : error);
                if (gate && !gate->committed(commit->keep_count, error))
                    throw std::runtime_error(error);
                trace_window(gate.get(), "committed");
                Frame ack = *commit;
                ack.kind = Kind::committed_prefix;
                send_frame(fd, ack, timeout_ms);
                if (executor.mtp_relay_enabled() && commit->keep_count) {
                    last_mtp_commit=control(plan,index,Kind::mtp_proposal_request,request->request_id,request->window_id,1);
                    last_mtp_commit->window=WindowSpec{request->window->pos0+commit->keep_count,
                        {result.predicted_tokens.at(commit->keep_count-1)}};
                }
            } catch (const std::exception& ex) {
                trace_window(gate.get(), "failure");
                if (gate) gate->poison();
                executor.abandon_window();
                failure(fd, *request, ex.what(), timeout_ms);
                // The outer catch drains the executor; a failed MTP domain
                // must exit nonzero, not masquerade as a cooperative close.
                if (mtp_service) throw;
                executor.shutdown();
                return;
            }
        }
    } catch (...) {
        if (gate) gate->poison();
        executor.abandon_window();
        executor.shutdown();
        throw;
    }
    executor.shutdown();
}

FinalPipelineSession::FinalPipelineSession(Plan plan, std::vector<int> fds,
                                           int timeout_ms, bool shared_memory, bool resident,
                                           bool fanout_control, bool require_compact, bool group_hip, bool receiver_arm, bool mtp_relay, bool mtp_anchor_only, bool mtp_resident, bool direct_verified, bool early_window_spec, bool guarded_direct, bool mtp_service)
    : plan_(std::move(plan)), fds_(std::move(fds)), timeout_ms_(timeout_ms),
      shared_memory_(shared_memory), resident_(resident),
      fanout_control_(fanout_control), require_compact_(require_compact), receiver_arm_(receiver_arm), mtp_relay_(mtp_relay), mtp_anchor_only_(mtp_anchor_only), mtp_resident_(mtp_resident), direct_verified_(direct_verified), early_window_spec_(early_window_spec), guarded_direct_(guarded_direct), mtp_service_(mtp_service) {
    for (const auto& domain : worker_domains(plan_, group_hip)) {
        starts_.push_back(domain.first_stage);
        endpoints_.push_back(domain.terminal_stage());
    }
    if (fds_.size() != endpoints_.size() || timeout_ms_ < 0)
        throw std::invalid_argument("final pipeline needs one worker per domain and valid timeout");
    if (direct_verified_ && (!group_hip || !shared_memory_ || !direct_verified_plan(plan_) ||
        receiver_arm_ || mtp_relay_ || require_compact_ || fds_.size() != 2 || !timeout_ms_))
        throw std::invalid_argument("direct verified mode requires two bounded mixed domains, shared and full logits");
    if (guarded_direct_ && !direct_verified_) throw std::invalid_argument("guarded direct requires direct verified transport");
    if (receiver_arm_ && (!group_hip || fds_.size() != 2 || !timeout_ms_))
        throw std::invalid_argument("receiver arm requires two backend domains and a bounded deadline");
    if (early_window_spec_ && (!resident_ || !group_hip || !shared_memory_ ||
        fds_.size() != 2 || receiver_arm_ || (mtp_relay_ && !mtp_service_) || direct_verified_ || !timeout_ms_))
        throw std::invalid_argument("early window metadata requires grouped mixed resident shared transport");
    const bool local_mtp_domain = mtp_service_ && group_hip && fds_.size()==1 &&
        plan_.stages.size()==2 && plan_.stages.front().backend==Backend::hip &&
        plan_.stages.back().backend==Backend::hip;
    if (mtp_service_ && (!mtp_resident_ || !shared_memory_ || !mtp_relay_ || !mtp_anchor_only_ ||
                         (fds_.size()!=2 && !local_mtp_domain) || !group_hip || receiver_arm_ || direct_verified_))
        throw std::invalid_argument("MTP service requires grouped resident shared HIP pair with optional CUDA front");
    if (mtp_anchor_only_ && !mtp_relay_) throw std::invalid_argument("anchor-only MTP requires relay");
    if (mtp_resident_ && (!resident_ || !mtp_relay_ || !mtp_anchor_only_))
        throw std::invalid_argument("resident MTP requires explicit resident and anchor-only relay");
    if (mtp_relay_ && (!group_hip || (resident_ && !mtp_resident_) || receiver_arm_ || (require_compact_ && !mtp_service_) || (fds_.size()!=2 && !local_mtp_domain) || !timeout_ms_))
        throw std::invalid_argument("MTP relay requires explicitly admitted grouped HIP; diagnostic relay requires full logits");
}
FinalPipelineSession::~FinalPipelineSession() = default;
void FinalPipelineSession::poison() noexcept {
    poisoned_ = true;
    pending_ = false;
    gate_.poison();
    for (int fd : fds_) if (fd >= 0) (void)::shutdown(fd, SHUT_RDWR);
}
void FinalPipelineSession::initialize() {
    if (initialized_ || poisoned_) throw std::logic_error("final pipeline already initialized or poisoned");
    try {
        // One bounded readiness barrier across all peers, not a fresh load
        // budget for every worker. Workers wait separately for READY.
        using Clock = std::chrono::steady_clock;
        const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms_);
        if (direct_verified_) {
            // Mode-specific pre-HELLO descriptor transfer. The ordinary
            // shared slot path and its handshake remain unchanged.
            shared_ = std::make_unique<SharedActivation>(SharedActivation::create());
            for (int fd : fds_) shared_->send(fd, timeout_ms_);
        }
        for (size_t i = 0; i < fds_.size(); ++i) {
            const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
            if (remaining < 0) throw std::runtime_error("final pipeline startup deadline expired");
            auto hello = receive_frame(fds_[i], static_cast<int>(remaining));
            if (!hello) throw std::runtime_error("worker closed before final pipeline hello");
            if (hello->kind == Kind::failure) throw std::runtime_error("worker preparation: " + hello->error);
            const bool last = i + 1 == fds_.size();
            const Kind kind = guarded_direct_ ? (last ? Kind::hello_guarded_direct_final : Kind::hello_guarded_direct) :
                          direct_verified_ ? (last ? Kind::hello_direct_shared_final : Kind::hello_direct_shared)
                          : last ? (mtp_service_ ? Kind::hello_mtp_service_final : shared_memory_ ? Kind::hello_shared_final : Kind::hello_final)
                                 : (shared_memory_ ? Kind::hello_shared : Kind::hello);
            const Frame expected = control(plan_, endpoints_[i], kind);
            match(*hello, expected, kind);
            if (hello->identity != (std::array<uint32_t,4>{
                static_cast<uint32_t>(plan_.stages[starts_[i]].backend), plan_.stages[starts_[i]].ordinal,
                plan_.stages[starts_[i]].first_layer, plan_.stages[endpoints_[i]].end_layer}))
                throw std::runtime_error("worker backend, ordinal or stage bounds mismatch");
        }
        if (shared_memory_ && !direct_verified_) shared_ = std::make_unique<SharedActivation>(SharedActivation::create());
        for (size_t i = 0; i < fds_.size(); ++i) {
            send_frame(fds_[i], control(plan_, endpoints_[i], resident_ ? Kind::ready_resident : Kind::ready), timeout_ms_);
            if (shared_memory_ && !direct_verified_) shared_->send(fds_[i], timeout_ms_);
        }
        initialized_ = true;
    } catch (...) { poison(); throw; }
}
void FinalPipelineSession::begin_request(uint64_t request) {
    if (!initialized_ || poisoned_ || pending_ || !resident_ || !request ||
        request <= resident_request_ || (mtp_resident_ && !mtp_service_ && !mtp_resident_next_request(resident_request_,request)) ||
        (mtp_service_ && request != resident_request_+1))
        throw std::logic_error("resident reset requires an idle pipeline and increasing request ID");
    try {
        PerformanceSpan total("coordinator", "reset_barrier", fds_.size(), request, 0, 1);
        // Stage-local state resets can overlap. Wait for every ACK.
        if (fanout_control_) {
            for (size_t i = 0; i < fds_.size(); ++i)
                send_frame(fds_[i], control(plan_, endpoints_[i], Kind::begin_request, request, 0, 1), timeout_ms_);
        }
        for (size_t i = 0; i < fds_.size(); ++i) {
            const auto reset = control(plan_, endpoints_[i], Kind::begin_request, request, 0, 1);
            PerformanceSpan stage("coordinator", "reset_ack", endpoints_[i], request, 0, 1);
            if (!fanout_control_) send_frame(fds_[i], reset, timeout_ms_);
            auto ack = receive_frame(fds_[i], timeout_ms_);
            if (!ack) throw std::runtime_error("worker closed during resident reset");
            if (ack->kind == Kind::failure)
                throw std::runtime_error("worker resident reset failed: " + ack->error);
            match(*ack, reset, Kind::request_ready);
        }
        resident_request_ = request;
        last_request_ = request;
        last_window_ = 0;
        mtp_anchor_.reset(); mtp_offer_.reset(); mtp_requested_=false;
        mtp_pending_inputs_.clear(); mtp_vocab_=0;
        mtp_service_policy_.reset();
    } catch (...) { poison(); throw; }
}

void FinalPipelineSession::configure_mtp_request(uint64_t request, const MtpRequestPolicy& policy) {
    if (!mtp_service_ || !initialized_ || poisoned_ || pending_ || last_window_ ||
        !request || request != resident_request_ || mtp_service_policy_)
        throw std::logic_error("MTP service requires a valid new request after all reset ACKs");
    auto admitted=admit_mtp_service_request(policy);
    if (!admitted) throw std::logic_error("MTP service requires a valid new request after all reset ACKs");
    try {
        Frame meta = control(plan_,endpoints_.back(),Kind::mtp_service_request,request,0,static_cast<uint32_t>(admitted->prompt.size()));
        meta.window=WindowSpec{0,admitted->prompt};
        meta.mtp_max_new=admitted->max_new;
        send_frame(fds_.back(),meta,timeout_ms_);
        auto ack=receive_frame(fds_.back(),timeout_ms_);
        if (!ack) throw std::runtime_error("terminal closed during MTP service metadata");
        if (ack->kind==Kind::failure) throw std::runtime_error("terminal MTP service metadata: "+ack->error);
        match(*ack,meta,Kind::mtp_service_ready);
        if (!ack->window || ack->window->pos0!=0 || ack->window->token_ids!=admitted->prompt || ack->mtp_max_new!=admitted->max_new)
            throw std::runtime_error("MTP service metadata ACK mismatch");
        mtp_service_policy_=std::move(admitted);
    } catch (...) { poison(); throw; }
}

void FinalPipelineSession::native_prefill(const std::vector<int32_t>& tokens, uint32_t chunk, bool isolated_stage_reference, int total_timeout_ms) {
    if (total_timeout_ms < 0 || total_timeout_ms > long_prefill_timeout_ms ||
        (total_timeout_ms && isolated_stage_reference))
        throw std::invalid_argument("invalid extended prefill deadline");
    if (receiver_arm_ || (mtp_relay_ && !mtp_service_) || direct_verified_)
        throw std::logic_error("receiver arm/legacy MTP relay/direct verified mode does not support native bulk prefill");
    if (mtp_service_ && (isolated_stage_reference || !mtp_service_policy_ ||
        (MtpRequestPolicy::requires_long_context(mtp_service_policy_->context) != bool(total_timeout_ms)) ||
        !mtp_service_policy_->bulk_prefix(tokens,chunk)))
        throw std::invalid_argument("MTP bulk prefill requires exact bounded request prefix");
    const bool local = endpoints_.size()==1 && starts_.front()==0 &&
        endpoints_.front()==1 && plan_.stages.size()==2 &&
        plan_.stages[0].backend==Backend::hip && plan_.stages[1].backend==Backend::hip;
    const bool bridge = endpoints_.size()==2 && starts_.back()!=endpoints_.back();
    if (!initialized_ || poisoned_ || pending_ || !resident_ || !resident_request_ || last_window_ ||
        (!local && !bridge) ||
        tokens.empty() || tokens.size()>max_native_prefill_tokens || !chunk || chunk>max_native_prefill_chunk)
        throw std::logic_error("native prefill requires freshly reset grouped resident pipeline");
    if (isolated_stage_reference && (resident_request_!=2 || tokens.size()>16 || chunk>tokens.size() ||
        plan_.stages.size()!=3 || plan_.total_layers!=48 ||
        plan_.stages[0].backend!=Backend::cuda || plan_.stages[0].ordinal!=0 ||
        plan_.stages[0].first_layer!=0 || plan_.stages[0].end_layer!=12 ||
        plan_.stages[1].backend!=Backend::hip || plan_.stages[1].ordinal!=0 ||
        plan_.stages[1].first_layer!=12 || plan_.stages[1].end_layer!=30 ||
        plan_.stages[2].backend!=Backend::hip || plan_.stages[2].ordinal!=1 ||
        plan_.stages[2].first_layer!=30 || plan_.stages[2].end_layer!=48 ||
        plan_.geometry.hc!=4 || plan_.geometry.n_embd!=2560))
        throw std::logic_error("isolated stage reference requires bounded A/B/A diagnostic");
    using Clock = std::chrono::steady_clock;
    const auto deadline = Clock::now() + std::chrono::milliseconds(total_timeout_ms);
    auto prefill_wait = [&](bool full_local = false) {
        if (!total_timeout_ms) return timeout_ms_;
        const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline-Clock::now()).count();
        if (remaining <= 0) throw std::runtime_error("native prefill total deadline exceeded");
        return full_local ? static_cast<int>(remaining) : std::min(timeout_ms_, static_cast<int>(remaining));
    };
    try {
        Frame spec=control(plan_,endpoints_[0],Kind::prefill_spec,resident_request_,1,static_cast<uint32_t>(tokens.size()));
        spec.window=WindowSpec{0,tokens}; spec.prefill_chunk=chunk;
        validate_wire_frame(spec);
        if (bridge) {
            auto receive_spec=spec; receive_spec.stage_index=static_cast<uint32_t>(endpoints_[1]);
            send_frame(fds_[1],receive_spec,prefill_wait());
        }
        // A complete local HIP domain consumes the token spec once. No host R relay.
        send_frame(fds_[0],spec,prefill_wait());
        const uint32_t cap=prefill_piece_rows(plan_.geometry);
        size_t offset=0;
        while (bridge && !isolated_stage_reference && offset<tokens.size()) {
            const size_t count=std::min({size_t(cap),size_t(chunk)-offset%chunk,tokens.size()-offset});
            auto piece=receive_frame(fds_[0],prefill_wait());
            if (!piece || piece->kind==Kind::failure)
                throw std::runtime_error(piece ? piece->error : "CUDA closed during native prefill");
            if (piece->kind!=Kind::prefill_rows || piece->stage_index!=endpoints_[0] ||
                piece->request_id!=resident_request_ || piece->plan_fingerprint!=spec.plan_fingerprint ||
                piece->window_id!=offset+1 || piece->rows!=count ||
                piece->geometry.hc!=plan_.geometry.hc || piece->geometry.n_embd!=plan_.geometry.n_embd ||
                !activation_finite(reinterpret_cast<const uint8_t*>(piece->payload.data()),piece->payload.size()*sizeof(float)))
                throw std::runtime_error("native prefill residual identity, order or finite gate failed");
            piece->stage_index=static_cast<uint32_t>(endpoints_[1]);
            send_frame(fds_[1],*piece,prefill_wait());
            offset+=count;
        }
        for (size_t i=0;i<fds_.size();++i) {
            auto done=receive_frame(fds_[i],prefill_wait(local));
            if (!done || done->kind==Kind::failure)
                throw std::runtime_error(done ? done->error : "native prefill peer closed before drain");
            const auto expected=control(plan_,endpoints_[i],Kind::prefill_done,resident_request_,1,spec.rows);
            match(*done,expected,Kind::prefill_done);
        }
        last_window_=1;
    } catch (...) { poison(); throw; }
}

FinalStageResult FinalPipelineSession::execute(Frame initial, const WindowSpec& spec) {
    execute_discard(std::move(initial), spec);
    try { return gate_.provisional(); }
    catch (...) { poison(); throw; }
}
void FinalPipelineSession::execute_discard(Frame initial, const WindowSpec& spec) {
    if (!initialized_ || poisoned_) throw std::logic_error("final pipeline unavailable");
    if (pending_) throw std::logic_error("final pipeline already has uncommitted result");
    if (resident_ && (!resident_request_ || initial.request_id != resident_request_))
        throw std::logic_error("resident request must be acknowledged before execute");
    try {
        if (initial.kind != Kind::data || !initial.request_id || !initial.window_id ||
            initial.plan_fingerprint != plan_.fingerprint() ||
            initial.geometry.hc != plan_.geometry.hc ||
            initial.geometry.n_embd != plan_.geometry.n_embd)
            throw std::invalid_argument("initial final pipeline activation identity/shape invalid");
        // Same validation as encode, without serializing and hashing a throwaway packet.
        validate_wire_frame(initial);
        spec.validate(initial.rows);
        if (direct_verified_ && initial.rows > 8)
            throw std::invalid_argument("direct verified slot supports at most eight rows");
        if (mtp_service_) {
            if (!mtp_service_policy_) throw std::logic_error("MTP service request metadata missing");
            const auto& policy=*mtp_service_policy_;
            if (spec.pos0 < int64_t(policy.prompt.size())) {
                if (!policy.prompt_window(spec.pos0,spec.token_ids)) throw std::invalid_argument("MTP service prompt mismatch");
            } else if (!mtp_anchor_ || mtp_anchor_->pos0!=spec.pos0 ||
                       !policy.decode_window(spec.pos0,mtp_anchor_->token_ids.front(),spec.token_ids) ||
                       (!mtp_offer_ && (spec.token_ids.size()!=1 || policy.can_offer(spec.pos0,spec.token_ids.front()))))
                throw std::invalid_argument("MTP service decode window mismatch");
        }
        if (mtp_offer_ && (initial.request_id!=last_request_ || initial.window_id!=last_window_+1 ||
                          spec.pos0!=mtp_offer_->pos0 || spec.token_ids!=mtp_offer_->token_ids))
            throw std::invalid_argument("next window must consume exact MTP offer");
        mtp_pending_inputs_.clear();
        if (mtp_offer_) mtp_pending_inputs_=spec.token_ids;
        mtp_offer_.reset();
        mtp_anchor_.reset();
        if (initial.request_id == last_request_ && initial.window_id <= last_window_)
            throw std::invalid_argument("final pipeline received replayed window");
        const size_t floats = static_cast<size_t>(initial.rows) * plan_.geometry.floats_per_row();
        const size_t bytes = floats * sizeof(float);
        if (shared_memory_) {
            shared_->store(initial.payload);
            initial.payload.clear();
            initial.kind = Kind::shared_data;
            initial.shared_checksum = shared_->checksum(bytes);
        }
        if (early_window_spec_) {
            // Admit downstream metadata while the producer is executing. It is
            // not an activation or a GPU ready signal: the final worker still
            // waits for a matching data frame and verifies checksum/finiteness.
            // On any producer failure, poison() closes both sockets. Do not
            // arm a GPU wait or read the shared slot before normal publication.
            PerformanceSpan early("coordinator", "stage_predispatch", endpoints_.back(),
                                  initial.request_id, initial.window_id, initial.rows);
            Frame downstream = control(plan_, endpoints_.back(), Kind::window_spec,
                                       initial.request_id, initial.window_id, initial.rows);
            downstream.window = spec;
            send_frame(fds_.back(), downstream, timeout_ms_);
        }
        if (receiver_arm_ || guarded_direct_) {
            PerformanceSpan arm("coordinator", "receiver_arm_ack", endpoints_.back(), initial.request_id,
                                initial.window_id, initial.rows);
            // No producer launch before a full-identity ACK from the armed consumer.
            Frame context = control(plan_, endpoints_.back(), Kind::window_spec,
                                    initial.request_id, initial.window_id, initial.rows);
            context.window = spec;
            send_frame(fds_.back(), context, timeout_ms_);
            const auto ack = receive_frame(fds_.back(), timeout_ms_);
            if (!ack) throw std::runtime_error("receiver closed before arm acknowledgement");
            if (ack->kind == Kind::failure) throw std::runtime_error("receiver arm failed: " + ack->error);
            match(*ack, context, Kind::window_armed);
        }
        for (size_t i = 0; i < fds_.size(); ++i) {
            PerformanceSpan stage("coordinator", "stage_roundtrip", endpoints_[i], initial.request_id,
                                  initial.window_id, initial.rows);
            Frame context = control(plan_, endpoints_[i], Kind::window_spec,
                                    initial.request_id, initial.window_id, initial.rows);
            context.window = spec;
            {
                PerformanceSpan send("coordinator", "stage_send_spec", endpoints_[i], initial.request_id,
                                     initial.window_id, initial.rows);
                if ((!receiver_arm_ && !early_window_spec_ && !guarded_direct_) || i + 1 != fds_.size())
                    send_frame(fds_[i], context, timeout_ms_);
            }
            initial.stage_index = static_cast<uint32_t>(endpoints_[i]);
            {
                PerformanceSpan send("coordinator", "stage_send_activation", endpoints_[i], initial.request_id,
                                     initial.window_id, initial.rows);
                send_frame(fds_[i], initial, timeout_ms_);
            }
            if (i + 1 == fds_.size()) {
                auto response = [&] {
                    PerformanceSpan recv("coordinator", "stage_receive", endpoints_[i], initial.request_id,
                                         initial.window_id, initial.rows);
                    return receive_final_result(fds_[i], timeout_ms_);
                }();
                if (!response) throw std::runtime_error("final worker closed without head result");
                if (require_compact_ && response->logits)
                    throw std::runtime_error("requested compact result, but worker returned diagnostic logits");
                gate_.stage(std::move(*response), static_cast<uint32_t>(endpoints_[i]),
                            initial.request_id, initial.window_id, initial.plan_fingerprint, initial.rows);
            } else {
                auto response = [&] {
                    PerformanceSpan recv("coordinator", "stage_receive", endpoints_[i], initial.request_id,
                                         initial.window_id, initial.rows);
                    return receive_frame(fds_[i], timeout_ms_);
                }();
                if (!response) throw std::runtime_error("intermediate worker closed before handoff");
                if (response->kind == Kind::failure)
                    throw std::runtime_error("intermediate worker failed: " + response->error);
                match(*response, initial, shared_memory_ ? Kind::shared_data : Kind::data);
                if (shared_memory_ && response->shared_checksum != shared_->checksum(bytes))
                    throw std::runtime_error("intermediate worker output checksum mismatch");
                initial = std::move(*response);
            }
        }
        pending_request_ = initial.request_id;
        pending_window_ = initial.window_id;
        pending_rows_ = initial.rows;
        pending_pos_ = spec.pos0;
        pending_ = true;
    } catch (...) { poison(); throw; }
}
FinalStageResult FinalPipelineSession::commit(uint32_t n_keep) {
    if (!initialized_ || poisoned_ || !pending_)
        throw std::logic_error("no provisional final-stage result to commit");
    if (n_keep > pending_rows_) throw std::invalid_argument("accepted prefix exceeds row count");
    try {
        if (gate_.provisional().has_nonfinite())
            throw std::runtime_error("nonfinite head result cannot be committed");
        if (mtp_service_policy_ && !n_keep)
            throw std::invalid_argument("MTP service requires a positive accepted prefix");
        if (mtp_service_policy_ && pending_pos_ < int64_t(mtp_service_policy_->prompt.size()) && n_keep!=pending_rows_)
            throw std::invalid_argument("MTP service prompt requires full commit");
        if (!mtp_pending_inputs_.empty() && n_keep!=greedy_accepted_prefix(mtp_pending_inputs_,gate_.provisional().predicted_tokens))
            throw std::invalid_argument("MTP commit differs from target-authoritative greedy prefix");
        PerformanceSpan total("coordinator", "commit_barrier", fds_.size(), pending_request_,
                              pending_window_, pending_rows_);
        // Only after the finite gate passes. Partial failure still poisons
        // all peers. No result is published before every valid ACK.
        if (fanout_control_) {
            for (size_t i = 0; i < fds_.size(); ++i) {
                Frame request = control(plan_, endpoints_[i], Kind::commit_prefix,
                                        pending_request_, pending_window_, pending_rows_);
                request.keep_count = n_keep;
                send_frame(fds_[i], request, timeout_ms_);
            }
        }
        for (size_t i = 0; i < fds_.size(); ++i) {
            Frame request = control(plan_, endpoints_[i], Kind::commit_prefix,
                                    pending_request_, pending_window_, pending_rows_);
            request.keep_count = n_keep;
            PerformanceSpan stage("coordinator", "commit_ack", endpoints_[i], pending_request_,
                                  pending_window_, pending_rows_);
            if (!fanout_control_) send_frame(fds_[i], request, timeout_ms_);
            auto reply = receive_frame(fds_[i], timeout_ms_);
            if (!reply) throw std::runtime_error("worker closed during final-result commit");
            if (reply->kind == Kind::failure)
                throw std::runtime_error("worker prefix commit failed: " + reply->error);
            match(*reply, request, Kind::committed_prefix);
            if (reply->keep_count != n_keep)
                throw std::runtime_error("accepted-prefix count in worker ACK differs from requested count");
        }
        gate_.mark_committed(pending_request_, pending_window_);
        FinalStageResult output = gate_.take_committed();
        last_request_ = pending_request_;
        last_window_ = pending_window_;
        mtp_requested_=false;
        if (mtp_relay_ && n_keep) {
            mtp_anchor_=WindowSpec{pending_pos_+n_keep,{output.predicted_tokens.at(n_keep-1)}};
            mtp_vocab_=output.vocab_size;
        }
        pending_ = false;
        return output;
    } catch (...) { poison(); throw; }
}
WindowSpec FinalPipelineSession::request_mtp_proposal() {
    if (!mtp_relay_ || !initialized_ || poisoned_ || pending_ || (resident_ && (!mtp_resident_ || !resident_request_)) ||
        (fds_.size()!=2 && !(mtp_service_ && fds_.size()==1)) ||
        !mtp_anchor_ || mtp_requested_ || mtp_offer_)
        throw std::logic_error("MTP relay requires fresh all-stage committed prefix");
    try {
        if (mtp_service_ && (!mtp_service_policy_ || !mtp_service_policy_->can_offer(mtp_anchor_->pos0,mtp_anchor_->token_ids.front())))
            throw std::invalid_argument("MTP service proposal outside safe request budget");
        mtp_requested_=true;
        Frame request=control(plan_,endpoints_.back(),Kind::mtp_proposal_request,last_request_,last_window_,1);
        request.window=*mtp_anchor_;
        send_frame(fds_.back(),request,timeout_ms_);
        auto reply=receive_frame(fds_.back(),timeout_ms_);
        if (!reply) throw std::runtime_error("terminal closed during MTP proposal");
        if (reply->kind==Kind::failure) throw std::runtime_error("MTP proposal failed: "+reply->error);
        request.rows=reply->rows;
        match(*reply,request,Kind::mtp_proposal);
        if (!reply->window || reply->window->pos0!=mtp_anchor_->pos0 ||
            reply->window->token_ids.size()<(mtp_anchor_only_?1u:2u) || reply->window->token_ids.size()>5 ||
            reply->window->token_ids.front()!=mtp_anchor_->token_ids.front() ||
            std::any_of(reply->window->token_ids.begin(),reply->window->token_ids.end(),
                [this](int32_t id){return id<0 || static_cast<uint32_t>(id)>=mtp_vocab_;}))
            throw std::runtime_error("MTP proposal invalid anchor, position or vocabulary");
        if (mtp_service_policy_ && !mtp_service_policy_->decode_window(reply->window->pos0,mtp_anchor_->token_ids.front(),reply->window->token_ids))
            throw std::runtime_error("MTP service proposal exceeds stop or completion budget");
        mtp_offer_=*reply->window;
        return *mtp_offer_;
    } catch (...) { poison(); throw; }
}
} // namespace strata::heterogeneous
