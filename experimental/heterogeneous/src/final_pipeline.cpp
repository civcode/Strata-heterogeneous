#include "strata_heterogeneous/final_pipeline.hpp"
#include "strata_heterogeneous/shared.hpp"
#include "strata_heterogeneous/performance.hpp"
#include "strata_heterogeneous/window_admission.hpp"

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
    if (kind == Kind::hello_final || kind == Kind::hello_shared_final) {
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
        f.shared_checksum = 0;
        f.error = error.empty() ? "final stage failed" : std::move(error);
        if (f.error.size() > 1024) f.error.resize(1024);
        send_frame(fd, f, timeout_ms);
    } catch (...) {} // a disconnected peer is expected after fail-stop
}
} // namespace

void serve_final_stage_worker(int fd, const Plan& plan, size_t index,
                              FinalStageExecutor& executor, int timeout_ms,
                              bool shared_memory, const WindowLimits* limits, int startup_wait_ms) {
    plan.validate();
    if (index + 1 != plan.stages.size() || timeout_ms < 0)
        throw std::invalid_argument("final worker must own the last stage and a valid deadline");
    const Frame hello = control(plan, index,
        shared_memory ? Kind::hello_shared_final : Kind::hello_final);
    std::unique_ptr<WindowAdmission> gate;
    if (limits) gate = std::make_unique<WindowAdmission>(plan, index, *limits);
    std::string error;
    if (!executor.prepare(plan, index, error)) {
        failure(fd, hello, error.empty() ? "final stage preparation failed" : error, timeout_ms);
        executor.shutdown();
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
        SharedActivation shared;
        if (shared_memory) shared = SharedActivation::receive(fd, timeout_ms);
        while (auto request = resident ? receive_idle_frame(fd, timeout_ms)
                                       : receive_frame(fd, timeout_ms)) {
            try {
                if (request->kind == Kind::begin_request) {
                    if (!resident || request->request_id <= active_request)
                        throw std::runtime_error("unexpected or replayed resident request");
                    const auto expected = control(plan, index, Kind::begin_request,
                                                  request->request_id, 0, 1);
                    match(*request, expected, Kind::begin_request);
                    if (!gate->reset_request(request->request_id, error) ||
                        !executor.reset_request(request->request_id, error))
                        throw std::runtime_error(error.empty() ? "resident reset failed" : error);
                    active_request = request->request_id;
                    trace_window(gate.get(), "request_ready");
                    Frame ack = *request;
                    ack.kind = Kind::request_ready;
                    send_frame(fd, ack, timeout_ms);
                    continue;
                }
                if (resident && (!active_request || request->request_id != active_request))
                    throw std::runtime_error("resident window lacks matching request reset");
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
                if (!commit) { executor.abandon_window(); break; }
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
            } catch (const std::exception& ex) {
                trace_window(gate.get(), "failure");
                if (gate) gate->poison();
                executor.abandon_window();
                failure(fd, *request, ex.what(), timeout_ms);
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
                                           bool fanout_control, bool require_compact)
    : plan_(std::move(plan)), fds_(std::move(fds)), timeout_ms_(timeout_ms),
      shared_memory_(shared_memory), resident_(resident),
      fanout_control_(fanout_control), require_compact_(require_compact) {
    plan_.validate();
    if (fds_.size() != plan_.stages.size() || timeout_ms_ < 0)
        throw std::invalid_argument("final pipeline needs one worker per stage and valid timeout");
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
        for (size_t i = 0; i < fds_.size(); ++i) {
            const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
            if (remaining < 0) throw std::runtime_error("final pipeline startup deadline expired");
            auto hello = receive_frame(fds_[i], static_cast<int>(remaining));
            if (!hello) throw std::runtime_error("worker closed before final pipeline hello");
            if (hello->kind == Kind::failure) throw std::runtime_error("worker preparation: " + hello->error);
            const bool last = i + 1 == fds_.size();
            const Kind kind = last ? (shared_memory_ ? Kind::hello_shared_final : Kind::hello_final)
                                   : (shared_memory_ ? Kind::hello_shared : Kind::hello);
            const Frame expected = control(plan_, i, kind);
            match(*hello, expected, kind);
            if (hello->identity != (std::array<uint32_t,4>{
                static_cast<uint32_t>(plan_.stages[i].backend), plan_.stages[i].ordinal,
                plan_.stages[i].first_layer, plan_.stages[i].end_layer}))
                throw std::runtime_error("worker backend, ordinal or stage bounds mismatch");
        }
        if (shared_memory_) shared_ = std::make_unique<SharedActivation>(SharedActivation::create());
        for (size_t i = 0; i < fds_.size(); ++i) {
            send_frame(fds_[i], control(plan_, i, resident_ ? Kind::ready_resident : Kind::ready), timeout_ms_);
            if (shared_memory_) shared_->send(fds_[i], timeout_ms_);
        }
        initialized_ = true;
    } catch (...) { poison(); throw; }
}
void FinalPipelineSession::begin_request(uint64_t request) {
    if (!initialized_ || poisoned_ || pending_ || !resident_ || !request ||
        request <= resident_request_)
        throw std::logic_error("resident reset requires an idle pipeline and increasing request ID");
    try {
        PerformanceSpan total("coordinator", "reset_barrier", fds_.size(), request, 0, 1);
        // Stage-local state resets can overlap. Wait for every ACK.
        if (fanout_control_) {
            for (size_t i = 0; i < fds_.size(); ++i)
                send_frame(fds_[i], control(plan_, i, Kind::begin_request, request, 0, 1), timeout_ms_);
        }
        for (size_t i = 0; i < fds_.size(); ++i) {
            const auto reset = control(plan_, i, Kind::begin_request, request, 0, 1);
            PerformanceSpan stage("coordinator", "reset_ack", i, request, 0, 1);
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
        for (size_t i = 0; i < fds_.size(); ++i) {
            PerformanceSpan stage("coordinator", "stage_roundtrip", i, initial.request_id,
                                  initial.window_id, initial.rows);
            Frame context = control(plan_, i, Kind::window_spec,
                                    initial.request_id, initial.window_id, initial.rows);
            context.window = spec;
            {
                PerformanceSpan send("coordinator", "stage_send_spec", i, initial.request_id,
                                     initial.window_id, initial.rows);
                send_frame(fds_[i], context, timeout_ms_);
            }
            initial.stage_index = static_cast<uint32_t>(i);
            {
                PerformanceSpan send("coordinator", "stage_send_activation", i, initial.request_id,
                                     initial.window_id, initial.rows);
                send_frame(fds_[i], initial, timeout_ms_);
            }
            if (i + 1 == fds_.size()) {
                auto response = [&] {
                    PerformanceSpan recv("coordinator", "stage_receive", i, initial.request_id,
                                         initial.window_id, initial.rows);
                    return receive_final_result(fds_[i], timeout_ms_);
                }();
                if (!response) throw std::runtime_error("final worker closed without head result");
                if (require_compact_ && response->logits)
                    throw std::runtime_error("requested compact result, but worker returned diagnostic logits");
                gate_.stage(std::move(*response), static_cast<uint32_t>(i),
                            initial.request_id, initial.window_id, initial.plan_fingerprint, initial.rows);
            } else {
                auto response = [&] {
                    PerformanceSpan recv("coordinator", "stage_receive", i, initial.request_id,
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
        PerformanceSpan total("coordinator", "commit_barrier", fds_.size(), pending_request_,
                              pending_window_, pending_rows_);
        // Only after the finite gate passes. Partial failure still poisons
        // all peers. No result is published before every valid ACK.
        if (fanout_control_) {
            for (size_t i = 0; i < fds_.size(); ++i) {
                Frame request = control(plan_, i, Kind::commit_prefix,
                                        pending_request_, pending_window_, pending_rows_);
                request.keep_count = n_keep;
                send_frame(fds_[i], request, timeout_ms_);
            }
        }
        for (size_t i = 0; i < fds_.size(); ++i) {
            Frame request = control(plan_, i, Kind::commit_prefix,
                                    pending_request_, pending_window_, pending_rows_);
            request.keep_count = n_keep;
            PerformanceSpan stage("coordinator", "commit_ack", i, pending_request_,
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
        pending_ = false;
        return output;
    } catch (...) { poison(); throw; }
}
} // namespace strata::heterogeneous
