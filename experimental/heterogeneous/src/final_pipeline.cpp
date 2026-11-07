#include "strata_heterogeneous/final_pipeline.hpp"
#include "strata_heterogeneous/shared.hpp"

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
                              bool shared_memory) {
    plan.validate();
    if (index + 1 != plan.stages.size() || timeout_ms < 0)
        throw std::invalid_argument("final worker must own the last stage and a valid deadline");
    const Frame hello = control(plan, index,
        shared_memory ? Kind::hello_shared_final : Kind::hello_final);
    std::string error;
    if (!executor.prepare(plan, index, error)) {
        failure(fd, hello, error.empty() ? "final stage preparation failed" : error, timeout_ms);
        executor.shutdown();
        return;
    }
    try {
        send_frame(fd, hello, timeout_ms);
        auto ready = receive_frame(fd, timeout_ms);
        if (!ready) { executor.shutdown(); return; }
        match(*ready, hello, Kind::ready);
        SharedActivation shared;
        if (shared_memory) shared = SharedActivation::receive(fd, timeout_ms);
        while (auto request = receive_frame(fd, timeout_ms)) {
            try {
                if (request->kind != Kind::window_spec || !request->window)
                    throw std::runtime_error("final worker requires verifier window metadata");
                if (request->stage_index != index ||
                    request->plan_fingerprint != hello.plan_fingerprint ||
                    request->geometry.hc != plan.geometry.hc ||
                    request->geometry.n_embd != plan.geometry.n_embd)
                    throw std::runtime_error("final worker window plan mismatch");
                if (!executor.begin_window(*request->window,
                                           request->request_id, request->window_id, error))
                    throw std::runtime_error(error.empty() ? "final window admission failed" : error);
                auto data = receive_frame(fd, timeout_ms);
                if (!data) throw std::runtime_error("final worker closed after window metadata");
                match(*data, *request, shared_memory ? Kind::shared_data : Kind::data);
                const size_t floats = static_cast<size_t>(request->rows) * plan.geometry.floats_per_row();
                const size_t bytes = floats * sizeof(float);
                // encode()/decode() bound the activation window to the 16MiB cap.
                Frame metadata = *data;
                metadata.kind = Kind::data;
                metadata.payload.clear();
                metadata.shared_checksum = 0;
                FinalStageResult result;
                if (shared_memory) {
                    if (shared.checksum(bytes) != data->shared_checksum)
                        throw std::runtime_error("final worker shared input checksum mismatch");
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
                send_final_result(fd, result, timeout_ms);
                auto commit = receive_frame(fd, timeout_ms);
                if (!commit) { executor.abandon_window(); break; }
                match(*commit, *request, Kind::commit_prefix);
                if (!executor.commit_prefix(request->request_id, request->window_id,
                                            commit->keep_count, error))
                    throw std::runtime_error(error.empty() ? "final stage prefix commit failed" : error);
                Frame ack = *commit;
                ack.kind = Kind::committed_prefix;
                send_frame(fd, ack, timeout_ms);
            } catch (const std::exception& ex) {
                executor.abandon_window();
                failure(fd, *request, ex.what(), timeout_ms);
                executor.shutdown();
                return;
            }
        }
    } catch (...) {
        executor.abandon_window();
        executor.shutdown();
        throw;
    }
    executor.shutdown();
}

FinalPipelineSession::FinalPipelineSession(Plan plan, std::vector<int> fds,
                                           int timeout_ms, bool shared_memory)
    : plan_(std::move(plan)), fds_(std::move(fds)), timeout_ms_(timeout_ms),
      shared_memory_(shared_memory) {
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
        for (size_t i = 0; i < fds_.size(); ++i) {
            auto hello = receive_frame(fds_[i], timeout_ms_);
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
            send_frame(fds_[i], control(plan_, i, Kind::ready), timeout_ms_);
            if (shared_memory_) shared_->send(fds_[i], timeout_ms_);
        }
        initialized_ = true;
    } catch (...) { poison(); throw; }
}
FinalStageResult FinalPipelineSession::execute(Frame initial, const WindowSpec& spec) {
    if (!initialized_ || poisoned_) throw std::logic_error("final pipeline unavailable");
    if (pending_) throw std::logic_error("final pipeline already has uncommitted result");
    try {
        if (initial.kind != Kind::data || !initial.request_id || !initial.window_id ||
            initial.plan_fingerprint != plan_.fingerprint() ||
            initial.geometry.hc != plan_.geometry.hc ||
            initial.geometry.n_embd != plan_.geometry.n_embd)
            throw std::invalid_argument("initial final pipeline activation identity/shape invalid");
        (void)encode(initial);
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
            Frame context = control(plan_, i, Kind::window_spec,
                                    initial.request_id, initial.window_id, initial.rows);
            context.window = spec;
            send_frame(fds_[i], context, timeout_ms_);
            initial.stage_index = static_cast<uint32_t>(i);
            send_frame(fds_[i], initial, timeout_ms_);
            if (i + 1 == fds_.size()) {
                auto response = receive_final_result(fds_[i], timeout_ms_);
                if (!response) throw std::runtime_error("final worker closed without head result");
                gate_.stage(std::move(*response), static_cast<uint32_t>(i),
                            initial.request_id, initial.window_id, initial.plan_fingerprint, initial.rows);
            } else {
                auto response = receive_frame(fds_[i], timeout_ms_);
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
        return gate_.provisional();
    } catch (...) { poison(); throw; }
}
FinalStageResult FinalPipelineSession::commit(uint32_t n_keep) {
    if (!initialized_ || poisoned_ || !pending_)
        throw std::logic_error("no provisional final-stage result to commit");
    if (n_keep > pending_rows_) throw std::invalid_argument("accepted prefix exceeds row count");
    try {
        if (!gate_.provisional().nonfinite_rows.empty())
            throw std::runtime_error("nonfinite head result cannot be committed");
        for (size_t i = 0; i < fds_.size(); ++i) {
            Frame request = control(plan_, i, Kind::commit_prefix,
                                    pending_request_, pending_window_, pending_rows_);
            request.keep_count = n_keep;
            send_frame(fds_[i], request, timeout_ms_);
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
