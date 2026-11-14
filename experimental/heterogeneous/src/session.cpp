// CPU-only worker handshake and transactional window coordinator.
// Production backends will implement StageExecutor in separate CUDA/HIP binaries.
#include "strata_heterogeneous/host.hpp"
#include "strata_heterogeneous/shared.hpp"
#include "strata_heterogeneous/window_admission.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sys/socket.h>
#include <utility>

namespace strata::heterogeneous {
namespace {
constexpr size_t max_failure_bytes = 1024;

Frame control(const Plan& plan, size_t index, Kind kind,
              uint64_t request = 1, uint64_t window = 1, uint32_t rows = 1) {
    Frame f;
    f.kind = kind;
    f.stage_index = static_cast<uint32_t>(index);
    f.request_id = request;
    f.window_id = window;
    f.rows = rows;
    f.geometry = plan.geometry;
    f.plan_fingerprint = plan.fingerprint();
    if (kind == Kind::hello || kind == Kind::hello_shared) {
        const Stage& stage = plan.stages.at(index);
        f.identity = {static_cast<uint32_t>(stage.backend), stage.ordinal,
                      stage.first_layer, stage.end_layer};
    }
    return f;
}

void require_match(const Frame& actual, const Frame& requested, Kind kind) {
    if (actual.kind != kind || actual.request_id != requested.request_id ||
        actual.window_id != requested.window_id || actual.rows != requested.rows ||
        actual.stage_index != requested.stage_index ||
        actual.plan_fingerprint != requested.plan_fingerprint ||
        actual.geometry.hc != requested.geometry.hc ||
        actual.geometry.n_embd != requested.geometry.n_embd)
        throw std::runtime_error("stale or mismatched worker reply");
}

void trace_window(const WindowAdmission* gate, const char* phase) noexcept {
    if (!gate) return;
    const char* opt = std::getenv("STRATA_HETERO_WINDOW_TRACE");
    if (!opt || std::strcmp(opt, "1") != 0) return;
    try { std::fprintf(stderr, "%s\n", gate->diagnostic(phase).c_str()); }
    catch (...) {} // diagnostics can never bypass fail-stop
}

void send_failure(int fd, Frame request, const std::string& reason, int timeout_ms) {
    request.kind = Kind::failure;
    request.payload.clear();
    request.identity = {};
    request.shared_checksum = 0;
    request.window.reset();
    request.keep_count = 0;
    request.error = reason.substr(0, max_failure_bytes);
    if (request.error.empty()) request.error = "unknown stage failure";
    send_frame(fd, request, timeout_ms);
}

class CallbackExecutor final : public StageExecutor {
public:
    explicit CallbackExecutor(Executor exec) : exec_(std::move(exec)) {}
    bool prepare(const Plan& plan, size_t index, std::string&) override {
        stage_ = plan.stages.at(index);
        return true;
    }
    bool run_window(const Frame& input, Frame& output, std::string& error) override {
        if (pending_) { error = "previous window not committed"; return false; }
        if (!exec_(stage_, input, output, error)) return false;
        pending_ = true;
        return true;
    }
    bool commit_window(uint64_t, uint64_t, std::string& error) override {
        if (!pending_) { error = "no window to commit"; return false; }
        pending_ = false;
        return true;
    }
    void abandon_window() noexcept override { pending_ = false; }
    void shutdown() noexcept override { pending_ = false; }
private:
    Stage stage_{};
    Executor exec_;
    bool pending_ = false;
};
} // namespace

bool StageExecutor::begin_window(const WindowSpec&, uint64_t, uint64_t, std::string& error) {
    error = "executor does not support token/position window metadata";
    return false;
}
bool StageExecutor::commit_prefix(uint64_t, uint64_t, uint32_t, std::string& error) {
    error = "executor does not support accepted-prefix commit";
    return false;
}

bool StageExecutor::run_staged_window(const Frame& metadata, ActivationView view, std::string& error) {
    if (metadata.kind != Kind::data || metadata.rows != view.rows() ||
        metadata.geometry.hc != view.geometry().hc ||
        metadata.geometry.n_embd != view.geometry().n_embd ||
        !metadata.payload.empty() || metadata.shared_checksum)
        throw std::invalid_argument("staged fallback metadata/view mismatch");
    // The default adapter is intentionally conservative. GPU adapters should
    // override this to perform an explicit host->device->host copy and complete
    // their streams before returning; never interpret view.data() as a device pointer.
    // Copy-assign validated metadata to avoid GCC 13's optimized empty-vector
    // copy-constructor array-bounds diagnostic in native Release builds.
    Frame input;
    input = metadata;
    input.kind = Kind::data;
    input.shared_checksum = 0;
    input.payload.resize(view.bytes() / sizeof(float));
    std::memcpy(input.payload.data(), view.data(), view.bytes());
    Frame output = input;
    output.payload.clear();
    if (!run_window(input, output, error)) return false;
    if (output.kind != Kind::data || output.request_id != metadata.request_id ||
        output.window_id != metadata.window_id || output.rows != metadata.rows ||
        output.stage_index != metadata.stage_index ||
        output.plan_fingerprint != metadata.plan_fingerprint ||
        output.geometry.hc != metadata.geometry.hc ||
        output.geometry.n_embd != metadata.geometry.n_embd)
        throw std::runtime_error("staged fallback returned incorrect window metadata");
    (void) encode(output); // validate exact output shape *before* mutating mapping
    if (output.payload.size() * sizeof(float) != view.bytes())
        throw std::runtime_error("staged fallback changed activation byte length");
    std::memcpy(view.data(), output.payload.data(), view.bytes());
    return true;
}

bool MockStageExecutor::prepare(const Plan& plan, size_t stage_index, std::string& error) {
    if (prepared_ || stage_index >= plan.stages.size()) {
        error = "mock stage prepared twice or invalid index";
        return false;
    }
    stage_ = plan.stages[stage_index];
    prepared_ = true;
    return true;
}
bool MockStageExecutor::reset_request(uint64_t request, std::string& error) {
    if (!prepared_ || pending_ || has_spec_ || !request) {
        error = "mock reset requires an idle executor"; return false;
    }
    pending_request_ = request;
    pending_window_ = 0;
    window_spec_ = {};
    return true;
}
bool MockStageExecutor::begin_window(const WindowSpec& spec, uint64_t request,
                                    uint64_t window, std::string& error) {
    if (!prepared_ || pending_ || has_spec_ || !request || !window) {
        error = "mock window context conflicts with pending state"; return false;
    }
    try { spec.validate(static_cast<uint32_t>(spec.token_ids.size())); }
    catch (const std::exception& e) { error = e.what(); return false; }
    has_spec_ = true;
    window_spec_ = spec;
    pending_request_ = request;
    pending_window_ = window;
    return true;
}
bool MockStageExecutor::commit_prefix(uint64_t request, uint64_t window,
                                     uint32_t n_keep, std::string& error) {
    if (!has_spec_ || !pending_ || n_keep > window_spec_.token_ids.size() ||
        request != pending_request_ || window != pending_window_) {
        error = "mock accepted-prefix commit mismatch"; return false;
    }
    pending_ = false; has_spec_ = false;
    return true;
}
bool MockStageExecutor::run_window(const Frame& input, Frame& output, std::string& error) {
    if (!prepared_ || pending_ || (has_spec_ &&
        (input.request_id != pending_request_ || input.window_id != pending_window_ ||
         input.rows != window_spec_.token_ids.size()))) {
        error = "mock stage not ready or metadata mismatch"; return false;
    }
    if (!mock_execute(stage_, input, output, error)) return false;
    pending_ = true;
    pending_request_ = input.request_id;
    pending_window_ = input.window_id;
    return true;
}
bool MockStageExecutor::run_staged_window(const Frame& metadata, ActivationView view,
                                           std::string& error) {
    if (!prepared_ || pending_ || (has_spec_ &&
        (metadata.request_id != pending_request_ || metadata.window_id != pending_window_ ||
         metadata.rows != window_spec_.token_ids.size()))) {
        error = "mock staged metadata mismatch"; return false;
    }
    if (metadata.kind != Kind::data || metadata.rows != view.rows() ||
        metadata.geometry.hc != view.geometry().hc ||
        metadata.geometry.n_embd != view.geometry().n_embd ||
        !metadata.payload.empty() || metadata.shared_checksum) {
        error = "staged metadata does not match mapped view";
        return false;
    }
    // All arithmetic uses memcpy: the memfd is raw bytes, not a C++ float[].
    // Preserve the same layer-major operation order as mock_execute().
    ByteRange all{view.data(), view.bytes()};
    for (uint32_t l = stage_.first_layer; l < stage_.end_layer; ++l)
        for (size_t i = 0; i < view.bytes() / sizeof(float); ++i)
            all.write_f32(i, all.read_f32(i) + static_cast<float>(l + 1) * 0.0001f);
    pending_ = true;
    pending_request_ = metadata.request_id;
    pending_window_ = metadata.window_id;
    return true;
}
bool MockStageExecutor::commit_window(uint64_t request_id, uint64_t window_id, std::string& error) {
    if (!pending_ || has_spec_ || request_id != pending_request_ || window_id != pending_window_) {
        error = "mock commit does not match pending window";
        return false;
    }
    pending_ = false; has_spec_ = false;
    return true;
}
void MockStageExecutor::abandon_window() noexcept { pending_ = false; has_spec_ = false; }
void MockStageExecutor::shutdown() noexcept { pending_ = false; has_spec_ = false; prepared_ = false; }

void serve_worker_stateful(int fd, const Plan& plan, size_t index,
                           StageExecutor& executor, int timeout_ms, bool shared_memory,
                           const WindowLimits* limits, int startup_wait_ms) {
    plan.validate();
    if (index >= plan.stages.size()) throw std::invalid_argument("worker index out of range");
    if (timeout_ms < 0) throw std::invalid_argument("negative worker timeout");
    const Frame hello = control(plan, index, shared_memory ? Kind::hello_shared : Kind::hello);
    std::unique_ptr<WindowAdmission> gate;
    if (limits) gate = std::make_unique<WindowAdmission>(plan, index, *limits);
    std::string error;
    if (!executor.prepare(plan, index, error)) {
        send_failure(fd, hello, error.empty() ? "worker preparation failed" : error, timeout_ms);
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
        require_match(*ready, hello, resident ? Kind::ready_resident : Kind::ready);
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
                    require_match(*request, expected, Kind::begin_request);
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
                bool has_window_spec = false;
                if (request->kind == Kind::window_spec) {
                    const Kind expected_data = shared_memory ? Kind::shared_data : Kind::data;
                    if (request->stage_index != index ||
                        request->plan_fingerprint != hello.plan_fingerprint ||
                        request->geometry.hc != plan.geometry.hc ||
                        request->geometry.n_embd != plan.geometry.n_embd || !request->window)
                        throw std::runtime_error("bad worker window specification");
                    error.clear();
                    if (gate && !gate->begin(*request->window, request->request_id, request->window_id, error))
                        throw std::runtime_error(error);
                    trace_window(gate.get(), "admitted");
                    if (!executor.begin_window(*request->window, request->request_id, request->window_id, error))
                        throw std::runtime_error(error.empty() ? "window metadata rejected" : error);
                    auto activation = receive_frame(fd, timeout_ms);
                    if (!activation) throw std::runtime_error("worker closed after window specification");
                    require_match(*activation, *request, expected_data);
                    request = std::move(activation);
                    has_window_spec = true;
                }
                if (gate && !has_window_spec)
                    throw std::runtime_error("CUDA worker requires a verifier window_spec before activation");
                if (request->kind != (shared_memory ? Kind::shared_data : Kind::data) ||
                    request->stage_index != index ||
                    request->plan_fingerprint != hello.plan_fingerprint ||
                    request->geometry.hc != plan.geometry.hc ||
                    request->geometry.n_embd != plan.geometry.n_embd)
                    throw std::runtime_error("worker plan/stage/geometry mismatch");
                if (gate && !gate->validate_input(*request, shared_memory, error))
                    throw std::runtime_error(error);
                Frame reply;
                error.clear();
                if (shared_memory) {
                    // A single mapped slot belongs to this stage until its
                    // reply is received. The supervisor serializes ownership.
                    const size_t floats = static_cast<size_t>(request->rows) * plan.geometry.floats_per_row();
                    const size_t bytes = floats * sizeof(float); // capped by frame decode
                    if (shared.checksum(bytes) != request->shared_checksum)
                        throw std::runtime_error("shared activation checksum mismatch at stage entry");
                    if (gate && !gate->validate_bytes(shared.data(), bytes, error))
                        throw std::runtime_error(error);
                    Frame metadata = *request;
                    metadata.kind = Kind::data;
                    metadata.shared_checksum = 0;
                    ActivationView view(shared.data(), bytes, metadata.geometry, metadata.rows);
                    if (!executor.run_staged_window(metadata, view, error))
                        throw std::runtime_error(error.empty() ? "worker staged execution failed" : error);
                    if (gate && !gate->provisional(shared.data(), bytes, error))
                        throw std::runtime_error(error);
                    reply = *request;
                    reply.shared_checksum = shared.checksum(bytes);
                } else {
                    Frame output = *request;
                    output.payload.clear();
                    if (!executor.run_window(*request, output, error))
                        throw std::runtime_error(error.empty() ? "worker execution failed" : error);
                    require_match(output, *request, Kind::data);
                    (void) encode(output); // validate exact shape before publication
                    if (gate && !gate->provisional(
                            reinterpret_cast<const uint8_t*>(output.payload.data()),
                            output.payload.size() * sizeof(float), error))
                        throw std::runtime_error(error);
                    reply = std::move(output);
                }
                trace_window(gate.get(), "provisional");
                send_frame(fd, reply, timeout_ms);
                auto commit = receive_frame(fd, timeout_ms);
                if (!commit) { executor.abandon_window(); break; }
                require_match(*commit, *request, has_window_spec ? Kind::commit_prefix : Kind::commit);
                if (gate && !gate->check_commit(*commit, error))
                    throw std::runtime_error(error);
                error.clear();
                const bool committed = has_window_spec
                    ? executor.commit_prefix(request->request_id, request->window_id, commit->keep_count, error)
                    : executor.commit_window(request->request_id, request->window_id, error);
                if (!committed)
                    throw std::runtime_error(error.empty() ? "worker commit failed" : error);
                if (gate && !gate->committed(commit->keep_count, error))
                    throw std::runtime_error(error);
                trace_window(gate.get(), "committed");
                Frame ack = *commit;
                ack.kind = has_window_spec ? Kind::committed_prefix : Kind::committed;
                send_frame(fd, ack, timeout_ms);
            } catch (const std::exception& ex) {
                trace_window(gate.get(), "failure");
                if (gate) gate->poison();
                executor.abandon_window();
                send_failure(fd, *request, ex.what(), timeout_ms);
                executor.shutdown();
                return; // fail-stop: never decode from possibly damaged stage state
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

void serve_worker(int fd, const Plan& plan, size_t index, const Executor& exec) {
    CallbackExecutor adapter(exec);
    serve_worker_stateful(fd, plan, index, adapter);
}

PipelineSession::PipelineSession(Plan plan, std::vector<int> worker_fds, int timeout_ms,
                                 bool shared_memory)
    : plan_(std::move(plan)), fds_(std::move(worker_fds)), timeout_ms_(timeout_ms),
      shared_memory_(shared_memory) {
    plan_.validate();
    if (fds_.size() != plan_.stages.size())
        throw std::invalid_argument("one worker fd required per stage");
    if (timeout_ms_ < 0) throw std::invalid_argument("negative pipeline timeout");
}
PipelineSession::~PipelineSession() = default;
void PipelineSession::poison() noexcept {
    poisoned_ = true;
    // Do not close caller-owned FDs; shutdown wakes workers awaiting commit.
    for (int fd : fds_) if (fd >= 0) (void) ::shutdown(fd, SHUT_RDWR);
}
void PipelineSession::initialize() {
    if (initialized_ || poisoned_) throw std::logic_error("pipeline already initialized or poisoned");
    try {
        for (size_t i = 0; i < fds_.size(); ++i) {
            auto hello = receive_frame(fds_[i], timeout_ms_);
            if (!hello) throw std::runtime_error("worker closed before hello");
            if (hello->kind == Kind::failure)
                throw std::runtime_error("worker " + std::to_string(i) + " preparation failed: " + hello->error);
            const Kind expected_kind = shared_memory_ ? Kind::hello_shared : Kind::hello;
            const Frame expected = control(plan_, i, expected_kind);
            require_match(*hello, expected, expected_kind);
            if (hello->identity != expected.identity)
                throw std::runtime_error("worker backend, device or layer range mismatch");
        }
        if (shared_memory_) shared_ = std::make_unique<SharedActivation>(SharedActivation::create());
        for (size_t i = 0; i < fds_.size(); ++i) {
            send_frame(fds_[i], control(plan_, i, Kind::ready), timeout_ms_);
            if (shared_memory_) shared_->send(fds_[i], timeout_ms_);
        }
        initialized_ = true;
    } catch (...) { poison(); throw; }
}
Frame PipelineSession::execute_impl(Frame initial, const WindowSpec* spec) {
    if (!initialized_ || poisoned_) throw std::logic_error("pipeline is uninitialized or poisoned");
    if (pending_) throw std::logic_error("previous window is still provisional");
    try {
        if (initial.kind != Kind::data ||
            initial.plan_fingerprint != plan_.fingerprint() ||
            initial.geometry.hc != plan_.geometry.hc ||
            initial.geometry.n_embd != plan_.geometry.n_embd)
            throw std::invalid_argument("initial frame doesn't match plan");
        (void) encode(initial); // reject malformed payload before worker contact
        if (initial.request_id == last_request_ && initial.window_id <= last_window_)
            throw std::invalid_argument("stale or replayed window ID");
        if (spec) spec->validate(initial.rows);
        const size_t floats = static_cast<size_t>(initial.rows) * plan_.geometry.floats_per_row();
        const size_t bytes = floats * sizeof(float); // capped by encode(initial)
        if (shared_memory_) {
            shared_->store(initial.payload);
            initial.payload.clear();
            initial.kind = Kind::shared_data;
            initial.shared_checksum = shared_->checksum(bytes);
        }
        for (size_t i = 0; i < fds_.size(); ++i) {
            initial.stage_index = static_cast<uint32_t>(i);
            if (spec) {
                Frame context = control(plan_, i, Kind::window_spec,
                                        initial.request_id, initial.window_id, initial.rows);
                context.window = *spec;
                send_frame(fds_[i], context, timeout_ms_);
            }
            send_frame(fds_[i], initial, timeout_ms_);
            auto answer = receive_frame(fds_[i], timeout_ms_);
            if (!answer) throw std::runtime_error("worker closed before replying");
            if (answer->kind == Kind::failure)
                throw std::runtime_error("stage " + std::to_string(i) + " failed: " + answer->error);
            require_match(*answer, initial, shared_memory_ ? Kind::shared_data : Kind::data);
            if (shared_memory_ && answer->shared_checksum != shared_->checksum(bytes))
                throw std::runtime_error("shared activation checksum mismatch after worker reply");
            initial = std::move(*answer);
        }
        if (shared_memory_) {
            initial.kind = Kind::data;
            initial.payload = shared_->load(floats);
            initial.shared_checksum = 0;
        }
        // Provisional output can be inspected to determine speculative n_keep.
        // It must not be published as committed inference or saved as session state.
        provisional_ = std::move(initial);
        pending_ = true;
        pending_spec_ = (spec != nullptr);
        return provisional_;
    } catch (...) { poison(); throw; }
}
Frame PipelineSession::commit_impl(uint32_t n_keep) {
    if (!initialized_ || poisoned_ || !pending_)
        throw std::logic_error("no live provisional window to commit");
    // Invalid client input does not mutate worker state; permit a corrected count.
    if (n_keep > provisional_.rows) throw std::invalid_argument("accepted prefix exceeds window rows");
    try {
        for (size_t i = 0; i < fds_.size(); ++i) {
            Frame request = control(plan_, i, pending_spec_ ? Kind::commit_prefix : Kind::commit,
                                    provisional_.request_id, provisional_.window_id, provisional_.rows);
            if (pending_spec_) request.keep_count = n_keep;
            send_frame(fds_[i], request, timeout_ms_);
            auto reply = receive_frame(fds_[i], timeout_ms_);
            if (!reply) throw std::runtime_error("worker exited during commit");
            if (reply->kind == Kind::failure)
                throw std::runtime_error("stage " + std::to_string(i) + " commit failed: " + reply->error);
            require_match(*reply, request, pending_spec_ ? Kind::committed_prefix : Kind::committed);
            if (pending_spec_ && reply->keep_count != n_keep)
                throw std::runtime_error("worker accepted-prefix acknowledgment mismatch");
        }
        Frame output = std::move(provisional_);
        last_request_ = output.request_id;
        last_window_ = output.window_id;
        pending_ = false;
        pending_spec_ = false;
        return output;
    } catch (...) { poison(); throw; }
}
Frame PipelineSession::run_window(Frame initial) {
    const uint32_t rows = initial.rows;
    (void) execute_impl(std::move(initial), nullptr);
    return commit_impl(rows);
}
Frame PipelineSession::execute_window(Frame initial, const WindowSpec& spec) {
    return execute_impl(std::move(initial), &spec);
}
Frame PipelineSession::commit_pending(uint32_t n_keep) {
    if (!pending_ || !pending_spec_)
        throw std::logic_error("no metadata-bearing provisional window to commit");
    return commit_impl(n_keep);
}
} // namespace strata::heterogeneous
