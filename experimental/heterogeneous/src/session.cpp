// CPU-only worker handshake and transactional window coordinator.
// Production backends will implement StageExecutor in separate CUDA/HIP binaries.
#include "strata_heterogeneous/host.hpp"
#include "strata_heterogeneous/shared.hpp"

#include <algorithm>
#include <cstring>
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

void send_failure(int fd, Frame request, const std::string& reason, int timeout_ms) {
    request.kind = Kind::failure;
    request.payload.clear();
    request.identity = {};
    request.shared_checksum = 0;
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

bool StageExecutor::run_staged_window(const Frame& metadata, ActivationView view, std::string& error) {
    if (metadata.kind != Kind::data || metadata.rows != view.rows() ||
        metadata.geometry.hc != view.geometry().hc ||
        metadata.geometry.n_embd != view.geometry().n_embd ||
        !metadata.payload.empty() || metadata.shared_checksum)
        throw std::invalid_argument("staged fallback metadata/view mismatch");
    // The default adapter is intentionally conservative. GPU adapters should
    // override this to perform an explicit host->device->host copy and complete
    // their streams before returning; never interpret view.data() as a device pointer.
    Frame input = metadata;
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
bool MockStageExecutor::run_window(const Frame& input, Frame& output, std::string& error) {
    if (!prepared_ || pending_) { error = "mock stage not ready"; return false; }
    if (!mock_execute(stage_, input, output, error)) return false;
    pending_ = true;
    pending_request_ = input.request_id;
    pending_window_ = input.window_id;
    return true;
}
bool MockStageExecutor::run_staged_window(const Frame& metadata, ActivationView view,
                                           std::string& error) {
    if (!prepared_ || pending_) { error = "mock stage not ready"; return false; }
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
    if (!pending_ || request_id != pending_request_ || window_id != pending_window_) {
        error = "mock commit does not match pending window";
        return false;
    }
    pending_ = false;
    return true;
}
void MockStageExecutor::abandon_window() noexcept { pending_ = false; }
void MockStageExecutor::shutdown() noexcept { pending_ = false; prepared_ = false; }

void serve_worker_stateful(int fd, const Plan& plan, size_t index,
                           StageExecutor& executor, int timeout_ms, bool shared_memory) {
    plan.validate();
    if (index >= plan.stages.size()) throw std::invalid_argument("worker index out of range");
    if (timeout_ms < 0) throw std::invalid_argument("negative worker timeout");
    const Frame hello = control(plan, index, shared_memory ? Kind::hello_shared : Kind::hello);
    std::string error;
    if (!executor.prepare(plan, index, error)) {
        send_failure(fd, hello, error.empty() ? "worker preparation failed" : error, timeout_ms);
        executor.shutdown();
        return;
    }
    try {
        send_frame(fd, hello, timeout_ms);
        auto ready = receive_frame(fd, timeout_ms);
        if (!ready) { executor.shutdown(); return; }
        require_match(*ready, hello, Kind::ready);
        SharedActivation shared;
        if (shared_memory) shared = SharedActivation::receive(fd, timeout_ms);
        while (auto request = receive_frame(fd, timeout_ms)) {
            try {
                if (request->kind != (shared_memory ? Kind::shared_data : Kind::data) ||
                    request->stage_index != index ||
                    request->plan_fingerprint != hello.plan_fingerprint ||
                    request->geometry.hc != plan.geometry.hc ||
                    request->geometry.n_embd != plan.geometry.n_embd)
                    throw std::runtime_error("worker plan/stage/geometry mismatch");
                Frame reply;
                error.clear();
                if (shared_memory) {
                    // A single mapped slot belongs to this stage until its
                    // reply is received. The supervisor serializes ownership.
                    const size_t floats = static_cast<size_t>(request->rows) * plan.geometry.floats_per_row();
                    const size_t bytes = floats * sizeof(float); // capped by frame decode
                    if (shared.checksum(bytes) != request->shared_checksum)
                        throw std::runtime_error("shared activation checksum mismatch at stage entry");
                    Frame metadata = *request;
                    metadata.kind = Kind::data;
                    metadata.shared_checksum = 0;
                    ActivationView view(shared.data(), bytes, metadata.geometry, metadata.rows);
                    if (!executor.run_staged_window(metadata, view, error))
                        throw std::runtime_error(error.empty() ? "worker staged execution failed" : error);
                    reply = *request;
                    reply.shared_checksum = shared.checksum(bytes);
                } else {
                    Frame output = *request;
                    output.payload.clear();
                    if (!executor.run_window(*request, output, error))
                        throw std::runtime_error(error.empty() ? "worker execution failed" : error);
                    require_match(output, *request, Kind::data);
                    (void) encode(output); // validate exact shape before publication
                    reply = std::move(output);
                }
                send_frame(fd, reply, timeout_ms);
                auto commit = receive_frame(fd, timeout_ms);
                if (!commit) { executor.abandon_window(); break; }
                require_match(*commit, *request, Kind::commit);
                error.clear();
                if (!executor.commit_window(request->request_id, request->window_id, error))
                    throw std::runtime_error(error.empty() ? "worker commit failed" : error);
                Frame ack = *commit;
                ack.kind = Kind::committed;
                send_frame(fd, ack, timeout_ms);
            } catch (const std::exception& ex) {
                executor.abandon_window();
                send_failure(fd, *request, ex.what(), timeout_ms);
                executor.shutdown();
                return; // fail-stop: never decode from possibly damaged stage state
            }
        }
    } catch (...) {
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
Frame PipelineSession::run_window(Frame initial) {
    if (!initialized_ || poisoned_) throw std::logic_error("pipeline is uninitialized or poisoned");
    try {
        if (initial.kind != Kind::data ||
            initial.plan_fingerprint != plan_.fingerprint() ||
            initial.geometry.hc != plan_.geometry.hc ||
            initial.geometry.n_embd != plan_.geometry.n_embd)
            throw std::invalid_argument("initial frame doesn't match plan");
        if (initial.request_id == last_request_ && initial.window_id <= last_window_)
            throw std::invalid_argument("stale or replayed window ID");
        const uint64_t committed_request = initial.request_id;
        const uint64_t committed_window = initial.window_id;
        // Execution phase: every stage returns provisional data; no stage commits.
        // Only one window owns the shared slot. Never overwrite until commits finish.
        size_t floats = 0;
        size_t bytes = 0;
        if (shared_memory_) {
            // Validation checks geometry multiplication, byte cap and exact shape
            // before any host size arithmetic or mapped-slot write.
            (void) encode(initial);
            floats = static_cast<size_t>(initial.rows) * plan_.geometry.floats_per_row();
            bytes = floats * sizeof(float);
            shared_->store(initial.payload);
            initial.payload.clear();
            initial.kind = Kind::shared_data;
            initial.shared_checksum = shared_->checksum(bytes);
        }
        for (size_t i = 0; i < fds_.size(); ++i) {
            initial.stage_index = static_cast<uint32_t>(i);
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
        // Commit phase: acknowledge every stage before publishing final output.
        // A failed commit may leave a partial transaction; poison all workers.
        for (size_t i = 0; i < fds_.size(); ++i) {
            Frame request = control(plan_, i, Kind::commit,
                                    initial.request_id, initial.window_id, initial.rows);
            send_frame(fds_[i], request, timeout_ms_);
            auto reply = receive_frame(fds_[i], timeout_ms_);
            if (!reply) throw std::runtime_error("worker exited during commit");
            if (reply->kind == Kind::failure)
                throw std::runtime_error("stage " + std::to_string(i) + " commit failed: " + reply->error);
            require_match(*reply, request, Kind::committed);
        }
        if (shared_memory_) {
            initial.kind = Kind::data;
            initial.payload = shared_->load(floats);
            initial.shared_checksum = 0;
        }
        last_request_ = committed_request;
        last_window_ = committed_window;
        return initial;
    } catch (...) { poison(); throw; }
}
} // namespace strata::heterogeneous
