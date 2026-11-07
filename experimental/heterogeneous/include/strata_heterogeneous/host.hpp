#pragma once

// A host-only prototype for cross-runtime Strata stage handoffs.
// No CUDA or HIP types cross this boundary.  See README.md for limitations.

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::heterogeneous {

enum class Backend : uint8_t { cuda = 1, hip = 2 };

struct Stage {
    Backend backend = Backend::cuda;
    uint32_t ordinal = 0;      // ordinal in this backend's own process
    uint32_t first_layer = 0;  // inclusive
    uint32_t end_layer = 0;    // exclusive
};

struct Geometry {
    uint32_t hc = 0;
    uint32_t n_embd = 0;
    // Same per-token layout as core::Verifier::handoff_floats():
    // [R: hc*n_embd floats | bo: n_embd floats | inject: hc floats].
    uint64_t floats_per_row() const;
};

struct Plan {
    uint32_t total_layers = 0;
    uint64_t model_fingerprint = 0;  // same nonzero model/config fingerprint on ALL workers
    Geometry geometry;
    std::vector<Stage> stages;

    void validate() const;
    uint64_t fingerprint() const;
};

// Example: "cuda:0:0-12,hip:0:12-30,hip:1:30-48".
// Device ordinals are backend-local. Does NOT discover GPUs.
Plan parse_plan(const std::string& spec, uint32_t layers, Geometry geometry, uint64_t model_fingerprint);

// Protocol v2: handshake and two-phase execution/commit control messages.
// Only stage results (data) carry R/bo/inject floats; hello carries 4 u32s.
enum class Kind : uint16_t {
    data = 1, failure = 2, hello = 3, ready = 4,
    commit = 5, committed = 6
};
struct Frame {
    Kind kind = Kind::data;
    uint32_t stage_index = 0;
    uint64_t request_id = 0;
    uint64_t window_id = 0;
    uint64_t plan_fingerprint = 0;
    uint32_t rows = 0;
    Geometry geometry;
    std::vector<float> payload;
    std::string error;  // only for Kind::failure
    // hello: [backend, ordinal, first_layer, end_layer]; other kinds: zero.
    std::array<uint32_t, 4> identity{};
};

constexpr uint32_t max_payload_bytes = 16 * 1024 * 1024;
constexpr size_t wire_header_bytes = 56;
constexpr int default_io_timeout_ms = 30000;

std::vector<uint8_t> encode(const Frame& f);
Frame decode(const std::vector<uint8_t>& encoded);

// POSIX AF_UNIX SOCK_STREAM transport: short reads/writes and EINTR handled, no GPU API.
// Receiving an orderly EOF returns nullopt; truncated/invalid frames throw.
// This deliberately copies payloads. Shared-memory/DMA optimization comes later.
// Total, monotonic wall-clock deadline per frame, not per read/write chunk.
// timeout_ms must be >= 0. A timeout throws; callers must poison the session.
void send_frame(int fd, const Frame& frame, int timeout_ms = default_io_timeout_ms);
std::optional<Frame> receive_frame(int fd, int timeout_ms = default_io_timeout_ms);

using Executor = std::function<bool(const Stage&, const Frame&, Frame&, std::string&)>;

// The stage owns layer-local state and may have a prepared, uncommitted window.
// run_window must not advance durable state; commit_window does so only when all
// stages produced valid outputs. A failure during commit poisons the session;
// rollback across partially committed GPU workers is NOT promised.
class StageExecutor {
public:
    virtual ~StageExecutor() = default;
    virtual bool prepare(const Plan& plan, size_t stage_index, std::string& error) = 0;
    virtual bool run_window(const Frame& input, Frame& output, std::string& error) = 0;
    virtual bool commit_window(uint64_t request_id, uint64_t window_id, std::string& error) = 0;
    virtual void abandon_window() noexcept = 0;
    virtual void shutdown() noexcept = 0;
};

// Stateless test adapter, not a real GPU-backed stage.
class MockStageExecutor final : public StageExecutor {
public:
    bool prepare(const Plan& plan, size_t stage_index, std::string& error) override;
    bool run_window(const Frame& input, Frame& output, std::string& error) override;
    bool commit_window(uint64_t request_id, uint64_t window_id, std::string& error) override;
    void abandon_window() noexcept override;
    void shutdown() noexcept override;
private:
    Stage stage_{};
    bool prepared_ = false;
    bool pending_ = false;
    uint64_t pending_request_ = 0;
    uint64_t pending_window_ = 0;
};

// Workers send an identity hello before accepting data. Coordinator verifies
// every worker's declared backend/ordinal/layer range, then sends ready.
void serve_worker_stateful(int fd, const Plan& plan, size_t stage_index,
                           StageExecutor& executor, int timeout_ms = default_io_timeout_ms);
// Backwards-compatible CPU-test convenience; now uses the same transactional
// handshake/commit path as StageExecutor. It is NOT a GPU adapter.
void serve_worker(int fd, const Plan& plan, size_t stage_index, const Executor& execute);

// Owns only the *protocol state*, not the child processes or socket descriptors.
// On any failure, shuts down sockets, rejects later windows, and requires new
// workers/session; it never publishes a partially completed window.
class PipelineSession {
public:
    PipelineSession(Plan plan, std::vector<int> worker_fds, int timeout_ms = default_io_timeout_ms);
    void initialize();
    Frame run_window(Frame initial);
    bool poisoned() const { return poisoned_; }
    bool initialized() const { return initialized_; }
private:
    Plan plan_;
    std::vector<int> fds_;
    int timeout_ms_;
    bool initialized_ = false;
    bool poisoned_ = false;
    uint64_t last_request_ = 0;
    uint64_t last_window_ = 0;
    void poison() noexcept;
};

// Legacy entrypoint is intentionally removed in v2. Use a PipelineSession
// to prevent attempts to reuse poisoned worker state after failures.

// Deterministic CPU-only stand-in. Never used as GPU inference.
bool mock_execute(const Stage& stage, const Frame& input, Frame& output, std::string& error);

} // namespace strata::heterogeneous
