#pragma once

// A host-only prototype for cross-runtime Strata stage handoffs.
// No CUDA or HIP types cross this boundary.  See README.md for limitations.

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::heterogeneous {
struct WindowLimits; // M1.18 optional CPU-only, strictly validated CUDA admission

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

// Protocol v4: explicit verifier window metadata and accepted-prefix commit.
// v3 workers are intentionally rejected by the v4 protocol version.
// Socket data carries R/bo/inject floats; shared_data carries a 32-bit digest.
// The memfd is delivered separately via SCM_RIGHTS immediately after ready.
enum class Kind : uint16_t {
    data = 1, failure = 2, hello = 3, ready = 4,
    commit = 5, committed = 6, shared_data = 7, hello_shared = 8,
    window_spec = 9, commit_prefix = 10, committed_prefix = 11,
    hello_final = 12, hello_shared_final = 13,
    ready_resident = 14, begin_request = 15, request_ready = 16,
    prefill_spec = 17, prefill_rows = 18, prefill_done = 19
};
// Input to Strata's Verifier::run(T,tokens,pos0). The number of accepted
// tokens is NOT known until execution completes and is chosen at commit.
// Token IDs must be nonnegative and fit the loaded model's vocabulary (the
// actual vocabulary check is the responsibility of the GPU worker).
struct WindowSpec {
    int64_t pos0 = 0;
    std::vector<int32_t> token_ids;
    void validate(uint32_t rows) const;
};
constexpr uint32_t max_native_prefill_chunk = 2048; // independent of decode max_t
constexpr uint32_t max_window_tokens = 8192; // protocol ceiling, not Verifier::max_t

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
    // hello/hello_shared: [backend, ordinal, first_layer, end_layer]; others: zero.
    std::array<uint32_t, 4> identity{};
    uint32_t shared_checksum = 0; // shared_data only, over mapped float32 bytes
    std::optional<WindowSpec> window; // window_spec only
    uint32_t keep_count = 0; // commit_prefix and committed_prefix only
    uint32_t prefill_chunk = 0; // prefill_spec only
};

constexpr uint32_t max_payload_bytes = 16 * 1024 * 1024;
constexpr size_t wire_header_bytes = 56;
constexpr int default_io_timeout_ms = 30000;
// Native prefill transfers R only. A native chunk can span several frames;
// residual frame window_id is the piece's absolute position plus one.
uint32_t prefill_piece_rows(Geometry);
using NativePrefillHandler = std::function<bool(const Frame&, int, std::string&)>;

// A borrowed view of one host-visible activation window. Bytes avoid treating
// a Linux mmap as a constructed float[] (C++17 object-lifetime rules) and give
// GPU adapters a runtime-neutral D2H/H2D staging interface. This is NOT device
// memory and cannot be passed directly to a CUDA or HIP kernel.
struct ByteRange {
    uint8_t* data = nullptr;
    size_t bytes = 0;
    float read_f32(size_t element) const;
    void write_f32(size_t element, float value) const;
};
struct HandoffRow {
    ByteRange residual;     // hc * n_embd float32 values
    ByteRange pending_write; // n_embd float32 values
    ByteRange inject;       // hc float32 values
};
class ActivationView {
public:
    // Bounds, geometry and the full 16 MiB protocol cap are validated.
    // The borrowed pointer must remain alive for the entire view lifetime.
    ActivationView(uint8_t* data, size_t bytes, Geometry geometry, uint32_t rows);
    HandoffRow row(uint32_t index) const;
    uint8_t* data() const { return data_; }
    size_t bytes() const { return bytes_; }
    uint32_t rows() const { return rows_; }
    Geometry geometry() const { return geometry_; }
private:
    uint8_t* data_;
    size_t bytes_;
    Geometry geometry_;
    uint32_t rows_;
    size_t stride_bytes_;
};

// Exactly the same host and frame checks as encode(), without serializing.
void validate_wire_frame(const Frame& f);
std::vector<uint8_t> encode(const Frame& f);
Frame decode(const std::vector<uint8_t>& encoded);

// POSIX AF_UNIX SOCK_STREAM transport: short reads/writes and EINTR handled, no GPU API.
// Receiving an orderly EOF returns nullopt; truncated/invalid frames throw.
// Socket-data mode copies payloads; shared_data uses an explicitly staged memfd.
// Total, monotonic wall-clock deadline per frame, not per read/write chunk.
// timeout_ms must be >= 0. A timeout throws; callers must poison the session.
void send_frame(int fd, const Frame& frame, int timeout_ms = default_io_timeout_ms);
std::optional<Frame> receive_frame(int fd, int timeout_ms = default_io_timeout_ms);
// Resident worker only: wait for first request byte/EOF while idle, then
// keep the usual bounded frame deadline. Never use during a pending window.
std::optional<Frame> receive_idle_frame(int fd, int timeout_ms);
// Startup barrier: separate bounded wait for the first READY byte; the
// complete frame still uses the ordinary per-frame I/O deadline.
std::optional<Frame> receive_startup_frame(int fd, int timeout_ms, int startup_wait_ms);

using Executor = std::function<bool(const Stage&, const Frame&, Frame&, std::string&)>;

// The stage owns layer-local state and may have a prepared, uncommitted window.
// run_window must not advance durable state; commit_window does so only when all
// stages produced valid outputs. A failure during commit poisons the session;
// rollback across partially committed GPU workers is NOT promised.
class StageExecutor {
public:
    virtual ~StageExecutor() = default;
    virtual bool prepare(const Plan& plan, size_t stage_index, std::string& error) = 0;
    // Explicit resident request boundary; unsupported executors fail closed.
    virtual bool reset_request(uint64_t, std::string& error) {
        error = "executor does not support resident requests"; return false;
    }
    virtual bool run_window(const Frame& input, Frame& output, std::string& error) = 0;
    // Optional in-place shared activation path. Metadata carries IDs/geometry
    // but no float payload; borrowed bytes contain the exact R|bo|inject rows.
    // No durable state may advance before commit_window. This default method
    // safely falls back to vector run_window plus validated memcpy; a CUDA/HIP
    // executor can override it to stage bytes to backend-owned device buffers.
    virtual bool run_staged_window(const Frame& metadata, ActivationView view, std::string& error);
    virtual bool commit_window(uint64_t request_id, uint64_t window_id, std::string& error) = 0;
    // Optional v4 verifier metadata support. Default rejects rather than
    // silently pretending that a real model can run without token positions.
    virtual bool begin_window(const WindowSpec& spec, uint64_t request_id,
                              uint64_t window_id, std::string& error);
    virtual bool commit_prefix(uint64_t request_id, uint64_t window_id,
                               uint32_t n_keep, std::string& error);
    virtual void abandon_window() noexcept = 0;
    virtual void shutdown() noexcept = 0;
};

// Stateless test adapter, not a real GPU-backed stage.
class MockStageExecutor final : public StageExecutor {
public:
    bool prepare(const Plan& plan, size_t stage_index, std::string& error) override;
    bool reset_request(uint64_t, std::string&) override;
    bool run_window(const Frame& input, Frame& output, std::string& error) override;
    bool run_staged_window(const Frame& metadata, ActivationView view, std::string& error) override;
    bool commit_window(uint64_t request_id, uint64_t window_id, std::string& error) override;
    bool begin_window(const WindowSpec& spec, uint64_t request_id,
                      uint64_t window_id, std::string& error) override;
    bool commit_prefix(uint64_t request_id, uint64_t window_id,
                       uint32_t n_keep, std::string& error) override;
    void abandon_window() noexcept override;
    void shutdown() noexcept override;
private:
    Stage stage_{};
    bool prepared_ = false;
    bool pending_ = false;
    uint64_t pending_request_ = 0;
    uint64_t pending_window_ = 0;
    bool has_spec_ = false;
    WindowSpec window_spec_;
};

// Workers send an identity hello before accepting data. Coordinator verifies
// every worker's declared backend/ordinal/layer range, then sends ready.
void serve_worker_stateful(int fd, const Plan& plan, size_t stage_index,
                           StageExecutor& executor, int timeout_ms = default_io_timeout_ms,
                           bool shared_memory = false,
                           const WindowLimits* limits = nullptr, int startup_wait_ms = -1,
                            NativePrefillHandler prefill = {});
// Backwards-compatible CPU-test convenience; now uses the same transactional
// handshake/commit path as StageExecutor. It is NOT a GPU adapter.
void serve_worker(int fd, const Plan& plan, size_t stage_index, const Executor& execute);

// Owns only the *protocol state*, not the child processes or socket descriptors.
// On any failure, shuts down sockets, rejects later windows, and requires new
// workers/session; it never publishes a partially completed window.
class PipelineSession {
public:
    PipelineSession(Plan plan, std::vector<int> worker_fds, int timeout_ms = default_io_timeout_ms,
                    bool shared_memory = false);
    ~PipelineSession();
    void initialize();
    // Legacy one-call CPU path: implicitly commits all rows (v4 control
    // with old commit kind); not valid for real speculative decode.
    Frame run_window(Frame initial);
    // Verifier-style two-phase path: execute stages with token/position
    // metadata, inspect provisional output, then choose n_keep in [0,rows].
    // The provisional result is NOT published/committed model state.
    Frame execute_window(Frame initial, const WindowSpec& spec);
    Frame commit_pending(uint32_t n_keep);
    bool has_pending() const { return pending_; }
    bool poisoned() const { return poisoned_; }
    bool initialized() const { return initialized_; }
private:
    Plan plan_;
    std::vector<int> fds_;
    int timeout_ms_;
    bool initialized_ = false;
    bool poisoned_ = false;
    bool shared_memory_ = false;
    std::unique_ptr<class SharedActivation> shared_;
    uint64_t last_request_ = 0;
    uint64_t last_window_ = 0;
    bool pending_ = false;
    bool pending_spec_ = false;
    Frame provisional_;
    void poison() noexcept;
    Frame execute_impl(Frame initial, const WindowSpec* spec);
    Frame commit_impl(uint32_t n_keep);
};

// Legacy entrypoint is intentionally removed in v2. Use a PipelineSession
// to prevent attempts to reuse poisoned worker state after failures.

// Deterministic CPU-only stand-in. Never used as GPU inference.
bool mock_execute(const Stage& stage, const Frame& input, Frame& output, std::string& error);

} // namespace strata::heterogeneous
