#pragma once

// A host-only prototype for cross-runtime Strata stage handoffs.
// No CUDA or HIP types cross this boundary.  See README.md for limitations.

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

enum class Kind : uint16_t { data = 1, failure = 2 };
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
};

constexpr uint32_t max_payload_bytes = 16 * 1024 * 1024;
constexpr size_t wire_header_bytes = 56;

std::vector<uint8_t> encode(const Frame& f);
Frame decode(const std::vector<uint8_t>& encoded);

// POSIX AF_UNIX SOCK_STREAM transport: short reads/writes and EINTR handled, no GPU API.
// Receiving an orderly EOF returns nullopt; truncated/invalid frames throw.
// This deliberately copies payloads. Shared-memory/DMA optimization comes later.
void send_frame(int fd, const Frame& frame);
std::optional<Frame> receive_frame(int fd);

using Executor = std::function<bool(const Stage&, const Frame&, Frame&, std::string&)>;

// One worker per process in the intended design, separately linked to CUDA or HIP.
// Worker does not publish a successful reply when its executor reports failure.
void serve_worker(int fd, const Plan& plan, size_t stage_index, const Executor& execute);

// Single request/window synchronous pipeline. All-or-error: no final output until
// each stage has replied successfully. This is NOT yet an asynchronous pipeline.
Frame run_pipeline(const Plan& plan, const std::vector<int>& worker_fds, Frame initial);

// Deterministic CPU-only stand-in. Never used as GPU inference.
bool mock_execute(const Stage& stage, const Frame& input, Frame& output, std::string& error);

} // namespace strata::heterogeneous
