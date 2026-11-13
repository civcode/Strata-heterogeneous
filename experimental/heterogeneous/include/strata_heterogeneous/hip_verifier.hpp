#pragma once

// HIP-worker-only header. Never include/link this in a HIP worker or host.
#include "strata_heterogeneous/device.hpp"
#include "strata_heterogeneous/layer_taps.hpp"
#include "strata_heterogeneous/layer_zero_probe.hpp"
#include "strata/core/verify.hpp"
#include <memory>

namespace strata::heterogeneous {

// Already loaded, exclusively owned by this worker. All referenced objects,
// residency tables, expert-pool context and device allocations must outlive
// the driver. Identity is supplied by the loader, NOT derived/attested here.
struct HipVerifierBindings {
    const core::WeightTable& weights;
    const core::ModelGeometry& geometry;
    core::SessionState& session;
    core::VerifyHits hits;
    core::PoolMultiFn pool;
    void* pool_user;
    uint64_t model_fingerprint;
    int max_t = 8;
    int timeout_ms = 30000;
    int64_t initial_position = 0; // position after caller's stage-local prefill/restore
};

// Non-final stages only: protocol v4 has no head-output channel. Construct in
// an exec-isolated, supervised HIP worker; run()/commit()/init() still contain
// blocking engine/runtime calls. The supervisor must kill/reap a hung worker.
// Zero-prefix commit is unsupported and fails closed, never acknowledged.
class HipVerifierDriver final : public DeviceDriver {
public:
    explicit HipVerifierDriver(HipVerifierBindings bindings) : bindings_(bindings) {}
    ~HipVerifierDriver() override { shutdown(); }
    HipVerifierDriver(const HipVerifierDriver&) = delete;
    HipVerifierDriver& operator=(const HipVerifierDriver&) = delete;
    bool prepare(const Plan&, size_t, std::string&) override;
    bool begin_window(const WindowSpec&, uint64_t, uint64_t, std::string&) override;
    bool enqueue_upload(const uint8_t*, size_t, std::string&) override;
    bool wait_upload(std::string&) override;
    bool enqueue_execute(const Frame&, std::string&) override;
    bool wait_execute(std::string&) override;
    bool enqueue_download(uint8_t*, size_t, std::string&) override;
    bool wait_download(std::string&) override;
    bool enqueue_commit(uint64_t, uint64_t, std::string&) override;
    bool enqueue_commit_prefix(uint64_t, uint64_t, uint32_t, std::string&) override;
    bool wait_commit(std::string&) override;
    void abort() noexcept override;
    void shutdown() noexcept override;
private:
    enum class State { fresh, ready, begun, uploading, uploaded, executing,
                       executed, downloading, downloaded, committing, dead };
    HipVerifierBindings bindings_;
    std::unique_ptr<core::Verifier> verifier_;
    Plan plan_;
    size_t index_ = 0;
    State state_ = State::fresh;
    float *input_ = nullptr, *output_ = nullptr;
    void* staging_ = nullptr; // owned pinned memory: DMA never borrows socket/memfd bytes
    cudaEvent_t fence_ = nullptr;
    uint8_t* download_host_ = nullptr;
    WindowSpec spec_;
    LayerTapRecorder taps_;
    LayerZeroProbeRecorder layer_zero_;
    uint64_t request_ = 0, window_ = 0;
    size_t bytes_ = 0;
    int device_ = -1;
    int64_t next_position_ = 0;
    uint32_t keep_ = 0;
    bool expect(State, std::string&);
    bool check(cudaError_t, std::string&);
    bool record(std::string&);
    bool wait(std::string&);
};
} // namespace strata::heterogeneous
