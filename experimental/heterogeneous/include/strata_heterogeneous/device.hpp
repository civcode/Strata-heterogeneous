#pragma once

// Backend-neutral GPU transfer contract for a *single* Strata layer stage.
// This header deliberately includes no CUDA/HIP types and must remain usable
// by host-only builds. A production worker links exactly one GPU runtime.
//
// M1.8 implements the ordering/error boundary, not device kernels, model
// loading, token metadata, KV ownership or GPU-based inference.
#include "strata_heterogeneous/host.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::heterogeneous {

// DeviceDriver owns all backend resources and stage-local allocations.
// Each enqueue may return before work is complete; the corresponding wait
// must prove that the operation completed successfully. Implementations
// must put a finite deadline on every GPU wait, and must not retain a host
// pointer after its matching wait returns.
//
// prepare() must load/validate model identity, stage weights and device
// capabilities before returning success (the CPU test fake does not).
// Compute receives window metadata only: Strata's real Verifier::run also
// needs token IDs, pos0, pool callbacks and commit(n_keep), which are NOT yet
// present in protocol v3. Real inference is impossible until integrated.
class DeviceDriver {
public:
    virtual ~DeviceDriver() = default;
    virtual bool prepare(const Plan& plan, size_t stage_index, std::string& error) = 0;

    // h2d: copy from a borrowed, host-visible staging slot to owned device
    // storage. The pointer may be used until wait_upload succeeds/fails.
    virtual bool enqueue_upload(const uint8_t* host, size_t bytes, std::string& error) = 0;
    virtual bool wait_upload(std::string& error) = 0;

    // Execute only this stage's contiguous layer range. This API proves
    // sequencing, but the current Frame lacks real verifier token metadata.
    virtual bool enqueue_execute(const Frame& metadata, std::string& error) = 0;
    virtual bool wait_execute(std::string& error) = 0;

    // d2h: copy completed stage activations into the borrowed host slot;
    // never signal success until wait_download observes completed transfer.
    virtual bool enqueue_download(uint8_t* host, size_t bytes, std::string& error) = 0;
    virtual bool wait_download(std::string& error) = 0;

    // Stage-local commit, after every pipeline stage provisionally succeeded.
    // Do not treat an enqueue as an acknowledged commit: wait is mandatory.
    virtual bool enqueue_commit(uint64_t request_id, uint64_t window_id,
                                std::string& error) = 0;
    virtual bool wait_commit(std::string& error) = 0;

    // Must not throw. After a failed/partial asynchronous operation abort()
    // must drain or terminate in-flight DMA before borrowed host buffers can
    // be unmapped or reused. If a hung GPU cannot be made safe, terminate the
    // worker process instead of returning with DMA still accessing pointers.
    // Abort does not guarantee rollback of stage-local state or a commit.
    virtual void abort() noexcept = 0;
    virtual void shutdown() noexcept = 0;
};

// Wraps a device-specific driver with protocol validation, transfer fences,
// one-provisional-window ownership, and fail-stop cleanup. The caller owns
// DeviceDriver and must keep it alive until the adapter has shut down.
class DeviceStageExecutor final : public StageExecutor {
public:
    explicit DeviceStageExecutor(DeviceDriver& driver) : driver_(driver) {}
    ~DeviceStageExecutor() override { shutdown(); }
    DeviceStageExecutor(const DeviceStageExecutor&) = delete;
    DeviceStageExecutor& operator=(const DeviceStageExecutor&) = delete;

    bool prepare(const Plan& plan, size_t stage_index, std::string& error) override;
    bool run_window(const Frame& input, Frame& output, std::string& error) override;
    bool run_staged_window(const Frame& metadata, ActivationView view,
                           std::string& error) override;
    bool commit_window(uint64_t request_id, uint64_t window_id, std::string& error) override;
    void abandon_window() noexcept override;
    void shutdown() noexcept override;

    bool poisoned() const { return poisoned_; }
    bool pending() const { return pending_; }
private:
    DeviceDriver& driver_;
    Plan plan_;
    size_t stage_index_ = 0;
    bool prepared_ = false;
    bool pending_ = false;
    bool poisoned_ = false;
    bool shutdown_ = false;
    bool aborted_ = false;
    uint64_t request_id_ = 0;
    uint64_t window_id_ = 0;

    bool fail(const char* phase, std::string& error) noexcept;
    void abort_once() noexcept;
};

} // namespace strata::heterogeneous
