#pragma once

// CUDA-only final stage for the opt-in STFR/v1 final-result protocol.
// Requires preloaded worker-owned weights, session and residency state.
// No CUDA/HIP runtime objects cross process boundaries.
#include "strata_heterogeneous/cuda_verifier.hpp"
#include "strata_heterogeneous/final_pipeline.hpp"
#include "strata_heterogeneous/layer_taps.hpp"
#include "strata_heterogeneous/layer_zero_probe.hpp"
#include "strata_heterogeneous/stage0_phase_probe.hpp"
#include "strata_heterogeneous/stage_branch_probe.hpp"

#include <memory>

namespace strata::heterogeneous {

struct CudaFinalVerifierOptions {
    // Optional diagnostic logits. A full rows*vocab matrix must fit STFR's
    // 16 MiB cap; otherwise the window fails closed before running a kernel.
    bool include_logits = true; // fail-closed default when GPU nonfinite guard is off
};

// Final-stage counterpart to CudaVerifierDriver. Deliberately not a
// DeviceStageExecutor: final stages produce real head token predictions,
// not another activation frame. Must run in an exec-isolated CUDA worker.
class CudaFinalVerifierExecutor final : public FinalStageExecutor {
public:
    explicit CudaFinalVerifierExecutor(CudaVerifierBindings bindings,
                                       CudaFinalVerifierOptions options = {})
        : bindings_(bindings), options_(options) {}
    ~CudaFinalVerifierExecutor() override { shutdown(); }
    CudaFinalVerifierExecutor(const CudaFinalVerifierExecutor&) = delete;
    CudaFinalVerifierExecutor& operator=(const CudaFinalVerifierExecutor&) = delete;

    bool prepare(const Plan&, size_t stage_index, std::string&) override;
    bool begin_window(const WindowSpec&, uint64_t request_id,
                      uint64_t window_id, std::string&) override;
    bool run_final_window(const Frame&, ActivationView, FinalStageResult&,
                          std::string&) override;
    bool commit_prefix(uint64_t request_id, uint64_t window_id,
                       uint32_t n_keep, std::string&) override;
    bool run_window(const Frame&, Frame&, std::string&) override;
    bool commit_window(uint64_t, uint64_t, std::string&) override;
    void abandon_window() noexcept override;
    void shutdown() noexcept override;

private:
    enum class State { fresh, initializing, ready, begun, provisional, dead };
    CudaVerifierBindings bindings_;
    CudaFinalVerifierOptions options_;
    std::unique_ptr<core::Verifier> verifier_;
    Plan plan_;
    State state_ = State::fresh;
    size_t index_ = 0;
    int device_ = -1;
    float* input_ = nullptr;       // backend-owned, on the last CUDA device
    void* staging_ = nullptr;      // backend-owned pinned host memory
    cudaEvent_t fence_ = nullptr;
    size_t capacity_ = 0;
    WindowSpec spec_;
    LayerTapRecorder taps_;
    LayerZeroProbeRecorder layer_zero_;
    Stage0PhaseRecorder stage0_phases_;
    StageBranchRecorder branch_probe_;
    uint64_t request_ = 0, window_ = 0;
    int64_t next_position_ = 0;
    uint32_t keep_ = 0;
    bool aborted_ = false;
    bool unsafe_result_ = false;

    bool select_device(std::string&);
    bool check(cudaError_t, std::string&);
    bool fence(std::string&);
    bool wait_fence(std::string&);
    void abort_once() noexcept;
};

} // namespace strata::heterogeneous
