#pragma once

// Opt-in final-stage integration of v4 handoff controls with the standalone
// STFR/v1 result packet. No default engine path, weights, CUDA or HIP runtime.
#include "strata_heterogeneous/host.hpp"
#include "strata_heterogeneous/final_result.hpp"

#include <memory>
#include <string>
#include <vector>

namespace strata::heterogeneous {

// Final verifier stages return head results, not another R|bo|inject buffer.
// This interface is intentionally separate from StageExecutor::run_window:
// a real final-stage verifier must generate actual predictions, never mock
// an activation response as logits. The input view is a borrowed host mapping.
class FinalStageExecutor : public StageExecutor {
public:
    virtual bool run_final_window(const Frame& metadata, ActivationView input,
                                  FinalStageResult& output, std::string& error) = 0;
};

// Runs only in the *last* independently supervised worker. The distinct
// hello_final/hello_shared_final handshake prevents an activation-only peer
// from being mistaken for a final-result worker. Requires window_spec and
// commit_prefix; no legacy full-window path. Returns on orderly socket EOF;
// on any execution or commit failure, abandons/shuts down and closes the
// protocol (best-effort failure frame while still connected).
void serve_final_stage_worker(int fd, const Plan& plan, size_t stage_index,
                              FinalStageExecutor& executor,
                              int timeout_ms = default_io_timeout_ms,
                              bool shared_memory = false);

// Owns only protocol state and the optional shared activation slot, never
// child processes or caller-owned file descriptors. The final worker uses
// the existing control socket for STFR/v1 *only* after the negotiated
// final hello; every intermediate stage uses v4 activation frames.
class FinalPipelineSession {
public:
    FinalPipelineSession(Plan plan, std::vector<int> worker_fds,
                         int timeout_ms = default_io_timeout_ms,
                         bool shared_memory = false);
    ~FinalPipelineSession();
    FinalPipelineSession(const FinalPipelineSession&) = delete;
    FinalPipelineSession& operator=(const FinalPipelineSession&) = delete;
    void initialize();
    // Returns provisional head predictions; DOES NOT commit model state.
    FinalStageResult execute(Frame initial, const WindowSpec& spec);
    // Returns publishable predictions only after matching prefix ACKs from
    // every stage. n_keep=0 is valid at protocol level but a real verifier
    // must reject it until state restoration is correctly implemented.
    FinalStageResult commit(uint32_t n_keep);
    bool initialized() const noexcept { return initialized_; }
    bool poisoned() const noexcept { return poisoned_; }
    bool has_pending() const noexcept { return pending_; }
private:
    Plan plan_;
    std::vector<int> fds_;
    int timeout_ms_;
    bool shared_memory_;
    bool initialized_ = false;
    bool poisoned_ = false;
    bool pending_ = false;
    uint64_t last_request_ = 0;
    uint64_t last_window_ = 0;
    uint64_t pending_request_ = 0;
    uint64_t pending_window_ = 0;
    uint32_t pending_rows_ = 0;
    std::unique_ptr<class SharedActivation> shared_;
    FinalResultGate gate_;
    void poison() noexcept;
};

} // namespace strata::heterogeneous
