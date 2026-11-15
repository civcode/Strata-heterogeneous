#pragma once

// Opt-in final-stage integration of v4 handoff controls with the standalone
// STFR/v1 result packet. No default engine path, weights, CUDA or HIP runtime.
#include "strata_heterogeneous/host.hpp"
#include "strata_heterogeneous/final_result.hpp"
#include "strata_heterogeneous/mtp_request_policy.hpp"

#include <memory>
#include <string>
#include <vector>

namespace strata::heterogeneous {
struct WindowLimits;

// Final verifier stages return head results, not another R|bo|inject buffer.
// This interface is intentionally separate from StageExecutor::run_window:
// a real final-stage verifier must generate actual predictions, never mock
// an activation response as logits. The input view is a borrowed host mapping.
class FinalStageExecutor : public StageExecutor {
public:
    // Only an explicitly configured native receiver may acknowledge an armed
    // window. No GPU execution is implied by the ordinary begin_window.
    virtual bool receiver_arm_enabled() const noexcept { return false; }
    virtual bool arm_window(std::string& error) { error = "receiver arm unsupported"; return false; }
    virtual bool mtp_relay_enabled() const noexcept { return false; }
    virtual bool mtp_anchor_only_enabled() const noexcept { return false; }
    virtual bool mtp_resident_enabled() const noexcept { return false; }
    virtual bool mtp_service_enabled() const noexcept { return false; }
    virtual bool configure_mtp_request(const Frame&, std::string& error) {
        error = "request-local native MTP unsupported"; return false;
    }
    virtual bool propose_mtp(const WindowSpec&, WindowSpec&, std::string& error) {
        error = "native MTP proposal relay unsupported"; return false;
    }
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
                              bool shared_memory = false,
                              const WindowLimits* limits = nullptr, int startup_wait_ms = -1,
                              bool group_hip = false, NativePrefillHandler prefill = {}, DirectSharedBinder direct_bind = {}, bool guarded_direct = false);

// Owns only protocol state and the optional shared activation slot, never
// child processes or caller-owned file descriptors. The final worker uses
// the existing control socket for STFR/v1 *only* after the negotiated
// final hello; every intermediate stage uses v4 activation frames.
class FinalPipelineSession {
public:
    FinalPipelineSession(Plan plan, std::vector<int> worker_fds,
                         int timeout_ms = default_io_timeout_ms,
                         bool shared_memory = false, bool resident = false,
                         bool fanout_control = false, bool require_compact = false, bool group_hip = false,
                         bool receiver_arm = false, bool mtp_relay = false, bool mtp_anchor_only = false, bool mtp_resident = false, bool direct_verified = false,
                         bool early_window_spec = false, bool guarded_direct = false, bool mtp_service = false);
    ~FinalPipelineSession();
    FinalPipelineSession(const FinalPipelineSession&) = delete;
    FinalPipelineSession& operator=(const FinalPipelineSession&) = delete;
    void initialize();
    // All workers must reset and ACK before a new resident request can run.
    void begin_request(uint64_t request);
    void configure_mtp_request(uint64_t request, const MtpRequestPolicy& policy);
    void native_prefill(const std::vector<int32_t>& tokens, uint32_t chunk, bool isolated_stage_reference=false,
                        int total_timeout_ms=0);
    // Returns provisional head predictions; DOES NOT commit model state.
    FinalStageResult execute(Frame initial, const WindowSpec& spec);
    // Keep the provisional result in the gate without copying discarded logits.
    void execute_discard(Frame initial, const WindowSpec& spec);
    // Returns publishable predictions only after matching prefix ACKs from
    // every stage. n_keep=0 is valid at protocol level but a real verifier
    // must reject it until state restoration is correctly implemented.
    FinalStageResult commit(uint32_t n_keep);
    // One proposal per globally committed prefix; next execute must consume
    // the exact returned anchor and candidates. Explicit diagnostic only.
    WindowSpec request_mtp_proposal();
    bool initialized() const noexcept { return initialized_; }
    bool poisoned() const noexcept { return poisoned_; }
    bool has_pending() const noexcept { return pending_; }
private:
    Plan plan_;
    std::vector<int> fds_;
    std::vector<size_t> endpoints_; // physical terminal-stage identity for each socket
    std::vector<size_t> starts_;
    int timeout_ms_;
    bool shared_memory_;
    bool resident_;
    bool fanout_control_;
    bool require_compact_;
    bool receiver_arm_;
    bool mtp_relay_;
    bool mtp_anchor_only_;
    bool mtp_resident_;
    bool direct_verified_;
    bool early_window_spec_;
    bool guarded_direct_;
    bool mtp_service_;
    std::unique_ptr<const MtpRequestPolicy> mtp_service_policy_;
    uint64_t resident_request_ = 0;
    bool initialized_ = false;
    bool poisoned_ = false;
    bool pending_ = false;
    uint64_t last_request_ = 0;
    uint64_t last_window_ = 0;
    uint64_t pending_request_ = 0;
    uint64_t pending_window_ = 0;
    uint32_t pending_rows_ = 0;
    int64_t pending_pos_ = 0;
    std::optional<WindowSpec> mtp_anchor_, mtp_offer_;
    uint32_t mtp_vocab_ = 0;
    bool mtp_requested_ = false;
    std::vector<int32_t> mtp_pending_inputs_;
    std::unique_ptr<class SharedActivation> shared_;
    FinalResultGate gate_;
    void poison() noexcept;
};

} // namespace strata::heterogeneous
