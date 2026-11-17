#pragma once

// The front stage of a mixed CUDA/HIP layer split (.ai/SEAM.md).
//
// The host process runs the ordinary engine over the later layers.  Its first stage does not start at layer 0: it
// takes its residual from the stage hand-off, as every later stage of a vanilla layer split does.  The layers before
// it run in another process, on a card of the other make.  `FrontStage` is what the host calls so that this front
// stage follows it: each call mirrors the call vanilla makes on its own first stage.
//
// This header has no GPU, engine or transport dependency.

#include <cstdint>
#include <string>

namespace strata::hetero {

/// A session operation of the first stage that the front stage must repeat on its own session.  The arguments `a`
/// and `b` are the ones the mirrored vanilla call takes; they are bound to those calls where the front link is wired
/// into the engine.
enum class SessionOp : int32_t {
    Reset = 1,      ///< a new sequence: the session is zeroed
    Rollback = 2,   ///< the state goes back to position `a`
    Snapshot = 3,   ///< save the state as checkpoint `a` (length `b`)
    Restore = 4,    ///< restore checkpoint `a`
    Truncate = 5,   ///< drop the state past position `a`
    Abandon = 6,    ///< the last window is dropped without a commit (vanilla runs the next window over it)
};

class FrontStage {
public:
    virtual ~FrontStage() = default;

    /// Run the window of `T` tokens at `pos0` on the front layers (Verifier::run / prelaunch of the first stage).
    /// Returns when the front stage has taken the window; it does not wait for the window.  The residual rows reach
    /// the host through the stage hand-off and its ready word, as between two cards of one make.
    virtual bool run(int T, const int32_t* tokens, int64_t pos0, std::string& err) = 0;

    /// Keep the first `n_keep` tokens of the last window (Verifier::commit and wait_commit).  Returns true only when
    /// the front stage has finished that window without an error and has committed it.
    virtual bool commit(int n_keep, std::string& err) = 0;

    /// Read `n` prompt tokens at `pos0` on the front layers (the first stage of the Prefill chain).  `last` marks the
    /// last chunk of the prompt.  Returns when the front stage has written the chunk's rows for the next stage.
    virtual bool prefill(const int32_t* tokens, int64_t n, int64_t pos0, bool last, std::string& err) = 0;

    /// Repeat a session operation on the front stage's session.
    virtual bool session(SessionOp op, int64_t a, int64_t b, std::string& err) = 0;
};

} // namespace strata::hetero
