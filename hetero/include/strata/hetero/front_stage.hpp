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
#include <vector>

namespace strata::hetero {

/// A session operation of the first stage that the front stage must repeat on its own session.  The arguments `a`
/// and `b` are the ones the mirrored vanilla call takes; they are bound to those calls where the front link is wired
/// into the engine.
enum class SessionOp : int32_t {
    Reset = 1,      ///< a new sequence: the session is zeroed
    Rollback = 2,   ///< the state goes back to position `a` (no call site in vanilla's serve loop: refused)
    Snapshot = 3,   ///< save a checkpoint of the state, which holds the `b` tokens whose token_key is `a`
    Restore = 4,    ///< restore the checkpoint of the `b` tokens whose token_key is `a`
    Truncate = 5,   ///< drop the state past position `a` (no call site in vanilla's serve loop: refused)
    Abandon = 6,    ///< the last window is dropped without a commit (vanilla runs the next window over it)
    Park = 7,       ///< park the whole session (K/V, state, checkpoints) of the `b` tokens whose token_key is `a`
    Unpark = 8,     ///< restore (and drop) the parked session of the `b` tokens whose token_key is `a`
    Drop = 9,       ///< the host's conversation cache dropped (or refused) its copy of the `b` tokens whose token_key is
                    ///< `a`: drop the front stage's.  No copy is not an error.  Leaves the session and an open window alone.
    Steer = 10,     ///< the request's control-vector switch: on (`a` = 1) or off (`a` = 0), vanilla's cvec_set_enabled.
                    ///< The keys of the states saved from then on carry it (steer_key).
};
constexpr SessionOp kLastSessionOp = SessionOp::Steer;
/// Whether `op` ends a window that was not committed (every session operation but Drop does).
constexpr bool ends_window(SessionOp op) { return op != SessionOp::Drop; }

/// A picture in a request (vanilla's ConversationImageKey): its first `<|image_pad|>` position and its hash.
struct ImageKey {
    int64_t start = 0;
    uint64_t hash = 0;
    bool operator==(const ImageKey& o) const { return start == o.start && hash == o.hash; }
    bool operator!=(const ImageKey& o) const { return !(*this == o); }
};

/// A picture's place in the prompt: its key and how many `<|image_pad|>` tokens (= embedding rows) it covers.
struct ImageSpan {
    int64_t start = 0;
    uint64_t hash = 0;
    int64_t rows = 0;
};

/// A request's image input as the host built it (vanilla's per-request vision setup in generate.cpp), sent to the
/// front stage before the request's first prompt chunk.  A text request sends no images and no table: the front stage
/// puts the identity table back if it holds another.
struct FrontImages {
    int64_t n = 0;                   ///< the prompt's tokens (the image rows are indexed by prompt position below it)
    std::vector<ImageSpan> images;   ///< in prompt order
    std::vector<int32_t> mrope;      ///< the (t, h, w) position of every cell; empty for a text request
    std::vector<float> rows;         ///< the image rows in prompt order, n_embd floats each (sum of `rows` rows)
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

    /// The request's image positions, rows and keys (or none, for a text request).  Ends a window not committed.
    virtual bool images(FrontImages& im, std::string& err) = 0;
};

} // namespace strata::hetero
