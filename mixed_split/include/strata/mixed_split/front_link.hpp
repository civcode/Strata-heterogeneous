#pragma once

// The wire between the host process and the front stage worker of a mixed CUDA/HIP split (.ai/SEAM.md).
//
// One AF_UNIX stream socket.  The host sends one request at a time and waits for its reply; the worker serves them
// in order.  Residual rows do not travel here: they go through the shared hand-off memory (shared_region.hpp).
// Host-only code: nothing here touches CUDA or HIP.

#include "strata/mixed_split/front_stage.hpp"
#include "strata/mixed_split/shared_region.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::mixed_split {

constexpr uint32_t kLinkProtocol = 4;   // 2: Images; 3: PlRun / PlCommit / PlUndo, a hand-off per window parity;
                                        // 4: PfStart / PfWait, two prompt row buffers
constexpr uint32_t kLinkMagic = 0x314c4653;          // "SFL1"
constexpr size_t kLinkMaxPayload = size_t(16) << 20; // tokens of one prompt chunk, with room to spare

/// What both processes must agree on before the first window.  The worker compares it with what it loaded.
struct LinkIdentity {
    uint32_t protocol = kLinkProtocol;
    uint64_t model_fingerprint = 0;   ///< of the model files both processes read
    int32_t n_layers = 0;
    int32_t front_end = 0;            ///< the front stage runs layers [0, front_end)
    int32_t hc = 0;
    int32_t n_embd = 0;
    int32_t max_rows = 0;             ///< rows a hand-off buffer holds
    int64_t max_context = 0;
    bool operator==(const LinkIdentity& o) const {
        return protocol == o.protocol && model_fingerprint == o.model_fingerprint && n_layers == o.n_layers &&
               front_end == o.front_end && hc == o.hc && n_embd == o.n_embd && max_rows == o.max_rows &&
               max_context == o.max_context;
    }
    bool operator!=(const LinkIdentity& o) const { return !(*this == o); }
    /// Plausible on its own (ranges only); `why` says what is not.
    bool valid(std::string& why) const;
};

/// A value for LinkIdentity::model_fingerprint that both engines compute alike: FNV-1a over the pack's index.txt and
/// the sizes of `files` (the model's GGUF shards), in that order.  Never 0.
uint64_t model_fingerprint(const std::string& pack_dir, const std::vector<std::string>& files);

/// Images: a request's FrontImages, serialized and sent in pieces of at most kLinkMaxPayload bytes (`a`: the total,
/// `b`: the piece's offset); each piece is answered, the last one after the worker's backend took the whole.
/// PlRun (`a` T, `b` pos0, `c` parity, the tokens), PlCommit (`a` n_keep, `c` parity), PlUndo (`c` parity): the
/// pipelined windows (FrontStage::pl_run, pl_commit, pl_undo).  PfStart (`a` n, `b` pos0, `c` buffer, the tokens),
/// PfWait (`c` buffer): a prompt chunk read ahead (FrontStage::prefill_start, prefill_wait).  Every other operation
/// ends a chunk in flight that nobody waited for.
enum class Op : uint16_t {
    Hello = 1, Run = 2, Commit = 3, Prefill = 4, Session = 5, Shutdown = 6, Images = 7, PlRun = 8, PlCommit = 9, PlUndo = 10,
    PfStart = 11, PfWait = 12, Reply = 100
};

/// FrontImages on the wire (little-endian, both processes on one host).  `decode` checks the layout against the
/// identity: positions inside the prompt, images in order and apart, n_embd floats per row.
std::vector<uint8_t> encode_images(const FrontImages& im);
bool decode_images(const std::vector<uint8_t>& bytes, const LinkIdentity& id, FrontImages& im, std::string& why);

struct Frame {
    Op op = Op::Reply;
    uint64_t seq = 0;
    int64_t a = 0, b = 0, c = 0;
    std::vector<uint8_t> payload;
};

enum class Recv { Ok, Eof, Error };

/// Send one frame; `pass_fd` >= 0 travels with it (SCM_RIGHTS).  `timeout_ms` < 0 waits without a limit.
bool send_frame(int fd, const Frame& f, int pass_fd, int timeout_ms, std::string& err);
/// Receive one frame.  `got_fd` (may be null) receives a passed descriptor or -1; the caller owns it.
/// Eof: the peer closed the socket before the first byte of a frame.
Recv recv_frame(int fd, Frame& f, int* got_fd, int timeout_ms, std::string& err);

/// The worker's side of the front stage: the forwarded operations, plus the hand-off memory it is given.
class FrontBackend : public FrontStage {
public:
    /// Called once, before any other operation.  Refuse (false, `err`) when `id` is not what this worker loaded.
    virtual bool attach(const LinkIdentity& id, SharedRegion&& region, std::string& err) = 0;
};

/// Serve the host's requests on `fd` until it closes the socket or sends Shutdown.  Returns 0 after such a clean
/// end and 1 on a protocol or transport error (`err` says which).  An operation the backend refuses is answered
/// with its error and serving goes on: the host decides what follows.
int serve_front(int fd, FrontBackend& backend, std::string& err);

/// The same in two steps, for a worker that needs the hand-off memory before it can build what serves the link (the
/// engine captures its window graphs with the hand-off's device address).  `front_accept` waits for the host's hello,
/// maps the region and returns the identity the host sent; nothing is answered yet.  The worker compares the identity
/// and answers with `front_accept_reply`; after an `ok` answer `serve_front_greeted` serves the other operations
/// (`backend.attach` is not called).  Ok / Eof (the host closed the socket first) / Error.
Recv front_accept(int fd, LinkIdentity& id, SharedRegion& region, uint64_t& seq, std::string& err);
bool front_accept_reply(int fd, uint64_t seq, bool ok, const std::string& why, std::string& err);
int serve_front_greeted(int fd, FrontBackend& backend, const LinkIdentity& id, std::string& err);

/// The host's side: a FrontStage whose calls travel to the worker.
class RemoteFrontStage final : public FrontStage {
public:
    /// `fd` is borrowed.  `timeout_ms` bounds every wait for a reply (< 0: no limit).
    RemoteFrontStage(int fd, int timeout_ms) : fd_(fd), timeout_ms_(timeout_ms) {}

    /// Identity and hand-off memory; must succeed before any other call.
    bool hello(const LinkIdentity& id, const SharedRegion& region, std::string& err);

    bool run(int T, const int32_t* tokens, int64_t pos0, std::string& err) override;
    bool commit(int n_keep, std::string& err) override;
    bool prefill(const int32_t* tokens, int64_t n, int64_t pos0, bool last, std::string& err) override;
    bool session(SessionOp op, int64_t a, int64_t b, std::string& err) override;
    bool images(FrontImages& im, std::string& err) override;
    bool pl_run(int parity, int T, const int32_t* tokens, int64_t pos0, std::string& err) override;
    bool pl_commit(int parity, int n_keep, std::string& err) override;
    bool pl_undo(int parity, std::string& err) override;
    bool prefill_start(const int32_t* tokens, int64_t n, int64_t pos0, int buf, std::string& err) override;
    bool prefill_wait(int buf, std::string& err) override;

    /// Ask the worker to end its serve loop.  After this (or any transport error) the link is closed for calls.
    bool shutdown(std::string& err);

    /// A transport or protocol error has closed the link; every later call fails with the same message.
    bool broken() const { return broken_; }

private:
    int fd_;
    int timeout_ms_;
    uint64_t seq_ = 0;
    bool greeted_ = false;
    bool broken_ = false;
    std::string broken_why_;
    bool call(Frame& req, int pass_fd, std::string& err);
};

} // namespace strata::mixed_split
