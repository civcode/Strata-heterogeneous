#include "strata/hetero/front_link.hpp"

#include <cstdio>
#include <sys/stat.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <poll.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>
#include <utility>

#if !defined(__BYTE_ORDER__) || __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "the front link's wire format is little-endian; both processes run on the same little-endian host"
#endif

namespace strata::hetero {
namespace {

constexpr size_t kHeader = 48;

using Clock = std::chrono::steady_clock;

struct Deadline {
    bool bounded;
    Clock::time_point at;
    explicit Deadline(int timeout_ms)
        : bounded(timeout_ms >= 0), at(Clock::now() + std::chrono::milliseconds(timeout_ms < 0 ? 0 : timeout_ms)) {}
    /// poll()'s timeout argument: what is left, -1 without a limit
    int left() const {
        if (!bounded) return -1;
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(at - Clock::now()).count();
        return ms <= 0 ? 0 : (int) ms;
    }
};

bool wait_io(int fd, short events, const Deadline& d, std::string& err) {
    for (;;) {
        struct pollfd p {};
        p.fd = fd;
        p.events = events;
        const int rc = ::poll(&p, 1, d.left());
        if (rc > 0) return true;   // readable/writable, or hung up: the read or write that follows reports it
        if (rc == 0) { err = "front link: timed out"; return false; }
        if (errno != EINTR) { err = std::string("front link: poll: ") + std::strerror(errno); return false; }
    }
}

template <typename T> void put(uint8_t*& p, T v) { std::memcpy(p, &v, sizeof v); p += sizeof v; }
template <typename T> T get(const uint8_t*& p) { T v; std::memcpy(&v, p, sizeof v); p += sizeof v; return v; }

bool write_rest(int fd, const uint8_t* p, size_t n, const Deadline& d, std::string& err) {
    while (n > 0) {
        if (!wait_io(fd, POLLOUT, d, err)) return false;
        const ssize_t w = ::send(fd, p, n, MSG_NOSIGNAL);
        if (w < 0) {
            if (errno == EINTR || errno == EAGAIN) continue;
            err = std::string("front link: send: ") + std::strerror(errno);
            return false;
        }
        p += w;
        n -= (size_t) w;
    }
    return true;
}

/// Reads exactly `n` bytes.  `eof_at_start` (may be null) is set when the peer closed before the first byte.
bool read_exact(int fd, uint8_t* p, size_t n, const Deadline& d, int* got_fd, bool* eof_at_start, std::string& err) {
    size_t have = 0;
    if (eof_at_start != nullptr) *eof_at_start = false;
    while (have < n) {
        if (!wait_io(fd, POLLIN, d, err)) return false;
        struct iovec iov {};
        iov.iov_base = p + have;
        iov.iov_len = n - have;
        alignas(struct cmsghdr) char ctl[CMSG_SPACE(sizeof(int))];
        struct msghdr msg {};
        msg.msg_iov = &iov;
        msg.msg_iovlen = 1;
        if (got_fd != nullptr) { msg.msg_control = ctl; msg.msg_controllen = sizeof ctl; }
        const ssize_t r = ::recvmsg(fd, &msg, MSG_CMSG_CLOEXEC);
        if (r < 0) {
            if (errno == EINTR || errno == EAGAIN) continue;
            err = std::string("front link: recv: ") + std::strerror(errno);
            return false;
        }
        if (got_fd != nullptr)
            for (struct cmsghdr* c = CMSG_FIRSTHDR(&msg); c != nullptr; c = CMSG_NXTHDR(&msg, c))
                if (c->cmsg_level == SOL_SOCKET && c->cmsg_type == SCM_RIGHTS && c->cmsg_len >= CMSG_LEN(sizeof(int))) {
                    int v = -1;
                    std::memcpy(&v, CMSG_DATA(c), sizeof v);
                    if (*got_fd >= 0) (void) ::close(*got_fd);
                    *got_fd = v;
                }
        if (r == 0) {
            if (have == 0 && eof_at_start != nullptr) { *eof_at_start = true; return false; }
            err = "front link: the peer closed the socket inside a frame";
            return false;
        }
        have += (size_t) r;
    }
    return true;
}

bool send_reply(int fd, uint64_t seq, bool ok, const std::string& text, std::string& err) {
    Frame r;
    r.op = Op::Reply;
    r.seq = seq;
    r.a = ok ? 0 : 1;
    r.payload.assign(text.begin(), text.end());
    return send_frame(fd, r, -1, -1, err);
}

} // namespace

bool LinkIdentity::valid(std::string& why) const {
    if (n_layers <= 1 || n_layers > 4096) { why = "bad layer count"; return false; }
    if (front_end <= 0 || front_end >= n_layers) { why = "the front stage must hold some but not all layers"; return false; }
    if (hc <= 0 || hc > 64 || n_embd <= 0 || n_embd > (1 << 20)) { why = "bad model geometry"; return false; }
    if (max_rows <= 0 || max_rows > (1 << 20)) { why = "bad hand-off row count"; return false; }
    if (max_context <= 0) { why = "bad context size"; return false; }
    return true;
}

bool send_frame(int fd, const Frame& f, int pass_fd, int timeout_ms, std::string& err) {
    if (f.payload.size() > kLinkMaxPayload) { err = "front link: payload too large"; return false; }
    uint8_t head[kHeader];
    uint8_t* p = head;
    put<uint32_t>(p, kLinkMagic);
    put<uint16_t>(p, (uint16_t) f.op);
    put<uint16_t>(p, 0);
    put<uint64_t>(p, f.seq);
    put<int64_t>(p, f.a);
    put<int64_t>(p, f.b);
    put<int64_t>(p, f.c);
    put<uint32_t>(p, (uint32_t) f.payload.size());
    put<uint32_t>(p, 0);
    const Deadline d(timeout_ms);
    size_t sent = 0;
    if (pass_fd >= 0) {   // the descriptor rides on the header's first bytes
        for (;;) {
            if (!wait_io(fd, POLLOUT, d, err)) return false;
            struct iovec iov {};
            iov.iov_base = head;
            iov.iov_len = kHeader;
            alignas(struct cmsghdr) char ctl[CMSG_SPACE(sizeof(int))];
            std::memset(ctl, 0, sizeof ctl);
            struct msghdr msg {};
            msg.msg_iov = &iov;
            msg.msg_iovlen = 1;
            msg.msg_control = ctl;
            msg.msg_controllen = sizeof ctl;
            struct cmsghdr* c = CMSG_FIRSTHDR(&msg);
            c->cmsg_level = SOL_SOCKET;
            c->cmsg_type = SCM_RIGHTS;
            c->cmsg_len = CMSG_LEN(sizeof(int));
            std::memcpy(CMSG_DATA(c), &pass_fd, sizeof pass_fd);
            const ssize_t w = ::sendmsg(fd, &msg, MSG_NOSIGNAL);
            if (w < 0) {
                if (errno == EINTR || errno == EAGAIN) continue;
                err = std::string("front link: sendmsg: ") + std::strerror(errno);
                return false;
            }
            sent = (size_t) w;
            break;
        }
    }
    if (!write_rest(fd, head + sent, kHeader - sent, d, err)) return false;
    return f.payload.empty() || write_rest(fd, f.payload.data(), f.payload.size(), d, err);
}

Recv recv_frame(int fd, Frame& f, int* got_fd, int timeout_ms, std::string& err) {
    int passed = -1;
    uint8_t head[kHeader];
    const Deadline d(timeout_ms);
    bool eof = false;
    auto fail = [&]() {
        if (passed >= 0) (void) ::close(passed);
        if (got_fd != nullptr) *got_fd = -1;
        return eof ? Recv::Eof : Recv::Error;
    };
    if (!read_exact(fd, head, kHeader, d, &passed, &eof, err)) return fail();
    const uint8_t* p = head;
    const uint32_t magic = get<uint32_t>(p);
    const uint16_t op = get<uint16_t>(p);
    (void) get<uint16_t>(p);
    f.seq = get<uint64_t>(p);
    f.a = get<int64_t>(p);
    f.b = get<int64_t>(p);
    f.c = get<int64_t>(p);
    const uint32_t len = get<uint32_t>(p);
    if (magic != kLinkMagic) { err = "front link: not a front-link frame"; return fail(); }
    if (len > kLinkMaxPayload) { err = "front link: payload too large"; return fail(); }
    f.op = (Op) op;
    f.payload.resize(len);
    if (len > 0 && !read_exact(fd, f.payload.data(), len, d, nullptr, nullptr, err)) return fail();
    if (got_fd != nullptr) *got_fd = passed;
    else if (passed >= 0) (void) ::close(passed);
    return Recv::Ok;
}

namespace {

// The hello's payload -> identity and the mapped region.  Takes ownership of `got` (the passed descriptor).
bool parse_hello(const Frame& f, int got, LinkIdentity& id, SharedRegion& region, std::string& why) {
    auto drop = [&](const char* w) {
        if (got >= 0) (void) ::close(got);
        why = w;
        return false;
    };
    const uint8_t* p = f.payload.data();
    constexpr size_t fixed = 4 + 4 + 8 + 4 * 6 + 8 + 8;
    if (f.payload.size() < fixed) return drop("short hello");
    id.protocol = get<uint32_t>(p);
    (void) get<uint32_t>(p);
    id.model_fingerprint = get<uint64_t>(p);
    id.n_layers = get<int32_t>(p);
    id.front_end = get<int32_t>(p);
    id.hc = get<int32_t>(p);
    id.n_embd = get<int32_t>(p);
    id.max_rows = get<int32_t>(p);
    (void) get<int32_t>(p);
    id.max_context = get<int64_t>(p);
    const uint64_t n_slots = get<uint64_t>(p);
    std::vector<size_t> sizes;
    if (n_slots == 0 || n_slots > SharedRegion::kMaxSlots || f.payload.size() != fixed + n_slots * 8)
        return drop("bad hello slot table");
    for (uint64_t i = 0; i < n_slots; ++i) sizes.push_back((size_t) get<uint64_t>(p));
    if (id.protocol != kLinkProtocol) return drop("protocol version mismatch");
    std::string e;
    if (!id.valid(e)) { if (got >= 0) (void) ::close(got); why = "identity: " + e; return false; }
    if (got < 0) { why = "the hello carries no hand-off memory"; return false; }
    return SharedRegion::adopt(got, sizes, region, why);   // adopt owns `got` from here
}

int serve_loop(int fd, FrontBackend& backend, std::string& err, bool greeted, LinkIdentity id) {
    int window = 0;   // the rows of the window that waits for its commit
    for (;;) {
        Frame f;
        int got = -1;
        const Recv r = recv_frame(fd, f, &got, -1, err);
        if (r == Recv::Eof) return 0;   // the host closed the socket: a clean end
        if (r == Recv::Error) return 1;
        if (f.op != Op::Hello && got >= 0) { (void) ::close(got); got = -1; }
        std::string e;
        auto fatal = [&](const std::string& why) {
            std::string ignored;
            (void) send_reply(fd, f.seq, false, why, ignored);
            err = "front link: " + why;
            return 1;
        };
        if (f.op == Op::Hello) {
            if (greeted) { if (got >= 0) (void) ::close(got); return fatal("a second hello"); }
            SharedRegion region;
            if (!parse_hello(f, got, id, region, e)) return fatal(e);
            if (!backend.attach(id, std::move(region), e)) return fatal("refused: " + e);
            greeted = true;
            if (!send_reply(fd, f.seq, true, "", err)) return 1;
            continue;
        }
        if (!greeted) return fatal("an operation before the hello");
        bool ok = false;
        switch (f.op) {
        case Op::Run: {
            const int64_t T = f.a, pos0 = f.b;
            if (T < 1 || T > id.max_rows || f.payload.size() != (size_t) T * sizeof(int32_t) || pos0 < 0 ||
                pos0 + T > id.max_context) {
                e = "bad window";
                break;
            }
            if (window != 0) { e = "a window is still waiting for its commit"; break; }
            std::vector<int32_t> tok((size_t) T);
            std::memcpy(tok.data(), f.payload.data(), f.payload.size());
            ok = backend.run((int) T, tok.data(), pos0, e);
            if (ok) window = (int) T;
            break;
        }
        case Op::Commit: {
            if (window == 0) { e = "no window to commit"; break; }
            if (f.a < 1 || f.a > window) { e = "bad commit length"; break; }
            ok = backend.commit((int) f.a, e);
            window = 0;   // a failed commit leaves no window either: the host resets the session before it goes on
            break;
        }
        case Op::Prefill: {
            const int64_t n = f.a, pos0 = f.b;
            if (n < 1 || f.payload.size() != (size_t) n * sizeof(int32_t) || pos0 < 0 || pos0 + n > id.max_context) {
                e = "bad prompt chunk";
                break;
            }
            if (window != 0) { e = "a window is still waiting for its commit"; break; }
            std::vector<int32_t> tok((size_t) n);
            std::memcpy(tok.data(), f.payload.data(), f.payload.size());
            ok = backend.prefill(tok.data(), n, pos0, f.c != 0, e);
            break;
        }
        case Op::Session: {
            if (f.a < (int64_t) SessionOp::Reset || f.a > (int64_t) SessionOp::Abandon) { e = "bad session operation"; break; }
            ok = backend.session((SessionOp) f.a, f.b, f.c, e);
            if (ok) window = 0;   // every session operation ends a window that was not committed
            break;
        }
        case Op::Shutdown:
            if (!send_reply(fd, f.seq, true, "", err)) return 1;
            return 0;
        default:
            return fatal("unknown operation");
        }
        if (!ok && e.empty()) e = "the front stage failed without a message";
        if (!send_reply(fd, f.seq, ok, ok ? "" : e, err)) return 1;
    }
}

} // namespace

int serve_front(int fd, FrontBackend& backend, std::string& err) { return serve_loop(fd, backend, err, false, LinkIdentity{}); }

int serve_front_greeted(int fd, FrontBackend& backend, const LinkIdentity& id, std::string& err) {
    return serve_loop(fd, backend, err, true, id);
}

Recv front_accept(int fd, LinkIdentity& id, SharedRegion& region, uint64_t& seq, std::string& err) {
    Frame f;
    int got = -1;
    const Recv r = recv_frame(fd, f, &got, -1, err);
    if (r != Recv::Ok) return r;
    seq = f.seq;
    std::string why;
    if (f.op != Op::Hello) {
        if (got >= 0) (void) ::close(got);
        why = "an operation before the hello";
    } else if (parse_hello(f, got, id, region, why)) {
        return Recv::Ok;
    }
    std::string ignored;
    (void) send_reply(fd, f.seq, false, why, ignored);
    err = "front link: " + why;
    return Recv::Error;
}

bool front_accept_reply(int fd, uint64_t seq, bool ok, const std::string& why, std::string& err) {
    return send_reply(fd, seq, ok, ok ? "" : "refused: " + why, err);
}

uint64_t model_fingerprint(const std::string& pack_dir, const std::vector<std::string>& files) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](const void* p, size_t n) {
        const uint8_t* b = static_cast<const uint8_t*>(p);
        for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    };
    if (std::FILE* f = std::fopen((pack_dir + "/index.txt").c_str(), "rb")) {
        char buf[65536];
        size_t n = 0;
        while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) mix(buf, n);
        std::fclose(f);
    }
    for (const std::string& p : files) {
        struct stat st {};
        const uint64_t v = ::stat(p.c_str(), &st) == 0 ? (uint64_t) st.st_size : ~0ull;
        mix(&v, sizeof v);
    }
    return h == 0 ? 1 : h;
}

bool RemoteFrontStage::call(Frame& req, int pass_fd, std::string& err) {
    if (broken_) { err = broken_why_; return false; }
    auto broke = [&](const std::string& why) {
        broken_ = true;
        broken_why_ = why;
        err = why;
        return false;
    };
    req.seq = ++seq_;
    std::string e;
    if (!send_frame(fd_, req, pass_fd, timeout_ms_, e)) return broke(e);
    Frame rep;
    const Recv r = recv_frame(fd_, rep, nullptr, timeout_ms_, e);
    if (r == Recv::Eof) return broke("front link: the front stage worker closed the socket");
    if (r == Recv::Error) return broke(e);
    if (rep.op != Op::Reply || rep.seq != req.seq) return broke("front link: a reply out of order");
    if (rep.a != 0) {
        err.assign(rep.payload.begin(), rep.payload.end());
        if (err.empty()) err = "the front stage failed without a message";
        return false;
    }
    return true;
}

bool RemoteFrontStage::hello(const LinkIdentity& id, const SharedRegion& region, std::string& err) {
    if (greeted_) { err = "front link: a second hello"; return false; }
    std::string why;
    if (!id.valid(why)) { err = "front link: identity: " + why; return false; }
    if (!region.valid()) { err = "front link: no hand-off memory"; return false; }
    Frame f;
    f.op = Op::Hello;
    const size_t n = region.slots();
    f.payload.resize(4 + 4 + 8 + 4 * 6 + 8 + 8 + n * 8);
    uint8_t* p = f.payload.data();
    put<uint32_t>(p, id.protocol);
    put<uint32_t>(p, 0);
    put<uint64_t>(p, id.model_fingerprint);
    put<int32_t>(p, id.n_layers);
    put<int32_t>(p, id.front_end);
    put<int32_t>(p, id.hc);
    put<int32_t>(p, id.n_embd);
    put<int32_t>(p, id.max_rows);
    put<int32_t>(p, 0);
    put<int64_t>(p, id.max_context);
    put<uint64_t>(p, (uint64_t) n);
    for (size_t i = 0; i < n; ++i) put<uint64_t>(p, (uint64_t) region.slot_bytes(i));
    if (!call(f, region.descriptor(), err)) {
        if (!broken_) { broken_ = true; broken_why_ = "front link: the front stage refused the hello: " + err; err = broken_why_; }
        return false;
    }
    greeted_ = true;
    return true;
}

bool RemoteFrontStage::run(int T, const int32_t* tokens, int64_t pos0, std::string& err) {
    if (!greeted_) { err = "front link: no hello yet"; return false; }
    if (T < 1 || tokens == nullptr) { err = "front link: bad window"; return false; }
    Frame f;
    f.op = Op::Run;
    f.a = T;
    f.b = pos0;
    f.payload.resize((size_t) T * sizeof(int32_t));
    std::memcpy(f.payload.data(), tokens, f.payload.size());
    return call(f, -1, err);
}

bool RemoteFrontStage::commit(int n_keep, std::string& err) {
    if (!greeted_) { err = "front link: no hello yet"; return false; }
    Frame f;
    f.op = Op::Commit;
    f.a = n_keep;
    return call(f, -1, err);
}

bool RemoteFrontStage::prefill(const int32_t* tokens, int64_t n, int64_t pos0, bool last, std::string& err) {
    if (!greeted_) { err = "front link: no hello yet"; return false; }
    if (n < 1 || tokens == nullptr || (size_t) n * sizeof(int32_t) > kLinkMaxPayload) {
        err = "front link: bad prompt chunk";
        return false;
    }
    Frame f;
    f.op = Op::Prefill;
    f.a = n;
    f.b = pos0;
    f.c = last ? 1 : 0;
    f.payload.resize((size_t) n * sizeof(int32_t));
    std::memcpy(f.payload.data(), tokens, f.payload.size());
    return call(f, -1, err);
}

bool RemoteFrontStage::session(SessionOp op, int64_t a, int64_t b, std::string& err) {
    if (!greeted_) { err = "front link: no hello yet"; return false; }
    Frame f;
    f.op = Op::Session;
    f.a = (int64_t) op;
    f.b = a;
    f.c = b;
    return call(f, -1, err);
}

bool RemoteFrontStage::shutdown(std::string& err) {
    Frame f;
    f.op = Op::Shutdown;
    const bool ok = call(f, -1, err);
    if (!broken_) { broken_ = true; broken_why_ = "front link: closed by shutdown"; }
    return ok;
}

} // namespace strata::hetero
