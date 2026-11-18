// CPU tests of the mixed-split seam: shared hand-off memory, the front link, and the worker process.
// Usage: mixed_split_tests <absolute path of mixed_split_mock_worker>

#include "stub_backend.hpp"
#include "strata/hetero/front_runner.hpp"
#include "strata/hetero/worker_process.hpp"

#include <atomic>
#include <chrono>
#include <functional>

#include <cstdio>
#include <cstring>
#include <sys/socket.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

using namespace strata::hetero;
using strata::hetero::test::StubBackend;
using strata::hetero::test::test_identity;

static int g_failed = 0;
#define CHECK(cond) \
    do { \
        if (!(cond)) { \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failed; \
        } \
    } while (0)

namespace {

struct Pair {
    int host = -1, worker = -1;
    Pair() {
        int p[2];
        if (::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, p) == 0) { host = p[0]; worker = p[1]; }
    }
    ~Pair() {
        if (host >= 0) ::close(host);
        if (worker >= 0) ::close(worker);
    }
    void close_host() { if (host >= 0) ::close(host); host = -1; }
    void close_worker() { if (worker >= 0) ::close(worker); worker = -1; }
};

bool make_region(SharedRegion& r) {
    std::string err;
    // slot 0: hand-off rows, slot 1: the ready word (64 bytes, as vanilla allocates it)
    return SharedRegion::create({8 * sizeof(float) * 16, 64}, r, err);
}

void test_shared_region() {
    SharedRegion a;
    std::string err;
    CHECK(SharedRegion::create({100, 8192, 64}, a, err));
    CHECK(a.valid() && a.slots() == 3);
    CHECK(a.bytes() == 4096 + 8192 + 4096);
    CHECK((static_cast<uint8_t*>(a.slot(1)) - static_cast<uint8_t*>(a.slot(0))) == 4096);
    CHECK((static_cast<uint8_t*>(a.slot(2)) - static_cast<uint8_t*>(a.slot(0))) == 4096 + 8192);
    CHECK(static_cast<uint8_t*>(a.slot(0))[0] == 0);
    std::memcpy(a.slot(2), "ready", 5);

    SharedRegion b;   // a second mapping of the same memory, as the other process has
    CHECK(SharedRegion::adopt(::dup(a.descriptor()), {100, 8192, 64}, b, err));
    CHECK(std::memcmp(b.slot(2), "ready", 5) == 0);
    static_cast<uint8_t*>(b.slot(0))[7] = 42;
    CHECK(static_cast<uint8_t*>(a.slot(0))[7] == 42);

    SharedRegion c;   // a slot table that does not match the descriptor is refused
    CHECK(!SharedRegion::adopt(::dup(a.descriptor()), {100, 8192}, c, err));
    CHECK(!c.valid());
    CHECK(!SharedRegion::create({}, c, err));
    CHECK(!SharedRegion::create({0}, c, err));
    CHECK(!SharedRegion::adopt(-1, {64}, c, err));
}

void test_link_round_trip() {
    Pair s;
    StubBackend backend;
    backend.check_identity = true;
    backend.expect = test_identity();
    int rc = -1;
    std::string serr;
    std::thread worker([&] { rc = serve_front(s.worker, backend, serr); });

    SharedRegion region;
    CHECK(make_region(region));
    RemoteFrontStage front(s.host, 5000);
    std::string err;
    const int32_t toks[3] = {11, 22, 33};
    CHECK(!front.run(3, toks, 0, err));   // before the hello
    CHECK(front.hello(test_identity(), region, err));
    CHECK(!front.hello(test_identity(), region, err));
    CHECK(front.session(SessionOp::Reset, 0, 0, err));
    const int32_t prompt[5] = {1, 2, 3, 4, 5};
    CHECK(front.prefill(prompt, 5, 0, true, err));
    CHECK(front.run(3, toks, 5, err));
    // the rows and the ready word the front stage wrote are visible through the host's own mapping
    uint32_t ready = 0;
    std::memcpy(&ready, region.slot(1), sizeof ready);
    CHECK(ready == 3);
    CHECK(static_cast<float*>(region.slot(0))[2] == 33.5f);
    CHECK(front.commit(2, err));
    CHECK(front.session(SessionOp::Snapshot, 7, 7, err));
    CHECK(front.shutdown(err));
    CHECK(front.broken());
    CHECK(!front.run(3, toks, 7, err));
    worker.join();
    CHECK(rc == 0);
    const std::vector<std::string> want = {"attach", "session 1 0 0", "prefill 5 0 last 5", "run 3 5 33", "commit 2",
                                           "session 3 7 7"};
    CHECK(backend.calls == want);
}

// The two-step accept the engine's worker uses: the region is mapped before the backend exists, a refused identity
// reaches the host as a failed hello, and an abandoned window lets the next one run.
void test_two_step_accept() {
    {
        Pair s;
        StubBackend backend;
        int rc = -1;
        std::string serr;
        std::thread worker([&] {
            LinkIdentity id;
            SharedRegion region;
            uint64_t seq = 0;
            if (front_accept(s.worker, id, region, seq, serr) != Recv::Ok) return;
            const bool same = id == test_identity() && region.valid() && region.slots() == 2;
            backend.region = std::move(region);
            if (!front_accept_reply(s.worker, seq, same, "identity mismatch", serr) || !same) return;
            rc = serve_front_greeted(s.worker, backend, id, serr);
        });
        SharedRegion region;
        CHECK(make_region(region));
        RemoteFrontStage front(s.host, 5000);
        std::string err;
        CHECK(front.hello(test_identity(), region, err));
        const int32_t toks[2] = {5, 6};
        CHECK(front.run(2, toks, 0, err));
        CHECK(!front.run(2, toks, 0, err));                        // the first window was neither committed nor dropped
        CHECK(front.session(SessionOp::Abandon, 0, 0, err));
        CHECK(front.run(2, toks, 0, err));
        CHECK(front.commit(2, err));
        CHECK(!front.session((SessionOp) ((int32_t) kLastSessionOp + 1), 0, 0, err));   // past the last operation
        CHECK(front.shutdown(err));
        worker.join();
        CHECK(rc == 0);
        const std::vector<std::string> want = {"run 2 0 6", "session 6 0 0", "run 2 0 6", "commit 2"};
        CHECK(backend.calls == want);
    }
    {
        Pair s;
        std::string serr;
        bool refused = false;
        std::thread worker([&] {
            LinkIdentity id;
            SharedRegion region;
            uint64_t seq = 0;
            if (front_accept(s.worker, id, region, seq, serr) != Recv::Ok) return;
            refused = front_accept_reply(s.worker, seq, false, "another model", serr);
        });
        SharedRegion region;
        CHECK(make_region(region));
        RemoteFrontStage front(s.host, 5000);
        std::string err;
        CHECK(!front.hello(test_identity(), region, err));
        CHECK(err.find("another model") != std::string::npos);
        CHECK(front.broken());
        worker.join();
        CHECK(refused);
    }
}

void test_fingerprint() {
    const std::string dir = "/tmp/strata-hetero-fp-" + std::to_string((long) ::getpid());
    CHECK(::mkdir(dir.c_str(), 0700) == 0);
    auto write = [&](const std::string& name, const std::string& text) {
        std::FILE* f = std::fopen((dir + "/" + name).c_str(), "wb");
        CHECK(f != nullptr);
        if (f) { std::fwrite(text.data(), 1, text.size(), f); std::fclose(f); }
    };
    write("index.txt", "blk.0.a 1\n");
    write("m.gguf", "12345");
    const uint64_t a = model_fingerprint(dir, {dir + "/m.gguf"});
    CHECK(a != 0 && a == model_fingerprint(dir, {dir + "/m.gguf"}));
    write("m.gguf", "123456");
    CHECK(a != model_fingerprint(dir, {dir + "/m.gguf"}));     // another model file
    write("m.gguf", "12345");
    write("index.txt", "blk.0.a 2\n");
    CHECK(a != model_fingerprint(dir, {dir + "/m.gguf"}));     // another pack
    CHECK(model_fingerprint(dir, {}) != model_fingerprint(dir, {dir + "/missing"}));
    (void) ::unlink((dir + "/index.txt").c_str());
    (void) ::unlink((dir + "/m.gguf").c_str());
    (void) ::rmdir(dir.c_str());
}

void test_link_rules() {
    Pair s;
    StubBackend backend;
    int rc = -1;
    std::string serr;
    std::thread worker([&] { rc = serve_front(s.worker, backend, serr); });
    SharedRegion region;
    CHECK(make_region(region));
    RemoteFrontStage front(s.host, 5000);
    std::string err;
    const int32_t toks[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    CHECK(front.hello(test_identity(), region, err));
    CHECK(!front.commit(1, err));                 // no window
    CHECK(err == "no window to commit");
    CHECK(!front.broken());                       // a refused operation does not close the link
    CHECK(!front.run(9, toks, 0, err));           // more rows than the hand-off holds
    CHECK(!front.run(2, toks, 262143, err));      // past the context
    CHECK(front.run(2, toks, 0, err));
    CHECK(!front.run(2, toks, 2, err));           // the first window was not committed
    CHECK(!front.prefill(toks, 3, 2, false, err));
    CHECK(!front.commit(3, err));                 // more than the window
    CHECK(front.run(2, toks, 0, err) == false);   // the bad commit above did not clear the window
    CHECK(front.commit(2, err));
    backend.fail_next_run = true;                 // a backend failure reaches the host with its message
    CHECK(!front.run(1, toks, 2, err));
    CHECK(err == "stub run failure");
    CHECK(!front.broken());
    CHECK(front.run(1, toks, 2, err));
    CHECK(front.session(SessionOp::Rollback, 2, 0, err));   // a session operation drops the open window
    CHECK(front.run(1, toks, 2, err));
    CHECK(front.commit(1, err));
    CHECK(!front.session((SessionOp) 99, 0, 0, err));
    s.close_host();                               // the host goes away: the worker's loop ends cleanly
    worker.join();
    CHECK(rc == 0);
}

void test_identity_refused() {
    Pair s;
    StubBackend backend;
    backend.check_identity = true;
    backend.expect = test_identity();
    backend.expect.front_end = 16;   // the worker loaded another split
    int rc = -1;
    std::string serr;
    std::thread worker([&] { rc = serve_front(s.worker, backend, serr); });
    SharedRegion region;
    CHECK(make_region(region));
    RemoteFrontStage front(s.host, 5000);
    std::string err;
    CHECK(!front.hello(test_identity(), region, err));
    CHECK(front.broken());
    CHECK(err.find("identity mismatch") != std::string::npos);
    const int32_t t = 1;
    CHECK(!front.run(1, &t, 0, err));
    worker.join();
    CHECK(rc == 1);

    LinkIdentity bad = test_identity();   // refused on the host before anything is sent
    bad.front_end = 48;
    Pair s2;
    RemoteFrontStage front2(s2.host, 100);
    CHECK(!front2.hello(bad, region, err));
    CHECK(!front2.broken());
}

void test_protocol_errors() {
    {   // an operation before the hello ends the worker's loop with an error
        Pair s;
        StubBackend backend;
        int rc = -1;
        std::string serr, err;
        std::thread worker([&] { rc = serve_front(s.worker, backend, serr); });
        Frame f;
        f.op = Op::Commit;
        f.seq = 1;
        f.a = 1;
        CHECK(send_frame(s.host, f, -1, 1000, err));
        Frame rep;
        CHECK(recv_frame(s.host, rep, nullptr, 1000, err) == Recv::Ok);
        CHECK(rep.op == Op::Reply && rep.seq == 1 && rep.a == 1);
        worker.join();
        CHECK(rc == 1);
    }
    {   // bytes that are not a frame
        Pair s;
        StubBackend backend;
        int rc = -1;
        std::string serr;
        std::thread worker([&] { rc = serve_front(s.worker, backend, serr); });
        const char junk[48] = "this is not a front link frame at all, really.";
        CHECK(::send(s.host, junk, sizeof junk, MSG_NOSIGNAL) == (ssize_t) sizeof junk);
        worker.join();
        CHECK(rc == 1);
    }
    {   // the socket closes inside a frame
        Pair s;
        StubBackend backend;
        int rc = -1;
        std::string serr;
        std::thread worker([&] { rc = serve_front(s.worker, backend, serr); });
        const char part[10] = "SFL1";
        CHECK(::send(s.host, part, sizeof part, MSG_NOSIGNAL) == (ssize_t) sizeof part);
        s.close_host();
        worker.join();
        CHECK(rc == 1);
    }
}

void test_worker_lost_and_timeout() {
    {   // the worker goes away between two calls
        Pair s;
        StubBackend backend;
        SharedRegion region;
        CHECK(make_region(region));
        std::string serr, err;
        std::thread worker([&] {
            Frame f;
            int got = -1;
            (void) recv_frame(s.worker, f, &got, 2000, serr);   // takes the hello, answers, and dies
            if (got >= 0) ::close(got);
            Frame r;
            r.op = Op::Reply;
            r.seq = f.seq;
            (void) send_frame(s.worker, r, -1, 2000, serr);
            s.close_worker();
        });
        RemoteFrontStage front(s.host, 2000);
        CHECK(front.hello(test_identity(), region, err));
        worker.join();
        const int32_t t = 1;
        CHECK(!front.run(1, &t, 0, err));
        CHECK(front.broken());
        CHECK(!front.commit(1, err));
    }
    {   // a reply that does not come in time closes the link
        Pair s;
        StubBackend backend;
        backend.commit_sleep_ms = 400;
        int rc = -1;
        std::string serr, err;
        std::thread worker([&] { rc = serve_front(s.worker, backend, serr); });
        SharedRegion region;
        CHECK(make_region(region));
        RemoteFrontStage front(s.host, 60);
        CHECK(front.hello(test_identity(), region, err));
        const int32_t t = 1;
        CHECK(front.run(1, &t, 0, err));
        CHECK(!front.commit(1, err));
        CHECK(err.find("timed out") != std::string::npos);
        CHECK(front.broken());
        s.close_host();
        worker.join();
    }
}

void test_worker_process(const std::string& mock) {
    std::string err;
    {   // a served request and a clean end
        WorkerProcess w;
        CHECK(WorkerProcess::launch(mock, {"serve"}, 5000, w, err));
        CHECK(w.started() && !w.ended());
        SharedRegion region;
        CHECK(make_region(region));
        RemoteFrontStage front(w.socket(), 5000);
        CHECK(front.hello(test_identity(), region, err));
        const int32_t toks[2] = {7, 9};
        CHECK(front.run(2, toks, 0, err));
        CHECK(static_cast<float*>(region.slot(0))[1] == 9.5f);   // written by the other process
        CHECK(front.commit(2, err));
        CHECK(w.finish(err));
        int code = -1;
        CHECK(w.ended(&code) && code == 0);
    }
    {   // the worker refuses another model
        WorkerProcess w;
        CHECK(WorkerProcess::launch(mock, {"serve"}, 5000, w, err));
        SharedRegion region;
        CHECK(make_region(region));
        RemoteFrontStage front(w.socket(), 5000);
        LinkIdentity other = test_identity();
        other.model_fingerprint ^= 1;
        CHECK(!front.hello(other, region, err));
        CHECK(!w.finish(err));
        CHECK(err.find("exit code 1") != std::string::npos);
    }
    {   // a worker that ends with a failure
        WorkerProcess w;
        CHECK(WorkerProcess::launch(mock, {"exit3"}, 5000, w, err));
        CHECK(!w.finish(err));
        CHECK(err.find("exit code 3") != std::string::npos);
    }
    {   // a worker that drains slowly is reported and left alone, and can be waited for again
        WorkerProcess w;
        CHECK(WorkerProcess::launch(mock, {"slow-exit"}, 50, w, err));
        const pid_t pid = w.pid();
        CHECK(!w.finish(err));
        CHECK(err.find("not signalled") != std::string::npos);
        CHECK(err.find(std::to_string((long) pid)) != std::string::npos);
        bool done = false;
        for (int i = 0; i < 100 && !done; ++i) done = w.finish(err);
        CHECK(done);
    }
    {
        WorkerProcess w;
        CHECK(!WorkerProcess::launch("relative/path", {}, 100, w, err));
        CHECK(!WorkerProcess::launch("/nonexistent/strata-hetero-worker", {}, 100, w, err));
        CHECK(!w.started());
    }
}

// token_key: the host's int64 prompt tokens and int32 checkpoint ids give the same key; length and order count.
void test_token_key() {
    const int64_t a64[4] = {248045, 846, 198, 7};
    const int32_t a32[4] = {248045, 846, 198, 7};
    const int32_t b32[4] = {248045, 846, 7, 198};
    CHECK(token_key(a64, 4) == token_key(a32, 4));
    CHECK(token_key(a32, 4) != token_key(b32, 4));
    CHECK(token_key(a32, 3) != token_key(a32, 4));
    CHECK(token_key(a32, 0) > 0);
}

// The session operations the host sends for vanilla's checkpoints and conversation cache (Snapshot, Restore, Park,
// Unpark), through the link to the front stage's bookkeeping: the keys travel, a state is restored only for the tokens
// it was saved at, and anything out of step is refused with a message.
void test_session_ops() {
    Pair s;
    test::BookBackend backend;
    int rc = -1;
    std::string serr;
    std::thread worker([&] { rc = serve_front(s.worker, backend, serr); });
    SharedRegion region;
    CHECK(make_region(region));
    RemoteFrontStage front(s.host, 5000);
    std::string err;
    CHECK(front.hello(test_identity(), region, err));
    CHECK(front.session(SessionOp::Reset, 0, 0, err));
    // turn 1: a prompt of 5, a window of 3 that keeps 2 -> the session holds 7 tokens
    std::vector<int32_t> conv = {1, 2, 3, 4, 5};
    CHECK(front.prefill(conv.data(), 5, 0, true, err));
    CHECK(front.session(SessionOp::Snapshot, token_key(conv.data(), 5), 5, err));          // Snapshot (mid-prompt)
    const int32_t win[3] = {6, 7, 99};
    CHECK(!front.run(3, win, 4, err));                                                     // not where the session is
    CHECK(err.find("out of step") != std::string::npos);
    CHECK(front.run(3, win, 5, err));
    CHECK(front.commit(2, err));
    conv.push_back(6);
    conv.push_back(7);
    CHECK(!front.session(SessionOp::Snapshot, token_key(conv.data(), 6), 6, err));         // wrong length
    std::vector<int32_t> other = conv;
    other[2] = 33;
    CHECK(!front.session(SessionOp::Snapshot, token_key(other.data(), 7), 7, err));        // other tokens
    CHECK(err.find("other tokens") != std::string::npos);
    CHECK(front.session(SessionOp::Snapshot, token_key(conv.data(), 7), 7, err));          // Snapshot (turn)
    CHECK(backend.book.parts() == 2);
    // Park the conversation (with its two checkpoints), start another one, Unpark the first: it is back at 7 tokens
    CHECK(!front.session(SessionOp::Park, token_key(conv.data(), 5), 5, err));             // not what it holds
    CHECK(front.session(SessionOp::Park, token_key(conv.data(), 7), 7, err));
    CHECK(backend.book.images() == 1);
    CHECK(front.session(SessionOp::Reset, 0, 0, err));
    const int32_t p2[4] = {40, 41, 42, 43};
    CHECK(front.prefill(p2, 4, 0, true, err));
    CHECK(front.session(SessionOp::Snapshot, token_key(p2, 4), 4, err));
    CHECK(front.session(SessionOp::Snapshot, token_key(p2, 2), 2, err) == false);          // (the session holds 4)
    CHECK(!front.session(SessionOp::Unpark, token_key(conv.data(), 6), 6, err));           // no such copy
    CHECK(front.session(SessionOp::Unpark, token_key(conv.data(), 7), 7, err));            // Unpark
    CHECK(backend.book.history() == conv);
    CHECK(backend.state == test::BookBackend::text(conv));
    CHECK(backend.book.images() == 0);                                                     // taken, as the host takes it
    CHECK(!front.session(SessionOp::Unpark, token_key(conv.data(), 7), 7, err));
    // Restore the mid-prompt checkpoint (it came back with the image even after the parts store moved on)
    CHECK(front.session(SessionOp::Restore, token_key(conv.data(), 5), 5, err));           // Restore
    CHECK(backend.book.history() == std::vector<int32_t>(conv.begin(), conv.begin() + 5));
    CHECK(backend.state == "1,2,3,4,5,");
    const int32_t nxt[1] = {8};
    CHECK(front.prefill(nxt, 1, 5, true, err));                                            // continues from 5
    CHECK(!front.session(SessionOp::Restore, token_key(conv.data(), 6), 6, err));          // never saved
    // the parts store keeps the 3 most recently used
    for (int k = 1; k <= 4; ++k) {
        CHECK(front.session(SessionOp::Reset, 0, 0, err));
        const int32_t t[1] = {100 + k};
        CHECK(front.prefill(t, 1, 0, true, err));
        CHECK(front.session(SessionOp::Snapshot, token_key(t, 1), 1, err));
    }
    CHECK(backend.book.parts() == 3);
    CHECK(!front.session(SessionOp::Restore, token_key(conv.data(), 5), 5, err));          // evicted: refused
    CHECK(!front.session(SessionOp::Rollback, 0, 0, err));                                 // no call site: refused
    CHECK(!front.session(SessionOp::Truncate, 0, 0, err));
    CHECK(!front.broken());
    CHECK(front.shutdown(err));
    worker.join();
    CHECK(rc == 0);
}

// Drop: the host's conversation cache let a parked copy go (evicted, superseded, cleared or refused).  The front stage
// drops its copy, and an open window survives it: the cache can drop entries while the host runs a window.
void test_session_drop() {
    Pair s;
    test::BookBackend backend;
    int rc = -1;
    std::string serr;
    std::thread worker([&] { rc = serve_front(s.worker, backend, serr); });
    SharedRegion region;
    CHECK(make_region(region));
    RemoteFrontStage front(s.host, 5000);
    std::string err;
    CHECK(front.hello(test_identity(), region, err));
    CHECK(front.session(SessionOp::Reset, 0, 0, err));
    std::vector<int32_t> a = {1, 2, 3};
    CHECK(front.prefill(a.data(), 3, 0, true, err));
    CHECK(front.session(SessionOp::Park, token_key(a.data(), 3), 3, err));
    CHECK(backend.book.images() == 1);
    const int32_t win[2] = {4, 5};
    CHECK(front.run(2, win, 3, err));                                                      // a window is open ...
    CHECK(front.session(SessionOp::Drop, token_key(a.data(), 3), 3, err));                // ... when the host drops A
    CHECK(backend.book.images() == 0);
    CHECK(front.commit(2, err));                                                           // the window still commits
    CHECK(backend.book.history() == std::vector<int32_t>({1, 2, 3, 4, 5}));
    CHECK(front.session(SessionOp::Drop, token_key(a.data(), 3), 3, err));                // no copy: not an error
    CHECK(!front.session(SessionOp::Unpark, token_key(a.data(), 3), 3, err));             // gone
    CHECK(!front.broken());
    CHECK(front.shutdown(err));
    worker.join();
    CHECK(rc == 0);
}

// Steer: the control-vector switch.  A state saved with the vector off has another key than the same tokens with it
// on, so a parked copy or checkpoint is never restored under the wrong switch.
void test_session_steer() {
    CHECK(steer_key(12345, true) == 12345);
    CHECK(steer_key(12345, false) != 12345 && steer_key(12345, false) > 0);
    Pair s;
    test::BookBackend backend;
    int rc = -1;
    std::string serr;
    std::thread worker([&] { rc = serve_front(s.worker, backend, serr); });
    SharedRegion region;
    CHECK(make_region(region));
    RemoteFrontStage front(s.host, 5000);
    std::string err;
    CHECK(front.hello(test_identity(), region, err));
    CHECK(front.session(SessionOp::Reset, 0, 0, err));
    std::vector<int32_t> a = {1, 2, 3};
    CHECK(front.session(SessionOp::Steer, 0, 0, err));                                     // a request with cvec=0
    CHECK(!backend.book.steer());
    CHECK(front.prefill(a.data(), 3, 0, true, err));
    const int64_t on = token_key(a.data(), 3), off = steer_key(on, false);
    CHECK(!front.session(SessionOp::Snapshot, on, 3, err));                                // the host's key says "on"
    CHECK(front.session(SessionOp::Snapshot, off, 3, err));
    CHECK(front.session(SessionOp::Park, off, 3, err));
    CHECK(front.session(SessionOp::Reset, 0, 0, err));
    CHECK(front.session(SessionOp::Steer, 1, 0, err));                                     // next request: cvec=1
    CHECK(!front.session(SessionOp::Unpark, on, 3, err));                                  // no copy computed with it on
    CHECK(!front.session(SessionOp::Restore, on, 3, err));
    CHECK(front.session(SessionOp::Unpark, off, 3, err));                                  // the off copy, asked for as such
    CHECK(!backend.book.steer());                                                          // the state's own switch
    CHECK(backend.book.history() == a);
    CHECK(front.session(SessionOp::Restore, off, 3, err));                                 // its checkpoint came back with it
    CHECK(!front.broken());
    CHECK(front.shutdown(err));
    worker.join();
    CHECK(rc == 0);
}


// image_key / state_key: the pictures below a prefix's length are part of its key; text keys are unchanged.
void test_image_key() {
    const int32_t ids[6] = {1, 248056, 248056, 248056, 2, 3};
    const std::vector<ImageKey> none, a = {{1, 0xaaaa}}, b = {{1, 0xbbbb}}, late = {{5, 0xaaaa}};
    const int64_t t = token_key(ids, 6);
    CHECK(image_key(t, none, 6) == t);
    CHECK(state_key(ids, 6, true, none) == t);
    CHECK(state_key(ids, 6, false, none) == steer_key(t, false));
    CHECK(state_key(ids, 6, true, a) != t);
    CHECK(state_key(ids, 6, true, a) != state_key(ids, 6, true, b));               // same tokens, other picture
    CHECK(state_key(ids, 6, true, a) != state_key(ids, 6, false, a));
    CHECK(state_key(ids, 5, true, late) == token_key(ids, 5));                       // a picture at or past n is not below it
    CHECK(state_key(ids, 1, true, a) == token_key(ids, 1));
    const std::vector<strata::hetero::ImageSpan> spans = {{1, 0xaaaa, 3}};
    CHECK(state_key(ids, 6, true, spans) == state_key(ids, 6, true, a));             // any type with start and hash
    CHECK(images_below(a, 1).empty() && images_below(a, 2) == a);
}

FrontImages picture(int64_t n, int64_t start, int64_t rows, uint64_t hash, int n_embd, float mark) {
    FrontImages im;
    im.n = n;
    im.images.push_back({start, hash, rows});
    im.mrope.assign(3 * 64, 0);
    for (size_t i = 0; i < im.mrope.size(); ++i) im.mrope[i] = (int32_t) (i / 3);
    im.rows.resize((size_t) rows * (size_t) n_embd);
    for (size_t i = 0; i < im.rows.size(); ++i) im.rows[i] = mark + (float) (i % 977);
    return im;
}

// Images over the link: the whole input arrives (in pieces past kLinkMaxPayload), a bad layout is refused without
// breaking the link, and a text request (no images, no table) passes.
void test_images_link() {
    Pair s;
    StubBackend backend;
    int rc = -1;
    std::string serr;
    std::thread worker([&] { rc = serve_front(s.worker, backend, serr); });
    SharedRegion region;
    CHECK(make_region(region));
    RemoteFrontStage front(s.host, 10000);
    std::string err;
    const LinkIdentity id = test_identity();
    CHECK(front.hello(id, region, err));
    // 2,000 rows of 2,560 floats: about 20 MB, two pieces
    FrontImages big = picture(3000, 10, 2000, 0x1234, id.n_embd, 0.25f);
    const FrontImages sent = big;
    CHECK(encode_images(sent).size() > kLinkMaxPayload);
    CHECK(front.images(big, err));
    CHECK(backend.last_images.n == 3000 && backend.last_images.images.size() == 1);
    CHECK(backend.last_images.images[0].start == 10 && backend.last_images.images[0].rows == 2000 &&
          backend.last_images.images[0].hash == 0x1234);
    CHECK(backend.last_images.mrope == sent.mrope);
    CHECK(backend.last_images.rows == sent.rows);
    // two pictures
    FrontImages two = picture(100, 2, 4, 7, id.n_embd, 1.0f);
    two.images.push_back({20, 8, 3});
    two.rows.resize((size_t) 7 * (size_t) id.n_embd, 2.0f);
    CHECK(front.images(two, err));
    CHECK(backend.last_images.images.size() == 2 && backend.last_images.rows.size() == (size_t) 7 * id.n_embd);
    // refused layouts (the link stays up)
    FrontImages past = picture(12, 10, 4, 1, id.n_embd, 0.0f);   // rows past the prompt
    CHECK(!front.images(past, err));
    CHECK(err.find("outside the prompt") != std::string::npos);
    FrontImages overlap = picture(100, 2, 4, 7, id.n_embd, 0.0f);
    overlap.images.push_back({4, 8, 3});
    overlap.rows.resize((size_t) 7 * (size_t) id.n_embd);
    CHECK(!front.images(overlap, err));
    FrontImages short_rows = picture(100, 2, 4, 7, id.n_embd, 0.0f);
    short_rows.rows.pop_back();
    CHECK(!front.images(short_rows, err));
    FrontImages table_only;
    table_only.n = 5;
    table_only.mrope.assign(30, 0);
    CHECK(!front.images(table_only, err));
    CHECK(!front.broken());
    FrontImages text;   // a text request
    text.n = 5;
    CHECK(front.images(text, err));
    CHECK(backend.last_images.images.empty() && backend.last_images.mrope.empty());
    // a window open: the host drops it first (HeteroHost); the link refuses images over it
    const int32_t win[2] = {1, 2};
    CHECK(front.run(2, win, 0, err));
    CHECK(!front.images(text, err));
    CHECK(front.commit(2, err));
    CHECK(front.shutdown(err));
    worker.join();
    CHECK(rc == 0);
}

// Pictures in the front stage's keys: two requests with the same tokens and different pictures never share a state,
// and a session holding one picture is not continued by a request with another.
void test_session_images() {
    Pair s;
    test::BookBackend backend;
    int rc = -1;
    std::string serr;
    std::thread worker([&] { rc = serve_front(s.worker, backend, serr); });
    SharedRegion region;
    CHECK(make_region(region));
    RemoteFrontStage front(s.host, 5000);
    std::string err;
    const LinkIdentity id = test_identity();
    CHECK(front.hello(id, region, err));
    const std::vector<int32_t> p = {1, 248056, 248056, 248056, 2, 3, 4};   // a picture of three rows at 1
    FrontImages a = picture(8, 1, 3, 0xaaaa, id.n_embd, 0.0f), b = picture(8, 1, 3, 0xbbbb, id.n_embd, 0.0f);
    const std::vector<ImageKey> ka = {{1, 0xaaaa}}, kb = {{1, 0xbbbb}};
    // request A: read, checkpoint, park
    CHECK(front.images(a, err));
    CHECK(front.session(SessionOp::Reset, 0, 0, err));
    CHECK(front.prefill(p.data(), 7, 0, true, err));
    CHECK(backend.book.images_held() == ka);
    CHECK(!front.session(SessionOp::Snapshot, token_key(p.data(), 7), 7, err));                 // the key without the picture
    CHECK(!front.session(SessionOp::Snapshot, state_key(p.data(), 7, true, kb), 7, err));      // another picture
    CHECK(front.session(SessionOp::Snapshot, state_key(p.data(), 7, true, ka), 7, err));
    CHECK(front.session(SessionOp::Park, state_key(p.data(), 7, true, ka), 7, err));
    // request B, the same tokens: A's state is neither restored nor continued
    CHECK(front.images(b, err));
    const int32_t more[1] = {5};
    CHECK(!front.prefill(more, 1, 7, true, err));                                                // the session holds A's picture
    CHECK(err.find("other pictures") != std::string::npos);
    CHECK(!front.session(SessionOp::Unpark, state_key(p.data(), 7, true, kb), 7, err));
    CHECK(!front.session(SessionOp::Restore, state_key(p.data(), 7, true, kb), 7, err));
    CHECK(front.session(SessionOp::Reset, 0, 0, err));
    CHECK(front.prefill(p.data(), 7, 0, true, err));
    CHECK(backend.book.images_held() == kb);
    CHECK(front.session(SessionOp::Snapshot, state_key(p.data(), 7, true, kb), 7, err));
    // request A again (a second turn): its parked copy comes back with its picture and its checkpoint
    CHECK(front.images(a, err));
    CHECK(front.session(SessionOp::Unpark, state_key(p.data(), 7, true, ka), 7, err));
    CHECK(backend.book.images_held() == ka);
    CHECK(front.prefill(more, 1, 7, true, err));                                                 // continues A
    CHECK(backend.book.images_held() == ka);
    CHECK(front.session(SessionOp::Restore, state_key(p.data(), 7, true, ka), 7, err));
    CHECK(backend.book.images_held() == ka);
    // a text request after it: the session is reset, the keys are the plain token keys again
    FrontImages text;
    text.n = 3;
    CHECK(front.images(text, err));
    CHECK(!front.prefill(more, 1, 7, true, err));                                                // A's picture is below 7
    CHECK(front.session(SessionOp::Reset, 0, 0, err));
    const int32_t t3[3] = {7, 8, 9};
    CHECK(front.prefill(t3, 3, 0, true, err));
    CHECK(backend.book.images_held().empty());
    CHECK(front.session(SessionOp::Snapshot, token_key(t3, 3), 3, err));
    CHECK(!front.broken());
    CHECK(front.shutdown(err));
    worker.join();
    CHECK(rc == 0);
}

// Pipelined windows over the link: a window per parity with its own hand-off and ready word, the asynchronous commit,
// the undo and the replayed commit, and the rules the worker enforces.
void test_pipelined_link() {
    Pair s;
    test::BookBackend backend;
    int rc = -1;
    std::string serr;
    std::thread worker([&] { rc = serve_front(s.worker, backend, serr); });
    SharedRegion region;   // the engine's layout: hand-off 0, ready word 0, prompt rows, hand-off 1, ready word 1
    std::string err;
    CHECK(SharedRegion::create({8 * sizeof(float) * 16, 64, 4096, 8 * sizeof(float) * 16, 64}, region, err));
    RemoteFrontStage front(s.host, 5000);
    CHECK(front.hello(test_identity(), region, err));
    const int32_t prompt[4] = {1, 2, 3, 4};
    CHECK(front.prefill(prompt, 4, 0, true, err));
    const int32_t a[3] = {5, 6, 7}, b[2] = {8, 9}, a2[2] = {6, 70};
    CHECK(!front.pl_commit(0, 1, err));                     // nothing launched on parity 0
    CHECK(err == "no pipelined window of that parity to commit");
    CHECK(!front.pl_undo(1, err));
    CHECK(!front.pl_run(2, 3, a, 4, err));                  // no such parity (refused on the host)
    CHECK(!front.pl_run(0, 3, a, 3, err));                  // not where the session is
    CHECK(err.find("out of step") != std::string::npos);
    CHECK(front.pl_run(0, 3, a, 4, err));                   // A (parity 0)
    uint32_t ready = 0;
    std::memcpy(&ready, region.slot(1), sizeof ready);
    CHECK(ready == 103 && static_cast<float*>(region.slot(0))[2] == 7.25f);
    CHECK(!front.pl_commit(0, 4, err));                     // more than the window
    CHECK(front.pl_commit(0, 3, err));                      // A committed whole, speculatively
    CHECK(front.pl_run(1, 2, b, 7, err));                   // B (parity 1) behind it, through hand-off 1
    std::memcpy(&ready, region.slot(4), sizeof ready);
    CHECK(ready == 102 && static_cast<float*>(region.slot(3))[1] == 9.25f);
    CHECK(backend.book.history() == std::vector<int32_t>({1, 2, 3, 4, 5, 6, 7}));
    // A's verdict kept 1 draft (a = 1): B was a wrong guess.  Undo A's commit, replay it with its real count (2)
    CHECK(front.pl_undo(0, err));
    CHECK(backend.book.history() == std::vector<int32_t>({1, 2, 3, 4}));
    CHECK(front.pl_commit(0, 2, err));
    CHECK(backend.book.history() == std::vector<int32_t>({1, 2, 3, 4, 5, 6}));
    CHECK(!front.pl_commit(0, 2, err));                     // a second replay does not continue the session
    CHECK(!front.pl_run(1, 2, a2, 7, err));                 // the next window starts where the real commit ended
    CHECK(front.pl_run(0, 2, a2, 6, err));
    CHECK(front.pl_commit(0, 2, err));
    CHECK(backend.book.history() == std::vector<int32_t>({1, 2, 3, 4, 5, 6, 6, 70}));
    // a serial window or a session operation ends the pipelined windows: nothing is left to commit or undo
    CHECK(front.session(SessionOp::Abandon, 0, 0, err));
    CHECK(!front.pl_commit(0, 1, err));
    CHECK(!front.pl_undo(0, err));
    const int32_t w[1] = {11};
    CHECK(front.run(1, w, 8, err));
    CHECK(!front.pl_run(0, 1, w, 8, err));                  // the serial window waits for its commit
    CHECK(front.commit(1, err));
    backend.fail_next_run = true;                           // a failed launch leaves no window to commit
    CHECK(!front.pl_run(1, 1, w, 9, err));
    CHECK(err == "stub run failure");
    CHECK(!front.pl_commit(1, 1, err));
    CHECK(!front.broken());
    CHECK(front.shutdown(err));
    worker.join();
    CHECK(rc == 0);
    const std::vector<std::string> want = {"attach", "prefill 4 0 last 4", "pl_run 0 3 4 7", "pl_commit 0 3", "pl_run 1 2 7 9",
                                           "pl_undo 0", "pl_commit 0 2", "pl_run 0 2 6 70", "pl_commit 0 2",
                                           "session 6 0", "run 1 8 11", "commit 1"};
    CHECK(backend.calls == want);
}

// The front stage's window in flight: the call returns at the launch, the next call waits for the window's end and
// gets an error it ended with.
void test_window_runner() {
    WindowRunner r;
    std::string err;
    CHECK(r.settle(err));                                   // nothing ran yet
    std::atomic<int> stage{0};
    std::atomic<bool> go{false};
    CHECK(r.start([&](const std::function<void()>& launched, std::string&) {
        stage = 1;
        launched();
        launched();                                         // (a second call does nothing)
        while (!go.load()) std::this_thread::yield();
        stage = 2;
        return true;
    }, err));
    CHECK(stage.load() == 1);                               // returned at the launch, the window still runs
    CHECK(r.busy());
    go = true;
    CHECK(r.settle(err));
    CHECK(stage.load() == 2 && !r.busy());
    // a failure before the launch is the start's
    err.clear();
    CHECK(!r.start([](const std::function<void()>&, std::string& e) { e = "no launch"; return false; }, err));
    CHECK(err == "no launch");
    CHECK(r.settle(err));                                   // reported once
    // a failure after the launch is the next call's
    err.clear();
    CHECK(r.start([](const std::function<void()>& launched, std::string& e) {
        launched();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        e = "layer 3 never rang";
        return false;
    }, err));
    CHECK(!r.settle(err));
    CHECK(err == "layer 3 never rang");
    CHECK(r.settle(err));
    // a job that ends without calling launched() and without an error
    CHECK(r.start([](const std::function<void()>&, std::string&) { return true; }, err));
    // start() settles the job before: its error reaches the start
    err.clear();
    CHECK(r.start([](const std::function<void()>& launched, std::string& e) {
        launched();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        e = "late";
        return false;
    }, err));
    bool ran = false;
    CHECK(!r.start([&](const std::function<void()>& launched, std::string&) { ran = true; launched(); return true; }, err));
    CHECK(err == "late" && !ran);
}

} // namespace

int main(int argc, char** argv) {
    test_token_key();
    test_pipelined_link();
    test_window_runner();
    test_session_ops();
    test_session_drop();
    test_session_steer();
    test_image_key();
    test_images_link();
    test_session_images();
    test_shared_region();
    test_link_round_trip();
    test_link_rules();
    test_two_step_accept();
    test_fingerprint();
    test_identity_refused();
    test_protocol_errors();
    test_worker_lost_and_timeout();
    if (argc > 1) test_worker_process(argv[1]);
    else std::fprintf(stderr, "note: no mock worker path given; the process tests were skipped\n");
    if (g_failed) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failed);
        return 1;
    }
    std::printf("hetero tests passed\n");
    return 0;
}
