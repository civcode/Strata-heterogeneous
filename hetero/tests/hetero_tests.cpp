// CPU tests of the mixed-split seam: shared hand-off memory, the front link, and the worker process.
// Usage: hetero_tests <absolute path of hetero_mock_worker>

#include "stub_backend.hpp"
#include "strata/hetero/worker_process.hpp"

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
        CHECK(!front.session((SessionOp) 7, 0, 0, err));
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

} // namespace

int main(int argc, char** argv) {
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
