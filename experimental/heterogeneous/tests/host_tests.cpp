#include "strata_heterogeneous/host.hpp"

#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

using namespace strata::heterogeneous;
namespace {
int assertions = 0;
void check(bool value, const std::string& why) {
    ++assertions;
    if (!value) throw std::runtime_error(why);
}
template<class F> void throws(F&& f, const std::string& why) {
    bool caught = false;
    try { f(); } catch (const std::exception&) { caught = true; }
    check(caught, why);
}
struct Worker {
    int fd = -1;
    pid_t pid = -1;
};
std::vector<Worker> spawn(const Plan& p, const Executor& executor) {
    std::vector<Worker> workers;
    for (size_t i = 0; i < p.stages.size(); ++i) {
        int fd[2]{};
        if (::socketpair(AF_UNIX, SOCK_STREAM, 0, fd) != 0) throw std::runtime_error("socketpair failed");
        pid_t pid = ::fork();
        if (pid < 0) throw std::runtime_error("fork failed");
        if (pid == 0) {
            // Close inherited coordinator fds from previous stage forks.
            for (const auto& w : workers) ::close(w.fd);
            ::close(fd[0]);
            try {
                serve_worker(fd[1], p, i, executor);
                ::close(fd[1]);
                _exit(0);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "mock worker %zu failed unexpectedly: %s\n", i, e.what());
                _exit(2);
            }
        }
        ::close(fd[1]);
        workers.push_back({fd[0], pid});
    }
    return workers;
}
void shutdown(std::vector<Worker>& workers) {
    for (auto& w : workers) { if (w.fd >= 0) { ::close(w.fd); w.fd = -1; } }
    for (auto& w : workers) {
        int status = 0;
        if (::waitpid(w.pid, &status, 0) < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
            throw std::runtime_error("mock worker did not exit cleanly");
    }
}
Frame input(const Plan& p, uint64_t request, uint64_t window, uint32_t rows) {
    Frame f;
    f.stage_index = 0;
    f.request_id = request;
    f.window_id = window;
    f.plan_fingerprint = p.fingerprint();
    f.rows = rows;
    f.geometry = p.geometry;
    const size_t count = static_cast<size_t>(p.geometry.floats_per_row()) * rows;
    f.payload.reserve(count);
    for (size_t i = 0; i < count; ++i) f.payload.push_back(float(i % 31) / 31.f);
    return f;
}
void test_plan() {
    Plan p = parse_plan("cuda:0:0-12,hip:0:12-30,hip:1:30-48", 48, {4, 8}, 0xabc12);
    check(p.stages.size() == 3, "3 stage plan");
    check(p.geometry.floats_per_row() == 44, "R+bo+inject ABI geometry");
    check(p.fingerprint() == p.fingerprint(), "stable plan fingerprint");
    check(p.fingerprint() != parse_plan("cuda:0:0-13,hip:0:13-30,hip:1:30-48", 48, {4,8}, 0xabc12).fingerprint(),
          "partition changes plan identity");
    throws([] { (void) parse_plan("cuda:0:0-12,hip:0:13-48", 48, {4,8}, 1); }, "gap rejected");
    throws([] { (void) parse_plan("cuda:0:0-12,hip:0:11-48", 48, {4,8}, 1); }, "overlap rejected");
    throws([] { (void) parse_plan("cuda:0:0-48,hip:0:48-48", 48, {4,8}, 1); }, "empty layer rejected");
    throws([] { (void) parse_plan("cuda:0:0-20,cuda:0:20-48", 48, {4,8}, 1); }, "duplicate device rejected");
    throws([] { (void) parse_plan("cuda:bad:0-48", 48, {4,8}, 1); }, "bad ordinal rejected");
    throws([] { (void) parse_plan("vulkan:0:0-48", 48, {4,8}, 1); }, "unknown backend rejected");
    throws([] { (void) parse_plan("hip:0:0-47", 48, {4,8}, 1); }, "missing end rejected");
    throws([] { (void) parse_plan("hip:0:0-48,", 48, {4,8}, 1); }, "trailing separator rejected");
    throws([] { (void) parse_plan("hip:0:0-48", 48, {4,8}, 0); }, "missing model identity rejected");
    check(Geometry{UINT32_MAX, UINT32_MAX}.floats_per_row() == UINT64_MAX,
          "maximal geometry is representable in 64-bit floats-per-row");
    Frame huge;
    huge.request_id = 1; huge.plan_fingerprint = 1; huge.rows = 2;
    huge.geometry = {UINT32_MAX, UINT32_MAX};
    throws([&] { (void) encode(huge); }, "frame byte arithmetic overflow rejected");
}
void test_wire() {
    const Plan p = parse_plan("cuda:0:0-48", 48, {4,8}, 0x12345);
    Frame f = input(p, 42, 900, 2);
    auto bytes = encode(f);
    check(bytes.size() == wire_header_bytes + f.payload.size() * sizeof(float), "encoded size");
    auto output = decode(bytes);
    check(output.request_id == 42 && output.window_id == 900, "ids roundtrip");
    check(output.payload == f.payload, "byte-identical float roundtrip");
    // Preserve arbitrary bit patterns, including NaN payloads; float == is not enough.
    const uint32_t arbitrary_nan = 0x7fc12345u;
    std::memcpy(&f.payload[3], &arbitrary_nan, sizeof(arbitrary_nan));
    output = decode(encode(f));
    check(std::memcmp(f.payload.data(), output.payload.data(), f.payload.size() * sizeof(float)) == 0,
          "NaN payload bits survive wire roundtrip");
    bytes = encode(f); bytes.back() ^= 1;
    throws([&] { (void)decode(bytes); }, "corrupt payload rejected");
    bytes = encode(f); bytes[48] ^= 1;
    throws([&] { (void)decode(bytes); }, "bad payload length rejected");
    bytes = encode(f); bytes[4] = 99;
    throws([&] { (void)decode(bytes); }, "unsupported version rejected");
    f.payload.pop_back();
    throws([&] { (void)encode(f); }, "partial row rejected");
    f = input(p, 43, 900, 1); f.geometry.hc = 1000000;
    throws([&] { (void)encode(f); }, "oversized frame rejected");
    Frame hello;
    hello.kind = Kind::hello; hello.stage_index = 0;
    hello.request_id = 1; hello.window_id = 1; hello.rows = 1;
    hello.plan_fingerprint = p.fingerprint(); hello.geometry = p.geometry;
    hello.identity = {1, 3, 0, 48};
    check(decode(encode(hello)).identity == hello.identity, "backend/device/range hello roundtrip");
    auto bad_hello = encode(hello); bad_hello[48] = 12;
    throws([&] { (void) decode(bad_hello); }, "invalid hello length rejected");
    hello.kind = Kind::hello_shared;
    check(decode(encode(hello)).identity == hello.identity, "shared worker handshake roundtrip");
    Frame shared = input(p, 52, 901, 2);
    shared.kind = Kind::shared_data; shared.payload.clear();
    shared.shared_checksum = 0xfedcba98;
    check(decode(encode(shared)).shared_checksum == shared.shared_checksum,
          "shared handoff sends only four digest bytes");
    check(encode(shared).size() == wire_header_bytes + sizeof(uint32_t),
          "shared handoff never sends bulk payload over socket");
    shared.payload = {1.f};
    throws([&] { (void) encode(shared); }, "shared handoff refuses in-band floats");
    shared.payload.clear(); shared.error = "unexpected";
    throws([&] { (void) encode(shared); }, "shared handoff refuses accidental errors");
    shared.error.clear(); shared.geometry.hc = 1000000;
    throws([&] { (void) encode(shared); }, "shared handoff enforces capacity");
    Frame ordinary = input(p, 91, 3, 1);
    ordinary.shared_checksum = 42;
    throws([&] { (void) encode(ordinary); }, "socket mode refuses shared-only checksum");
    Frame failure = input(p, 44, 900, 1);
    failure.kind = Kind::failure; failure.payload.clear(); failure.error = "GPU stream failed";
    check(decode(encode(failure)).error == "GPU stream failed", "error frame roundtrip");
    int sockets[2]{};
    check(::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0, "socket pair");
    f = input(p, 45, 901, 1);
    send_frame(sockets[0], f);
    check(receive_frame(sockets[1])->payload == f.payload, "socket roundtrip");
    // Half of a header is not a successful EOF. A killed worker must fail closed.
    auto frame_bytes = encode(f);
    check(::write(sockets[0], frame_bytes.data(), 4) == 4, "write partial header");
    ::close(sockets[0]);
    throws([&] { (void)receive_frame(sockets[1]); }, "truncated header rejected");
    check(!receive_frame(sockets[1]), "clean EOF after rejected partial header");
    ::close(sockets[1]);
}
void test_process_pipeline() {
    const Plan p = parse_plan("cuda:0:0-3,hip:0:3-7,hip:1:7-10", 10, {2,6}, 0xfeedface);
    auto workers = spawn(p, mock_execute);
    std::vector<int> fds;
    for (auto& w : workers) fds.push_back(w.fd);
    PipelineSession session(p, fds);
    session.initialize();
    for (uint32_t rows : {1u, 8u, 13u}) {
        Frame initial = input(p, 100 + rows, 777 + rows, rows);
        Frame actual = session.run_window(initial);
        Frame expected = initial;
        mock_execute(Stage{Backend::cuda, 0, 0, p.total_layers}, initial, expected, expected.error);
        check(actual.payload == expected.payload, "split and unsplit layer order agree exactly");
        check(actual.stage_index == 2, "final stage owns output");
        check(actual.window_id == initial.window_id, "window id preserved");
    }
    shutdown(workers);
}
// Invariant: a worker's declared backend, ordinal, and layer interval must be
// checked against the complete plan before it can receive any activation.
void test_handshake_rejection() {
    Plan p = parse_plan("cuda:0:0-3,hip:0:3-10", 10, {2, 6}, 0xabcd);
    int sockets[2]{};
    check(::socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0, "identity test socketpair");
    Frame hello;
    hello.kind = Kind::hello;
    hello.request_id = 1; hello.window_id = 1; hello.rows = 1;
    hello.geometry = p.geometry; hello.plan_fingerprint = p.fingerprint();
    hello.identity = {static_cast<uint32_t>(Backend::hip), 0, 0, 3}; // expected CUDA
    send_frame(sockets[1], hello);
    PipelineSession session(p, {sockets[0], sockets[1]}); // intentionally mismatched workers
    throws([&] { session.initialize(); }, "wrong worker identity fails handshake");
    check(session.poisoned(), "handshake mismatch poisons session");
    ::close(sockets[0]); ::close(sockets[1]);

    // Separately reject a perfectly framed hello carrying an incompatible plan.
    int fd[2]{};
    check(::socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0, "plan test socketpair");
    hello.identity = {static_cast<uint32_t>(Backend::cuda), 0, 0, 3};
    hello.plan_fingerprint ^= 123;
    send_frame(fd[1], hello);
    PipelineSession bad_plan(parse_plan("cuda:0:0-10", 10, {2, 6}, 0xabcd), {fd[0]});
    throws([&] { bad_plan.initialize(); }, "wrong model/plan handshake rejected");
    check(bad_plan.poisoned(), "bad plan poisons session");
    ::close(fd[0]); ::close(fd[1]);
}
void test_io_deadlines() {
    int fd[2]{};
    check(::socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0, "timeout test socketpair");
    throws([&] { (void) receive_frame(fd[0], 20); }, "idle socket receive times out");
    const Plan p = parse_plan("cuda:0:0-10", 10, {2, 6}, 123);
    Frame f = input(p, 3, 4, 100000); // > socket buffer; receiver never drains it
    throws([&] { send_frame(fd[0], f, 20); }, "blocked socket send times out");
    throws([&] { (void) receive_frame(fd[0], -1); }, "negative timeout rejected");
    ::close(fd[0]); ::close(fd[1]);
}
void test_mock_stage_lifecycle() {
    const Plan p = parse_plan("hip:1:0-10", 10, {2, 6}, 0x123);
    MockStageExecutor stage;
    std::string error;
    Frame in = input(p, 55, 66, 1), out;
    check(stage.prepare(p, 0, error), "prepare mock stage");
    check(stage.run_window(in, out, error), "run prepared stage");
    check(!stage.run_window(in, out, error), "cannot launch second window before commit");
    check(!stage.commit_window(55, 67, error), "wrong commit window rejected");
    check(stage.commit_window(55, 66, error), "correct commit succeeds");
    check(stage.run_window(in, out, error), "can run next window after commit");
    stage.abandon_window();
    check(!stage.commit_window(55, 66, error), "abandoned window cannot be committed");
    stage.shutdown();
    check(!stage.run_window(in, out, error), "shutdown stage cannot execute");
}
struct RejectCommit final : StageExecutor {
    MockStageExecutor base;
    bool reject = false;
    bool prepare(const Plan& p, size_t i, std::string& e) override {
        reject = (i == 1); return base.prepare(p, i, e);
    }
    bool run_window(const Frame& in, Frame& out, std::string& e) override {
        return base.run_window(in, out, e);
    }
    bool commit_window(uint64_t req, uint64_t win, std::string& e) override {
        if (reject) { e = "injected HIP commit failure"; return false; }
        return base.commit_window(req, win, e);
    }
    void abandon_window() noexcept override { base.abandon_window(); }
    void shutdown() noexcept override { base.shutdown(); }
};
void test_commit_failure_and_worker_exit() {
    const Plan p = parse_plan("cuda:0:0-3,hip:0:3-10", 10, {2, 6}, 0x999);
    std::vector<Worker> workers;
    for (size_t i = 0; i < p.stages.size(); ++i) {
        int fd[2]{};
        check(::socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0, "commit-test socketpair");
        pid_t pid = ::fork();
        check(pid >= 0, "commit-test fork");
        if (!pid) {
            for (const auto& w : workers) ::close(w.fd);
            ::close(fd[0]);
            try {
                RejectCommit executor;
                serve_worker_stateful(fd[1], p, i, executor, 500);
                ::close(fd[1]); _exit(0);
            } catch (...) { _exit(2); }
        }
        ::close(fd[1]); workers.push_back({fd[0], pid});
    }
    PipelineSession session(p, {workers[0].fd, workers[1].fd}, 500);
    session.initialize();
    std::string error;
    try { (void) session.run_window(input(p, 13, 14, 2)); }
    catch (const std::exception& e) { error = e.what(); }
    check(error.find("injected HIP commit failure") != std::string::npos, "commit error propagated");
    check(session.poisoned(), "partially committed session is poisoned");
    throws([&] { (void) session.run_window(input(p, 13, 15, 2)); },
           "poisoned pipeline cannot continue inference");
    shutdown(workers);

    // A process that dies after a valid hello must never be treated as a
    // successful stage, even when its socket closes cleanly.
    int fd[2]{};
    check(::socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0, "worker-exit socketpair");
    const Plan one = parse_plan("cuda:0:0-3", 3, {2, 6}, 0x999);
    pid_t child = ::fork();
    check(child >= 0, "worker-exit fork");
    if (!child) {
        ::close(fd[0]);
        Frame hello;
        hello.kind = Kind::hello; hello.request_id = 1; hello.window_id = 1;
        hello.rows = 1; hello.geometry = p.geometry;
        hello.plan_fingerprint = one.fingerprint();
        hello.identity = {1, 0, 0, 3};
        try {
            send_frame(fd[1], hello, 500);
            (void) receive_frame(fd[1], 500); // coordinator ready
        } catch (...) { _exit(2); }
        ::close(fd[1]); _exit(0);
    }
    ::close(fd[1]);
    PipelineSession exited(one, {fd[0]}, 500);
    exited.initialize();
    throws([&] { (void) exited.run_window(input(one, 81, 82, 2)); },
           "worker exit after valid handshake rejects inference");
    check(exited.poisoned(), "worker exit poisons session");
    ::close(fd[0]);
    int status = 0; check(::waitpid(child, &status, 0) == child, "reap exited worker");
}
void test_window_replay() {
    const Plan p = parse_plan("cuda:0:0-10", 10, {2, 6}, 0xabc7);
    auto workers = spawn(p, mock_execute);
    PipelineSession session(p, {workers[0].fd}, 500);
    session.initialize();
    Frame f = input(p, 5, 100, 2);
    check(session.run_window(f).kind == Kind::data, "first window published after commit");
    check(session.run_window(input(p, 5, 101, 2)).window_id == 101, "later window commits");
    throws([&] { (void) session.run_window(f); }, "replayed request/window rejected");
    check(session.poisoned(), "replayed window poisons running session");
    shutdown(workers);
}
void test_fail_closed() {
    Plan p = parse_plan("cuda:0:0-3,hip:0:3-7,hip:1:7-10", 10, {2,6}, 0x555);
    auto failing = [](const Stage& stage, const Frame& in, Frame& out, std::string& error) -> bool {
        if (stage.first_layer == 3) { error = "simulated HIP failure"; return false; }
        return mock_execute(stage, in, out, error);
    };
    auto workers = spawn(p, failing);
    std::vector<int> fds; for (auto& w : workers) fds.push_back(w.fd);
    PipelineSession session(p, fds);
    session.initialize();
    std::string observed;
    try { (void) session.run_window(input(p, 91, 1, 2)); }
    catch (const std::exception& ex) { observed = ex.what(); }
    check(session.poisoned(), "failed window poisons coordinator session");
    check(observed.find("stage 1 failed: simulated HIP failure") != std::string::npos,
          "failed HIP stage never publishes a final result");
    shutdown(workers);
    workers = spawn(p, mock_execute);
    fds.clear(); for (auto& w : workers) fds.push_back(w.fd);
    PipelineSession fresh(p, fds);
    fresh.initialize();
    auto incorrect = input(p, 19, 2, 1);
    incorrect.plan_fingerprint ^= 123;
    throws([&] { (void) fresh.run_window(incorrect); }, "wrong model/config fingerprint rejected upfront");
    shutdown(workers);
}
}
int main() {
    try {
        test_plan(); test_wire(); test_process_pipeline(); test_fail_closed();
        test_handshake_rejection(); test_io_deadlines(); test_mock_stage_lifecycle();
        test_commit_failure_and_worker_exit(); test_window_replay();
        std::cout << "heterogeneous host tests: PASS (" << assertions << " assertions, real worker processes, no GPU)\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "heterogeneous host tests: FAIL: " << ex.what() << '\n';
        return 1;
    }
}
