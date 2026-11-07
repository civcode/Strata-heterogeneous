// M1.9: verifier request metadata and deferred accepted-prefix commit.
// Real OS worker processes; no GPU runtime, model weights or token sampling.
#include "strata_heterogeneous/host.hpp"
#include "strata_heterogeneous/process.hpp"
#include "strata_heterogeneous/device.hpp"
#include <climits>
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
void check(bool ok, const char* why) {
    ++assertions;
    if (!ok) throw std::runtime_error(why);
}
template<class F> void fails(F&& f, const char* why) {
    bool threw = false;
    try { f(); } catch (const std::exception&) { threw = true; }
    check(threw, why);
}
Frame make_frame(const Plan& p, uint64_t req = 11, uint64_t window = 1, uint32_t rows = 3) {
    Frame f;
    f.request_id = req; f.window_id = window; f.rows = rows;
    f.geometry = p.geometry; f.plan_fingerprint = p.fingerprint();
    f.payload.assign(static_cast<size_t>(rows) * p.geometry.floats_per_row(), 0.125f);
    return f;
}
WindowSpec make_spec(int64_t pos0 = 37) { return {pos0, {4, 51, 9}}; }

void wire_tests() {
    Plan p = parse_plan("cuda:0:0-5", 5, {2,3}, 0xabcd);
    Frame f = make_frame(p);
    f.kind = Kind::window_spec; f.payload.clear(); f.window = make_spec();
    auto bytes = encode(f);
    check(bytes.size() == wire_header_bytes + 24, "context has fixed 12-byte prefix plus token IDs");
    Frame restored = decode(bytes);
    check(restored.kind == Kind::window_spec && restored.window.has_value(), "context decodes");
    check(restored.window->pos0 == 37 && restored.window->token_ids == make_spec().token_ids,
          "position and all IDs are exact");
    bytes.back() ^= 1;
    fails([&] { (void) decode(bytes); }, "wire checksum detects changed token");
    bytes = encode(f); bytes[4] = 3;
    fails([&] { (void) decode(bytes); }, "old v3 workers are rejected");
    f.window = WindowSpec{INT64_MAX, {1,2,3}};
    fails([&] { (void) encode(f); }, "position overflow refused");
    f.window = WindowSpec{-1, {1,2,3}};
    fails([&] { (void) encode(f); }, "negative position refused");
    f.window = WindowSpec{0, {1,-1,3}};
    fails([&] { (void) encode(f); }, "negative token refused");
    f.window = WindowSpec{0, {1,2}};
    fails([&] { (void) encode(f); }, "wrong number of tokens refused");
    f.window = WindowSpec{0, {1,2,3}}; f.rows = max_window_tokens + 1;
    fails([&] { (void) encode(f); }, "protocol row cap enforced");
    Frame prefix = make_frame(p);
    prefix.kind = Kind::commit_prefix; prefix.payload.clear();
    check(decode(encode(prefix)).keep_count == 0, "zero accepted prefix allowed");
    prefix.keep_count = 2;
    check(decode(encode(prefix)).keep_count == 2, "partial prefix roundtrip");
    prefix.kind = Kind::committed_prefix;
    check(decode(encode(prefix)).keep_count == 2, "commit ACK carries count");
    prefix.keep_count = 4;
    fails([&] { (void) encode(prefix); }, "oversized accepted prefix refused");
    prefix = make_frame(p); prefix.keep_count = 1;
    fails([&] { (void) encode(prefix); }, "data cannot carry hidden commit count");
    prefix.keep_count = 0; prefix.window = make_spec();
    fails([&] { (void) encode(prefix); }, "data cannot hide context");
}

void run_three_workers(bool shared) {
    Plan p = parse_plan("cuda:0:0-3,hip:0:3-7,hip:1:7-10", 10, {2,3}, 0xaaa);
    const std::string dir = executable_directory();
    auto workers = WorkerProcessGroup::launch(p,
        dir + "/strata_heterogeneous_mock_cuda_worker",
        dir + "/strata_heterogeneous_mock_hip_worker", shared);
    PipelineSession pipeline(p, workers.sockets(), 1000, shared);
    pipeline.initialize();
    auto input = make_frame(p);
    Frame provisional = pipeline.execute_window(input, make_spec());
    Frame reference;
    std::string error;
    check(mock_execute(Stage{Backend::cuda,0,0,10}, input, reference, error), "reference computed");
    check(provisional.payload == reference.payload, "all stages execute ordered layers");
    check(pipeline.has_pending(), "output provisional until accepted-prefix decision");
    fails([&] { (void) pipeline.run_window(input); }, "cannot overlap uncommitted windows");
    check(!pipeline.poisoned(), "overlap API mistake leaves pending work intact");
    fails([&] { (void) pipeline.commit_pending(4); }, "commit count must not exceed row count");
    check(pipeline.has_pending() && !pipeline.poisoned(), "bad count can be corrected");
    check(pipeline.commit_pending(2).payload == reference.payload, "partial accepted prefix commits");
    check(!pipeline.has_pending(), "stage acknowledgments drain pending window");
    fails([&] { (void) pipeline.commit_pending(2); }, "duplicate commit refused");
    check(pipeline.execute_window(make_frame(p,11,2), WindowSpec{40,{7,8,9}}).rows == 3,
          "second verifier window forwards new tokens and position");
    check(pipeline.commit_pending(0).request_id == 11, "zero accepted prefix committed");
    check(pipeline.execute_window(make_frame(p,12,1), WindowSpec{0,{1,2,3}}).request_id == 12,
          "new request permits reset window ID");
    check(pipeline.commit_pending(3).rows == 3, "full accepted prefix committed");
    check(pipeline.run_window(make_frame(p,13,1)).payload == reference.payload,
          "legacy CPU mock workflow remains supported");
    check(workers.finish(error,1000), "all child workers closed and reaped");
}

class Driver final : public DeviceDriver {
public:
    WindowSpec spec;
    uint64_t req = 0, win = 0;
    uint32_t keep = UINT32_MAX;
    bool reject_begin = false, reject_commit = false;
    int aborts = 0;
    bool prepare(const Plan&, size_t, std::string&) override { return true; }
    bool begin_window(const WindowSpec& w, uint64_t r, uint64_t n, std::string& e) override {
        if (reject_begin) { e = "injected begin failure"; return false; }
        spec = w; req = r; win = n; return true;
    }
    bool enqueue_upload(const uint8_t*, size_t, std::string&) override { return true; }
    bool wait_upload(std::string&) override { return true; }
    bool enqueue_execute(const Frame& m, std::string& e) override {
        if (m.request_id != req || m.window_id != win || m.rows != spec.token_ids.size()) {
            e = "mismatched device window context"; return false;
        }
        return true;
    }
    bool wait_execute(std::string&) override { return true; }
    bool enqueue_download(uint8_t*, size_t, std::string&) override { return true; }
    bool wait_download(std::string&) override { return true; }
    bool enqueue_commit(uint64_t, uint64_t, std::string& e) override {
        e = "legacy full-window commit forbidden"; return false;
    }
    bool enqueue_commit_prefix(uint64_t r, uint64_t w, uint32_t k, std::string& e) override {
        if (reject_commit || r != req || w != win) { e = "injected prefix failure"; return false; }
        keep = k; return true;
    }
    bool wait_commit(std::string&) override { return true; }
    void abort() noexcept override { ++aborts; }
    void shutdown() noexcept override {}
};
void device_tests() {
    Plan p = parse_plan("cuda:0:0-3", 3, {2,3}, 1000);
    auto frame = make_frame(p);
    std::string error;
    Driver fake;
    DeviceStageExecutor worker(fake);
    check(worker.prepare(p,0,error), "device adapter prepares");
    check(worker.begin_window(make_spec(),11,1,error), "device receives tokens/position");
    Frame out;
    check(worker.run_window(frame,out,error), "device sequence executes with metadata");
    check(fake.spec.pos0 == 37 && fake.spec.token_ids == make_spec().token_ids,
          "device retains exact context");
    check(worker.commit_prefix(11,1,1,error), "device commits partial prefix");
    check(fake.keep == 1 && !worker.poisoned(), "no hidden full-window commit");
    check(worker.begin_window(WindowSpec{40,{7,8,9}},11,2,error), "device accepts next window");
    frame.window_id = 2;
    check(worker.run_window(frame,out,error), "second staged execution succeeds");
    check(worker.commit_prefix(11,2,0,error), "device accepts zero accepted tokens");
    check(fake.keep == 0, "zero prefix reaches device unchanged");
    worker.shutdown();

    Driver refusal; refusal.reject_begin = true;
    DeviceStageExecutor denied(refusal);
    check(denied.prepare(p,0,error), "refusing driver prepared");
    check(!denied.begin_window(make_spec(),11,1,error), "metadata admission failure returned");
    check(denied.poisoned() && refusal.aborts == 1, "device abort after failed admission");

    Driver partial; partial.reject_commit = true;
    DeviceStageExecutor faulty(partial);
    check(faulty.prepare(p,0,error), "faulty driver prepared");
    check(faulty.begin_window(make_spec(),11,1,error), "faulty driver context received");
    check(faulty.run_window(make_frame(p),out,error), "provisional device execution succeeds");
    check(!faulty.commit_prefix(11,1,2,error), "commit failure propagated");
    check(faulty.poisoned() && partial.aborts == 1, "failed commit aborts stage");
}
void mismatched_stage() {
    Plan p = parse_plan("hip:0:0-3", 3, {2,3}, 456);
    int sock[2]{};
    check(::socketpair(AF_UNIX, SOCK_STREAM, 0, sock)==0, "socketpair");
    pid_t child = ::fork();
    check(child >= 0, "fork");
    if(child==0) {
        ::close(sock[0]);
        try { MockStageExecutor e; serve_worker_stateful(sock[1],p,0,e,1000,false); ::_exit(0); }
        catch (...) { ::_exit(2); }
    }
    ::close(sock[1]);
    auto hello = receive_frame(sock[0],1000);
    check(hello && hello->kind==Kind::hello, "worker hello");
    Frame ready = *hello; ready.kind = Kind::ready; ready.identity = {};
    send_frame(sock[0], ready,1000);
    Frame context = make_frame(p); context.payload.clear();
    context.kind = Kind::window_spec; context.window = make_spec();
    send_frame(sock[0],context,1000);
    send_frame(sock[0],make_frame(p,11,2),1000); // mismatched window ID
    auto failure = receive_frame(sock[0],1000);
    check(failure && failure->kind==Kind::failure, "context/data mismatch is rejected");
    int status=0;
    check(::waitpid(child,&status,0)==child && WIFEXITED(status) && WEXITSTATUS(status)==0,
          "worker exits after mismatch");
    ::close(sock[0]);
}
void incorrect_ack() {
    Plan p = parse_plan("cuda:0:0-3",3,{2,3},789);
    int fd[2]{};
    check(::socketpair(AF_UNIX,SOCK_STREAM,0,fd)==0, "ack socketpair");
    pid_t child=::fork();
    check(child>=0,"ack fork");
    if(child==0) {
        ::close(fd[0]);
        try {
            Frame hello=make_frame(p,1,1,1);
            hello.kind=Kind::hello; hello.payload.clear();
            hello.identity={static_cast<uint32_t>(Backend::cuda),0,0,3};
            send_frame(fd[1],hello,1000);
            auto ready=receive_frame(fd[1],1000);
            if(!ready || ready->kind!=Kind::ready) ::_exit(2);
            auto context=receive_frame(fd[1],1000);
            auto data=receive_frame(fd[1],1000);
            if(!context || !data || context->kind!=Kind::window_spec || data->kind!=Kind::data)
                ::_exit(3);
            send_frame(fd[1],*data,1000);
            auto commit=receive_frame(fd[1],1000);
            if(!commit || commit->kind!=Kind::commit_prefix || commit->keep_count!=1)
                ::_exit(4);
            commit->kind=Kind::committed_prefix; commit->keep_count=2;
            send_frame(fd[1],*commit,1000);
            ::_exit(0);
        } catch (...) { ::_exit(5); }
    }
    ::close(fd[1]);
    PipelineSession pipeline(p,{fd[0]},1000,false);
    pipeline.initialize();
    check(pipeline.execute_window(make_frame(p),make_spec()).rows==3, "provisional result from peer");
    fails([&] { (void) pipeline.commit_pending(1); }, "wrong ack count must not commit");
    check(pipeline.poisoned(), "mismatched accepted prefix poisons session");
    fails([&] { (void) pipeline.execute_window(make_frame(p,11,2),make_spec()); },
          "poisoned session cannot publish more work");
    ::close(fd[0]);
    int status=0;
    check(::waitpid(child,&status,0)==child && WIFEXITED(status) && WEXITSTATUS(status)==0,
          "bad ack peer exited");
}
}
int main() {
    try {
        wire_tests();
        run_three_workers(false);
        run_three_workers(true);
        device_tests();
        mismatched_stage();
        incorrect_ack();
        std::cout << "verifier window metadata tests: PASS (" << assertions
                  << " assertions, real isolated CPU mocks, no GPU)\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "verifier window metadata tests: FAIL: " << e.what() << '\n';
        return 1;
    }
}
