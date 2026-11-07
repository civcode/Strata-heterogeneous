// CPU-only simulated device: validates the GPU transfer synchronization
// contract and worker fail-stop behavior without CUDA/HIP headers or GPUs.
#include "strata_heterogeneous/device.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

using namespace strata::heterogeneous;
namespace {
int assertions = 0;
void check(bool good, const std::string& reason) {
    ++assertions;
    if (!good) throw std::runtime_error(reason);
}
template<class Fn> void fails(Fn&& fn, const char* why) {
    bool failed = false;
    try { fn(); } catch (const std::exception&) { failed = true; }
    check(failed, why);
}

// Simulates queued operations that are not effective until the matching wait.
// A test of sequencing and failure handling, NOT a GPU kernel simulation.
class FakeDevice final : public DeviceDriver {
public:
    enum class State { fresh, ready, upload_queued, uploaded, execute_queued, executed,
                       download_queued, downloaded, commit_queued, committed, failed };
    std::vector<std::string> calls;
    std::string fault;
    bool throw_fault = false;
    int aborts = 0;
    int shutdowns = 0;
    bool finished_commit = false;
    State state = State::fresh;

    bool prepare(const Plan& p, size_t index, std::string& error) override {
        if (!step("prepare", error)) return false;
        check(state == State::fresh, "fake state starts empty");
        stage_ = p.stages.at(index);
        state = State::ready;
        return true;
    }
    bool enqueue_upload(const uint8_t* src, size_t bytes, std::string& error) override {
        if (!step("enqueue_upload", error)) return false;
        check(state == State::ready || state == State::committed,
              "upload waits for previous commit completion");
        source_ = src;
        bytes_ = bytes;
        state = State::upload_queued;
        return true;
    }
    bool wait_upload(std::string& error) override {
        if (!step("wait_upload", error)) return false;
        check(state == State::upload_queued, "input transfer was enqueued");
        storage_.assign(source_, source_ + bytes_);
        source_ = nullptr;
        state = State::uploaded;
        return true;
    }
    bool enqueue_execute(const Frame& metadata, std::string& error) override {
        if (!step("enqueue_execute", error)) return false;
        check(state == State::uploaded, "execution cannot precede upload completion");
        metadata_ = metadata;
        state = State::execute_queued;
        return true;
    }
    bool wait_execute(std::string& error) override {
        if (!step("wait_execute", error)) return false;
        check(state == State::execute_queued, "execution was enqueued");
        ActivationView view(storage_.data(), storage_.size(), metadata_.geometry, metadata_.rows);
        ByteRange all{view.data(), view.bytes()};
        for (uint32_t layer = stage_.first_layer; layer < stage_.end_layer; ++layer)
            for (size_t i = 0; i < all.bytes / sizeof(float); ++i)
                all.write_f32(i, all.read_f32(i) + static_cast<float>(layer + 1) * 0.0001f);
        state = State::executed;
        return true;
    }
    bool enqueue_download(uint8_t* destination, size_t bytes, std::string& error) override {
        if (!step("enqueue_download", error)) return false;
        check(state == State::executed, "download requires completed compute");
        check(bytes == storage_.size(), "download size must match input window");
        destination_ = destination;
        state = State::download_queued;
        return true;
    }
    bool wait_download(std::string& error) override {
        if (!step("wait_download", error)) {
            // Partial host mutation during failed D2H is explicitly possible.
            if (destination_) destination_[0] ^= 0x44;
            return false;
        }
        check(state == State::download_queued, "download was enqueued");
        std::memcpy(destination_, storage_.data(), storage_.size());
        destination_ = nullptr;
        state = State::downloaded;
        return true;
    }
    bool enqueue_commit(uint64_t request, uint64_t window, std::string& error) override {
        if (!step("enqueue_commit", error)) return false;
        check(state == State::downloaded, "cannot commit before download completes");
        check(request == metadata_.request_id && window == metadata_.window_id,
              "commit must match provisional stage window");
        state = State::commit_queued;
        return true;
    }
    bool wait_commit(std::string& error) override {
        if (!step("wait_commit", error)) return false;
        check(state == State::commit_queued, "commit must be enqueued");
        finished_commit = true;
        state = State::committed;
        return true;
    }
    void abort() noexcept override { ++aborts; state = State::failed; calls.push_back("abort"); }
    void shutdown() noexcept override { ++shutdowns; calls.push_back("shutdown"); }
private:
    bool step(const char* name, std::string& error) {
        calls.emplace_back(name);
        if (fault == name) {
            if (throw_fault) throw std::runtime_error(std::string("injected exception: ") + name);
            error = std::string("injected failure: ") + name;
            return false;
        }
        return true;
    }
    Stage stage_{};
    Frame metadata_;
    std::vector<uint8_t> storage_;
    const uint8_t* source_ = nullptr;
    uint8_t* destination_ = nullptr;
    size_t bytes_ = 0;
};

Frame input(const Plan& p, uint64_t req = 9, uint64_t win = 1) {
    Frame f;
    f.stage_index = 0;
    f.request_id = req;
    f.window_id = win;
    f.rows = 2;
    f.plan_fingerprint = p.fingerprint();
    f.geometry = p.geometry;
    f.payload.resize(f.rows * p.geometry.floats_per_row());
    for (size_t i = 0; i < f.payload.size(); ++i) f.payload[i] = static_cast<float>(i % 17) / 17.f;
    return f;
}
void test_normal_and_view_modes() {
    const Plan p = parse_plan("cuda:0:0-4", 4, {2, 3}, 111);
    FakeDevice gpu;
    DeviceStageExecutor worker(gpu);
    std::string error;
    check(worker.prepare(p, 0, error), "prepare device");
    Frame in = input(p), out;
    check(worker.run_window(in, out, error), "socket mode staged device execution");
    check(worker.pending() && !worker.poisoned(), "window provisional until commit");
    check(!gpu.finished_commit, "device state not committed during execution");
    Frame reference;
    check(mock_execute(p.stages[0], in, reference, error), "CPU reference");
    check(out.payload == reference.payload, "fake GPU result equals stage CPU arithmetic");
    check(gpu.calls == std::vector<std::string>({"prepare", "enqueue_upload", "wait_upload",
          "enqueue_execute", "wait_execute", "enqueue_download", "wait_download"}),
          "every transfer fence reached before output");
    check(worker.commit_window(in.request_id, in.window_id, error), "device commit acknowledged");
    check(gpu.finished_commit && !worker.pending(), "commit waits for completion");
    check(gpu.calls.at(gpu.calls.size() - 2) == "enqueue_commit" &&
          gpu.calls.back() == "wait_commit", "ordered commit and wait");
    // Follow-up request reuses the stage without requiring a process restart.
    auto next = input(p, 9, 2);
    std::vector<uint8_t> bytes(next.payload.size() * sizeof(float));
    std::memcpy(bytes.data(), next.payload.data(), bytes.size());
    ActivationView view(bytes.data(), bytes.size(), next.geometry, next.rows);
    Frame metadata = next;
    metadata.payload.clear();
    check(worker.run_staged_window(metadata, view, error), "direct mapped host view staging");
    Frame expected;
    check(mock_execute(p.stages[0], next, expected, error), "shared reference");
    check(std::memcmp(bytes.data(), expected.payload.data(), bytes.size()) == 0,
          "direct and socket staging parity");
    check(worker.commit_window(next.request_id, next.window_id, error), "second commit");
    worker.shutdown(); worker.shutdown();
    check(gpu.shutdowns == 1 && gpu.aborts == 0, "clean shutdown exactly once without abort");
}
void test_invalid_and_failed_steps() {
    const Plan p = parse_plan("hip:0:0-3", 3, {2, 3}, 222);
    const Frame in = input(p);
    const std::vector<std::string> steps = {"enqueue_upload", "wait_upload", "enqueue_execute",
      "wait_execute", "enqueue_download", "wait_download"};
    for (const std::string& target : steps) {
        FakeDevice gpu;
        DeviceStageExecutor worker(gpu);
        std::string error;
        check(worker.prepare(p, 0, error), "prepare before fault test");
        gpu.fault = target;
        Frame output;
        check(!worker.run_window(in, output, error), "injected step failure propagates");
        check(error.find(target) != std::string::npos, "step error identifies failing operation");
        check(worker.poisoned() && !worker.pending(), "step failure poisons stage");
        check(gpu.aborts == 1, "device aborted exactly once");
        check(output.payload.empty(), "socket mode never publishes failed data");
        const auto n = gpu.calls.size();
        check(!worker.run_window(in, output, error), "failed stage cannot restart");
        check(gpu.calls.size() == n, "failed stage schedules no further GPU work");
        worker.abandon_window(); worker.shutdown();
        check(gpu.aborts == 1 && gpu.shutdowns == 1, "cleanup idempotent");
    }
    for (const std::string& target : {std::string("enqueue_commit"), std::string("wait_commit")}) {
        FakeDevice gpu;
        DeviceStageExecutor worker(gpu);
        std::string error;
        check(worker.prepare(p, 0, error), "prepare for failed commit");
        Frame out;
        check(worker.run_window(in, out, error), "provisional window succeeds");
        gpu.fault = target;
        check(!worker.commit_window(in.request_id, in.window_id, error), "commit failure propagated");
        check(worker.poisoned() && gpu.aborts == 1 && !worker.pending(),
              "failed commit forces unusable state");
        check(error.find(target) != std::string::npos, "commit error identifies failing phase");
        worker.shutdown();
    }
    for (const std::string& target : {std::string("enqueue_upload"), std::string("wait_execute")}) {
        FakeDevice gpu;
        DeviceStageExecutor worker(gpu);
        std::string error;
        check(worker.prepare(p, 0, error), "prepare for thrown driver exception");
        gpu.fault = target; gpu.throw_fault = true;
        Frame out;
        check(!worker.run_window(in, out, error), "driver exception translated into failure");
        check(error.find("exception") != std::string::npos, "exception detail surfaced");
        check(worker.poisoned() && gpu.aborts == 1, "exception fails closed");
    }
    FakeDevice gpu;
    DeviceStageExecutor worker(gpu);
    std::string error;
    check(worker.prepare(p, 0, error), "prepare before malformed metadata");
    Frame metadata = in;
    metadata.payload.clear();
    std::vector<uint8_t> bytes(in.payload.size() * sizeof(float));
    ActivationView view(bytes.data(), bytes.size(), in.geometry, in.rows);
    metadata.stage_index = 1;
    check(!worker.run_staged_window(metadata, view, error), "wrong stage metadata refused");
    check(gpu.calls.size() == 2 && gpu.calls.back() == "abort",
          "invalid metadata rejected before GPU scheduling");
    check(worker.poisoned(), "bad metadata poisons stage");

    FakeDevice other;
    DeviceStageExecutor bad_prepare(other);
    other.fault = "prepare";
    check(!bad_prepare.prepare(p, 0, error) && bad_prepare.poisoned(), "prepare failure poison");
    check(other.aborts == 1, "prepare failure aborts resources");
}
void test_commit_identity_and_failed_download() {
    const Plan p = parse_plan("cuda:0:0-3", 3, {2, 3}, 111);
    FakeDevice gpu;
    DeviceStageExecutor worker(gpu);
    std::string error;
    check(worker.prepare(p, 0, error), "device prepared");
    Frame in = input(p), out;
    check(worker.run_window(in, out, error), "window is provisional");
    check(!worker.commit_window(in.request_id, in.window_id + 1, error), "wrong commit ID rejected");
    check(worker.poisoned() && gpu.aborts == 1 && !gpu.finished_commit,
          "mismatched commit poisons with no commit side effects");

    FakeDevice failing;
    DeviceStageExecutor stage(failing);
    check(stage.prepare(p, 0, error), "prepare for partial D2H failure");
    failing.fault = "wait_download";
    auto original = input(p);
    std::vector<uint8_t> bytes(original.payload.size() * sizeof(float));
    std::memcpy(bytes.data(), original.payload.data(), bytes.size());
    Frame metadata = original;
    metadata.payload.clear();
    ActivationView view(bytes.data(), bytes.size(), original.geometry, original.rows);
    check(!stage.run_staged_window(metadata, view, error), "injected partial download failure");
    check(bytes[0] != reinterpret_cast<const uint8_t*>(original.payload.data())[0],
          "failed D2H may already have modified host handoff slot");
    check(stage.poisoned() && failing.aborts == 1,
          "partial D2H poisons stage and must never publish result");
}
void test_in_real_worker_process_failure(bool shared, bool commit_failure) {
    const Plan p = parse_plan("hip:0:0-4", 4, {2, 3}, 2026);
    int fds[2]{};
    check(::socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0, "failure worker socketpair");
    const pid_t pid = ::fork(); // test-only fake backend, before any GPU initialization
    check(pid >= 0, "fork failing worker");
    if (pid == 0) {
        ::close(fds[0]);
        try {
            FakeDevice driver;
            driver.fault = commit_failure ? "wait_commit" : "wait_download";
            DeviceStageExecutor executor(driver);
            serve_worker_stateful(fds[1], p, 0, executor, 1000, shared);
            ::close(fds[1]);
            ::_exit(0);
        } catch (...) { ::_exit(2); }
    }
    ::close(fds[1]);
    PipelineSession session(p, {fds[0]}, 1000, shared);
    session.initialize();
    std::string cause;
    try { (void) session.run_window(input(p)); }
    catch (const std::exception& e) { cause = e.what(); }
    check(cause.find(commit_failure ? "wait_commit" : "wait_download") != std::string::npos,
          "device failure propagated through worker and coordinator");
    check(session.poisoned(), "device error poisons pipeline and suppresses result");
    fails([&] { (void) session.run_window(input(p, 9, 2)); },
          "new window rejected on poisoned process session");
    ::close(fds[0]);
    int status = 0;
    check(::waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0,
          "faulted worker stopped after sending failure");
}
void test_in_real_worker_process(bool shared) {
    const Plan p = parse_plan("cuda:0:0-3,hip:0:3-7", 7, {2, 3}, 123);
    std::vector<int> fds;
    std::vector<pid_t> pids;
    for (size_t i = 0; i < p.stages.size(); ++i) {
        int pair[2]{};
        check(::socketpair(AF_UNIX, SOCK_STREAM, 0, pair) == 0, "worker socketpair");
        pid_t pid = ::fork(); // CPU-only; no device runtime initialized
        check(pid >= 0, "worker fork");
        if (pid == 0) {
            for (int other : fds) ::close(other);
            ::close(pair[0]);
            try {
                FakeDevice driver;
                DeviceStageExecutor executor(driver);
                serve_worker_stateful(pair[1], p, i, executor, 1000, shared);
                ::close(pair[1]);
                ::_exit(0);
            } catch (const std::exception& ex) {
                std::cerr << "CPU fake worker exception: " << ex.what() << '\n';
                ::_exit(2);
            }
        }
        ::close(pair[1]);
        fds.push_back(pair[0]);
        pids.push_back(pid);
    }
    PipelineSession session(p, fds, 1000, shared);
    session.initialize();
    for (uint64_t w : {1ULL, 2ULL, 3ULL}) {
        Frame in = input(p, 50, w);
        const Frame got = session.run_window(in);
        Frame expected;
        std::string error;
        check(mock_execute({Backend::cuda, 0, 0, 7}, in, expected, error), "unsplit CPU reference");
        check(got.payload == expected.payload, "two-stage fake device output parity");
        check(!session.poisoned(), "successful staging and commit stays valid");
    }
    for (int fd : fds) ::close(fd);
    for (pid_t pid : pids) {
        int status = 0;
        check(::waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0,
              "device worker completes without errors");
    }
}
} // namespace
int main() {
    try {
        test_normal_and_view_modes();
        test_invalid_and_failed_steps();
        test_commit_identity_and_failed_download();
        test_in_real_worker_process(false);
        test_in_real_worker_process(true);
        for (bool shared : {false, true}) {
            test_in_real_worker_process_failure(shared, false);
            test_in_real_worker_process_failure(shared, true);
        }
        std::cout << "device transfer tests: PASS (" << assertions
                  << " assertions, ordered fake H2D/compute/D2H/commit, no GPU)\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "device transfer tests: FAIL: " << ex.what() << '\n';
        return 1;
    }
}
