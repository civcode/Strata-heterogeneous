// Exec-isolated process supervision tests: no CUDA/HIP libraries or GPUs.
#include "strata_heterogeneous/host.hpp"
#include "strata_heterogeneous/process.hpp"

#include <csignal>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>

using namespace strata::heterogeneous;
namespace {
int assertions = 0;
void check(bool value, const char* reason) {
    ++assertions;
    if (!value) throw std::runtime_error(reason);
}
template<class Fn> void should_throw(Fn&& fn, const char* reason) {
    bool caught = false;
    try { fn(); } catch (const std::exception&) { caught = true; }
    check(caught, reason);
}
Frame input(const Plan& plan, uint64_t request, uint64_t window) {
    Frame f;
    f.request_id = request; f.window_id = window;
    f.rows = 2; f.geometry = plan.geometry; f.plan_fingerprint = plan.fingerprint();
    f.payload.assign(static_cast<size_t>(f.rows * plan.geometry.floats_per_row()), 0.75f);
    return f;
}
} // namespace
int main() {
    try {
        static_assert(!std::is_copy_constructible_v<WorkerProcessGroup>);
        static_assert(std::is_move_constructible_v<WorkerProcessGroup>);
        const std::string dir = executable_directory();
        const std::string cuda = dir + "/strata_heterogeneous_mock_cuda_worker";
        const std::string hip = dir + "/strata_heterogeneous_mock_hip_worker";
        const Plan plan = parse_plan("cuda:0:0-3,hip:0:3-7,hip:1:7-10", 10, {2,6}, 999);
        check(plan_spec(plan) == "cuda:0:0-3,hip:0:3-7,hip:1:7-10", "plan serialization");
        auto group = WorkerProcessGroup::launch(plan, cuda, hip);
        check(group.size() == 3, "three independent worker processes");
        PipelineSession session(plan, group.sockets(), 1000);
        session.initialize();
        check(session.initialized(), "worker identity exchange finished");
        auto in = input(plan, 41, 8);
        auto out = session.run_window(in);
        Frame expected = in;
        std::string error;
        check(mock_execute({Backend::cuda, 0, 0, 10}, in, expected, error), "reference mock");
        check(out.payload == expected.payload, "multi-process exact layer order");
        check(session.run_window(input(plan, 41, 9)).payload == expected.payload,
              "same-request second window");
        check(session.run_window(input(plan, 42, 1)).payload == expected.payload,
              "new request id allowed");
        check(group.finish(error, 1000), "all workers exit on coordinator EOF");
        check(group.size() == 0, "workers reaped");
        check(group.sockets().empty(), "socket descriptors closed");
        group.terminate(); // safe repeated cleanup
        check(group.size() == 0, "terminate idempotent");

        // Backend marker must be enforced by each independent executable.
        auto wrong = WorkerProcessGroup::launch(plan, hip, hip);
        PipelineSession rejected(plan, wrong.sockets(), 200);
        should_throw([&] { rejected.initialize(); }, "HIP mock refuses CUDA stage");
        check(rejected.poisoned(), "worker mismatch poisons session");
        check(!wrong.finish(error, 500), "wrong backend child exits nonzero");

        // Failure in a *later* spawn must not leave the first worker orphaned.
        should_throw([&] {
            (void) WorkerProcessGroup::launch(plan, cuda, "/nonexistent/strata-mock-hip");
        }, "missing second executable cleans already-spawned first child");
        should_throw([&] {
            (void) WorkerProcessGroup::launch(plan, "relative/cuda", hip);
        }, "relative worker executable rejected");

        // A worker exiting before hello is not a successful handshake.
        auto dead = WorkerProcessGroup::launch(plan, cuda, "/bin/false");
        PipelineSession failed(plan, dead.sockets(), 250);
        should_throw([&] { failed.initialize(); }, "worker exit before hello rejected");
        check(failed.poisoned(), "closed subprocess poisons session");
        check(!dead.finish(error, 500), "non-zero child status surfaced");

        // Incomplete startup also gets bounded forced cleanup.
        auto abandoned = WorkerProcessGroup::launch(plan, cuda, hip);
        abandoned.terminate();
        check(abandoned.size() == 0, "abandoned workers killed and reaped");

        // A process ignoring EOF must not hang cleanup indefinitely.
        const Plan hang_plan = parse_plan("hip:0:0-10", 10, {2, 6}, 999);
        auto stuck = WorkerProcessGroup::launch(hang_plan, cuda, dir + "/strata_heterogeneous_hang_worker");
        check(!stuck.finish(error, 50), "nonresponding worker forcibly terminated at deadline");
        check(error.find("deadline") != std::string::npos, "timeout reported explicitly");
        check(stuck.size() == 0, "timed-out child reaped");
        std::cout << "heterogeneous process tests: PASS (" << assertions
                  << " assertions; separate backend-tagged CPU workers; no GPU)\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "heterogeneous process tests: FAIL: " << ex.what() << '\n';
        return 1;
    }
}
