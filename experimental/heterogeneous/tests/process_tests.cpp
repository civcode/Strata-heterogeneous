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

        // The opt-in shared-memory path carries only control frames on the
        // sockets. All workers map the same sealed memfd from SCM_RIGHTS.
        auto shared_group = WorkerProcessGroup::launch(plan, cuda, hip, true);
        PipelineSession shared_session(plan, shared_group.sockets(), 1000, true);
        shared_session.initialize();
        for (uint64_t window : {1ULL, 2ULL, 3ULL}) {
            auto shared_input = input(plan, 100, window);
            auto shared_output = shared_session.run_window(shared_input);
            Frame shared_reference = shared_input;
            std::string mock_error;
            check(mock_execute({Backend::cuda, 0, 0, 10}, shared_input, shared_reference, mock_error),
                  "shared reference runs");
            check(shared_output.payload == shared_reference.payload, "shared workers preserve layer order");
            check(shared_output.kind == Kind::data, "shared output returns ordinary data frame");
        }
        auto many_rows = input(plan, 101, 1);
        many_rows.rows = 4096;
        many_rows.payload.assign(static_cast<size_t>(many_rows.rows * plan.geometry.floats_per_row()), 0.375f);
        auto many_output = shared_session.run_window(many_rows);
        Frame many_expected = many_rows;
        std::string many_error;
        check(mock_execute({Backend::cuda, 0, 0, 10}, many_rows, many_expected, many_error),
              "large shared-memory reference");
        check(many_output.payload == many_expected.payload, "multirow shared handoff matches reference");
        check(shared_group.finish(error, 1000), "all shared workers finish and are reaped");
        check(shared_group.size() == 0, "shared process cleanup");

        // M2.0: stage-specific argv must reach only its own worker process.
        const std::string probe = dir + "/strata_heterogeneous_worker_argv_probe";
        const std::vector<std::string> paths = {
            "/tmp/m20-stage-0.cfg", "/tmp/m20-stage-1.cfg", "/tmp/m20-stage-2.cfg"
        };
        auto routed = WorkerProcessGroup::launch(plan, probe, probe, false, {}, paths);
        check(routed.size() == 3, "three stage-indexed children launched");
        check(routed.finish(error, 1000), "correct config in every child process");
        auto routed_shared = WorkerProcessGroup::launch(plan, probe, probe, true, {}, paths);
        check(routed_shared.finish(error, 1000), "stage config routes in shared mode");
        should_throw([&] {
            (void)WorkerProcessGroup::launch(plan, probe, probe, false, {},
                                             {paths[0]});
        }, "missing stage entries refused before any spawn");
        should_throw([&] {
            (void)WorkerProcessGroup::launch(plan, probe, probe, false,
                                             "/tmp/legacy.cfg", paths);
        }, "mixed legacy/per-stage config refused");
        should_throw([&] {
            (void)WorkerProcessGroup::launch(plan, probe, probe, false, {},
                                             {"relative.cfg", paths[1], paths[2]});
        }, "relative stage config refused");
        should_throw([&] {
            (void)WorkerProcessGroup::launch(plan, probe, probe, false, {},
                                             {paths[0], "/tmp/../bad.cfg", paths[2]});
        }, "parent directory component refused");
        should_throw([&] {
            (void)WorkerProcessGroup::launch(plan, probe, probe, false, {},
                                             {paths[0], "/tmp/hip\n.cfg", paths[2]});
        }, "newline stage config refused");

        // Mismatched data transport modes must fail closed rather than hang.
        auto mismatched = WorkerProcessGroup::launch(plan, cuda, hip, true);
        PipelineSession no_shared(plan, mismatched.sockets(), 200, false);
        should_throw([&] { no_shared.initialize(); }, "shared worker rejects socket mode at handshake");
        check(no_shared.poisoned(), "mismatched transport poisons coordinator");
        mismatched.terminate();
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
