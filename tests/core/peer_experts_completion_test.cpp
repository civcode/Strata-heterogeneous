// R1 failure/completion policy unit tests, CPU only; actual PeerExperts is tested
// separately in a native fixture on HIP/CUDA.
#include "strata/core/peer_experts_completion.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace strata::core::detail;

static void require(bool condition, const char* what) {
    if (!condition) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(2); }
}

static int run(const std::string& scenario) {
    fake::reset();
    const cudaStream_t stream = reinterpret_cast<cudaStream_t>(uintptr_t{1});
    const cudaEvent_t event = reinterpret_cast<cudaEvent_t>(uintptr_t{1});
    if (scenario == "close-success") {
        {
            PeerDeviceScope scope(1);
            peer_check(cudaDeviceSynchronize(), "peer close");
            require(fake::current_device == 1, "selected helper");
            require(fake::device_sync_calls == 1, "completion before release");
        }
        require(fake::current_device == 0, "restored entry device");
    } else if (scenario == "select-failure") {
        fake::fail_set = true;
        PeerDeviceScope scope(1);
    } else if (scenario == "close-sync-failure") {
        PeerDeviceScope scope(1);
        fake::fail_device_sync = true;
        peer_check(cudaDeviceSynchronize(), "peer close");
    } else if (scenario == "issue-drain-success") {
        peer_drain_or_exit(stream, "peer failed copy issue");
        require(fake::stream_calls == 1, "drained once before release");
    } else if (scenario == "issue-drain-failure" || scenario == "finish-drain-failure") {
        fake::fail_stream = true;
        peer_drain_or_exit(stream, "peer queued work completion unknown");
    } else if (scenario == "event-not-ready") {
        fake::event_not_ready = true;
        require(!peer_event_ready(event, "peer event"), "unready event not published");
    } else if (scenario == "event-query-failure") {
        fake::fail_event_query = true;
        (void) peer_event_ready(event, "peer event");
    } else if (scenario == "event-wait-failure") {
        fake::fail_event_wait = true;
        peer_check(cudaEventSynchronize(event), "peer pending wait");
    } else {
        std::fprintf(stderr, "unknown scenario: %s\n", scenario.c_str());
        return 2;
    }
    std::puts("RELEASE");
    std::fflush(stdout);
    return 0;
}

int main(int argc, char** argv) {
    require(argc == 2, "one scenario required");
    const std::string scenario = argv[1];
    const bool fatal = scenario == "select-failure" || scenario == "close-sync-failure" ||
                       scenario == "issue-drain-failure" || scenario == "finish-drain-failure" ||
                       scenario == "event-query-failure" || scenario == "event-wait-failure";
    int fds[2];
    require(pipe(fds) == 0, "pipe");
    const pid_t child = fork();
    require(child >= 0, "fork");
    if (child == 0) {
        close(fds[0]);
        require(dup2(fds[1], STDOUT_FILENO) >= 0, "dup2");
        close(fds[1]);
        const int rc = run(scenario);
        std::_Exit(rc);
    }
    close(fds[1]);
    std::string output;
    char buf[256];
    for (ssize_t count; (count = read(fds[0], buf, sizeof(buf))) > 0;)
        output.append(buf, static_cast<size_t>(count));
    close(fds[0]);
    int status = 0;
    require(waitpid(child, &status, 0) == child, "waitpid");
    require(WIFEXITED(status), "child exited");
    require(WEXITSTATUS(status) == (fatal ? 1 : 0), "exit contract");
    require((output.find("RELEASE") != std::string::npos) != fatal, "no owner release after terminal error");
    if (fatal) require(output.find("ERR ") != std::string::npos, "terminal error marker");
    std::printf("PASS: %s\n", scenario.c_str());
}
