// Fail-closed mapped-host teardown contract. No CUDA/HIP device is required.
#include "strata/core/remote_opt_completion.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {
int active = 0;
int fail_select = -1, fail_sync = -1;
int completions[2] = {0, 0};
void check(bool yes, const char* what) {
    if (!yes) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(2); }
}
} // namespace

cudaError_t cudaGetDevice(int* device) { *device = active; return cudaSuccess; }
cudaError_t cudaSetDevice(int device) {
    if (device == fail_select) return cudaErrorUnknown;
    active = device;
    return cudaSuccess;
}
cudaError_t cudaDeviceSynchronize() {
    if (active == fail_sync) return cudaErrorUnknown;
    if (active >= 0 && active < 2) ++completions[active];
    return cudaSuccess;
}
const char* cudaGetErrorString(cudaError_t) { return "injected GPU completion failure"; }

int main(int argc, char** argv) {
    check(argc == 2, "expected one test scenario");
    const std::string mode = argv[1];
    const bool success = mode == "success";
    check(success || mode == "primary-sync-failure" || mode == "helper-sync-failure" ||
          mode == "helper-select-failure", "unknown scenario");
    int fds[2]; check(pipe(fds) == 0, "pipe");
    const pid_t child = fork(); check(child >= 0, "fork");
    if (child == 0) {
        close(fds[0]);
        check(dup2(fds[1], STDOUT_FILENO) >= 0, "dup2");
        close(fds[1]);
        std::setvbuf(stdout, nullptr, _IONBF, 0);
        if (mode == "primary-sync-failure") fail_sync = 0;
        if (mode == "helper-sync-failure") fail_sync = 1;
        if (mode == "helper-select-failure") fail_select = 1;
        // The destructor executes these in this order before any cudaFree/cudaFreeHost.
        strata::core::detail::remote_opt_complete(0, "primary teardown");
        strata::core::detail::remote_opt_complete(1, "helper teardown");
        strata::core::detail::remote_opt_select(0, "restore device");
        check(active == 0 && completions[0] == 1 && completions[1] == 1, "completed both devices");
        std::puts("RELEASE"); // owner release is reached only after both devices completed
        std::_Exit(0);
    }
    close(fds[1]);
    std::string output; char buf[256]; ssize_t n = 0;
    while ((n = read(fds[0], buf, sizeof buf)) > 0) output.append(buf, (size_t)n);
    close(fds[0]);
    int status = 0;
    check(waitpid(child, &status, 0) == child, "waitpid");
    check(WIFEXITED(status), "child exit");
    if (success) check(WEXITSTATUS(status) == 0 && output.find("RELEASE") != std::string::npos,
                       "success releases after both device completions");
    else check(WEXITSTATUS(status) == 1 && output.find("ERR ") != std::string::npos &&
               output.find("RELEASE") == std::string::npos,
               "unknown completion or device selection exits before release");
    return 0;
}
