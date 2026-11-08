#include "strata/core/expert_cache_completion.hpp"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {
void require(bool ok, const char* label = "completion invariant") {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(3); }
}
int stream_calls = 0;
int device_calls = 0;
cudaError_t stream_result = cudaSuccess;
cudaError_t device_result = cudaSuccess;
int current_device = 0;
int synchronized_device = -1;
int get_calls = 0, select_calls = 0;
bool get_fails = false, select_fails = false, restore_fails = false;

bool fatal_child(const char* scenario) {
    int fds[2];
    require(pipe(fds) == 0);
    const pid_t pid = fork();
    require(pid >= 0);
    if (pid == 0) {
        close(fds[0]);
        // An automatic owner of staging/readback data; a failed drain must
        // terminate without invoking its destructor.
        struct Owner {
            int fd;
            ~Owner() { const char released = 'D'; (void)write(fd, &released, 1); }
        };
        {
        Owner owner{fds[1]};
        stream_calls = device_calls = 0;
        stream_result = cudaErrorUnknown;
        device_result = std::strcmp(scenario, "owner-restore-unknown") == 0 ? cudaSuccess : cudaErrorUnknown;
        std::string error;
        if (std::strcmp(scenario, "issue-unknown") == 0 ||
            std::strcmp(scenario, "readback-unknown") == 0)
            (void) strata::core::detail::expert_cache_status(
                cudaErrorUnknown, nullptr, error, "ExpertCache::copy");
        else if (std::strcmp(scenario, "sync-unknown") == 0)
            (void) strata::core::detail::expert_cache_status(
                cudaErrorUnknown, nullptr, error, "ExpertCache::sync_queued");
        else if (std::strcmp(scenario, "close-unknown") == 0)
            strata::core::detail::expert_cache_close_or_exit();
        else if (std::strcmp(scenario, "owner-get-unknown") == 0) {
            get_fails = true;
            strata::core::detail::ExpertCacheCloseDevice scope(1, true);
        } else if (std::strcmp(scenario, "owner-select-unknown") == 0) {
            select_fails = true;
            strata::core::detail::ExpertCacheCloseDevice scope(1, true);
        } else if (std::strcmp(scenario, "owner-sync-unknown") == 0) {
            strata::core::detail::ExpertCacheCloseDevice scope(1, true);
            strata::core::detail::expert_cache_close_or_exit();
        } else if (std::strcmp(scenario, "owner-restore-unknown") == 0) {
            restore_fails = true;
            { strata::core::detail::ExpertCacheCloseDevice scope(1, true);
              strata::core::detail::expert_cache_close_or_exit(); }
        } else if (std::strcmp(scenario, "owner-missing") == 0) {
            strata::core::detail::ExpertCacheCloseDevice scope(-1, true);
        }
        } // An incorrectly returning helper destroys the owner and writes D.
        _exit(42);  // A return means the fail-closed policy was violated.
    }
    close(fds[1]);
    char marker = 0;
    const ssize_t got = read(fds[0], &marker, 1);
    close(fds[0]);
    int status = 0;
    require(waitpid(pid, &status, 0) == pid);
    return WIFEXITED(status) && WEXITSTATUS(status) == 1 && got == 0;
}
}

cudaError_t cudaGetDevice(int* device) {
    ++get_calls;
    if (get_fails) return cudaErrorUnknown;
    *device = current_device;
    return cudaSuccess;
}
cudaError_t cudaSetDevice(int device) {
    ++select_calls;
    if ((select_fails && device == 1) || (restore_fails && device == 0)) return cudaErrorUnknown;
    current_device = device;
    return cudaSuccess;
}
cudaError_t cudaStreamSynchronize(cudaStream_t) {
    ++stream_calls;
    return stream_result;
}
cudaError_t cudaDeviceSynchronize() {
    ++device_calls;
    synchronized_device = current_device;
    return device_result;
}
const char* cudaGetErrorString(cudaError_t e) {
    return e == cudaSuccess ? "success" : "injected failure";
}

int main(int argc, char** argv) {
    require(argc == 2);
    const std::string scenario = argv[1];
    std::string error;
    if (scenario == "success") {
        require(strata::core::detail::expert_cache_status(cudaSuccess, nullptr, error, "copy"));
        require(error.empty() && stream_calls == 0 && device_calls == 0);
    } else if (scenario == "issue-drain" || scenario == "readback-drain" || scenario == "sync-drain") {
        require(!strata::core::detail::expert_cache_status(cudaErrorUnknown, nullptr, error, "copy"));
        require(error.find("injected failure") != std::string::npos);
        require(stream_calls == 1 && device_calls == 0);
    } else if (scenario == "owner-switch") {
        current_device = 0;
        { strata::core::detail::ExpertCacheCloseDevice scope(1, true);
          require(current_device == 1 && get_calls == 1 && select_calls == 1);
          strata::core::detail::expert_cache_close_or_exit();
          require(synchronized_device == 1 && device_calls == 1);
        }
        require(current_device == 0 && select_calls == 2);
    } else if (scenario == "owner-already-selected") {
        current_device = 1;
        { strata::core::detail::ExpertCacheCloseDevice scope(1, true);
          strata::core::detail::expert_cache_close_or_exit(); }
        require(current_device == 1 && synchronized_device == 1 && select_calls == 0);
    } else if (scenario == "owner-no-storage") {
        { strata::core::detail::ExpertCacheCloseDevice scope(-1, false); }
        require(get_calls == 0 && select_calls == 0 && device_calls == 0);
    } else if (scenario == "close-success") {
        strata::core::detail::expert_cache_close_or_exit();
        require(device_calls == 1 && stream_calls == 0);
    } else if (scenario == "issue-unknown" || scenario == "readback-unknown" ||
               scenario == "sync-unknown" || scenario == "close-unknown" ||
               scenario == "owner-get-unknown" || scenario == "owner-select-unknown" ||
               scenario == "owner-sync-unknown" || scenario == "owner-restore-unknown" ||
               scenario == "owner-missing") {
        require(fatal_child(scenario.c_str()));
    } else {
        std::fprintf(stderr, "unknown test scenario %s\n", argv[1]);
        return 2;
    }
    std::printf("ExpertCache completion: %s passed\n", argv[1]);
}
