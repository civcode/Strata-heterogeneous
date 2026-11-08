#include "strata/prefill/host_group_readback.hpp"
#include "strata/core/transfer_terminal.hpp"
#include <cstdlib>
#include <cstdio>
#if defined(__unix__)
#include <sys/wait.h>
#include <unistd.h>
#endif

static int calls = 0, fail_at = 0;
cudaError_t cudaStreamSynchronize(cudaStream_t) {
    return ++calls == fail_at ? cudaErrorUnknown : cudaSuccess;
}
cudaError_t cudaEventSynchronize(cudaEvent_t) {
    return ++calls == fail_at ? cudaErrorUnknown : cudaSuccess;
}
static void require(bool ok) { if (!ok) std::abort(); }
using namespace strata::prefill::detail;

int main() {
    // KV/PLE drains run even on early return, including clean cancellation. The caller reads
    // completion only AFTER these scopes and pipeline workers have finished.
    for (int failing : {0, 1, 2}) {
        calls = 0; fail_at = failing;
        PrefillCompletion completion;
        {
            PrefillStreamDrain kv{completion, reinterpret_cast<void*>(1)};
            PrefillStreamDrain ple{completion, reinterpret_cast<void*>(2)};
        }
        require(calls == 2 && completion.failed() == (failing != 0));
        (void) completion.stream(nullptr); // a later successful cleanup cannot clear the failure
        require(completion.failed() == (failing != 0));
    }
    for (int failing : {0, 1, 2, 3, 4}) {
        calls = 0; fail_at = failing;
        PrefillCompletion completion;
        { PrefillRunDrain cancelled{completion, nullptr, reinterpret_cast<void*>(1),
                                    reinterpret_cast<void*>(2), reinterpret_cast<void*>(3)}; }
        require(calls == 4 && completion.failed() == (failing != 0));
        calls = 0;
        { PrefillRunDrain healthy{completion, nullptr, reinterpret_cast<void*>(1)}; healthy.completed = true; }
        require(calls == 0); // no new synchronization on the successful path
    }
    // Helper cleanup must attempt all three independent streams; final compute/copy/KV waits
    // and event waits use the same sticky observation, including a valid null compute stream.
    for (int failing : {1, 2, 3}) {
        calls = 0; fail_at = failing;
        PrefillCompletion completion;
        for (int i = 0; i < 3; ++i) (void) completion.stream(nullptr);
        require(calls == 3 && completion.failed());
        (void) completion.event(nullptr);
        require(completion.failed());
    }
#if defined(__unix__)
    int pipefd[2]; require(pipe(pipefd) == 0);
    const pid_t pid = fork(); require(pid >= 0);
    if (pid == 0) {
        close(pipefd[0]);
        struct OperandOwner {
            int fd;
            ~OperandOwner() { const char freed = 'F'; (void) write(fd, &freed, 1); }
        } owner{pipefd[1]};
        PrefillCompletion completion;
        calls = 0; fail_at = 1;
        { PrefillStreamDrain kv{completion, reinterpret_cast<void*>(1)}; }
        // Production generation boundary: no refill/DONE/owner unwind once terminal is visible.
        if (completion.failed()) strata::core::transfer_terminal("prefill regression", "unknown completion");
        const char resumed = 'R'; (void) write(pipefd[1], &resumed, 1);
        std::exit(9);
    }
    close(pipefd[1]);
    int status = 0; require(waitpid(pid, &status, 0) == pid);
    char byte; require(read(pipefd[0], &byte, 1) == 0); close(pipefd[0]);
    require(WIFEXITED(status) && WEXITSTATUS(status) == 1);
#endif
    std::puts("prefill completion: cleanup, sticky state, clean cancel and no-unwind boundary PASS");
}
