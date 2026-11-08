// Production copy and restore gate, with deferred CPU DMA. No GPU fault is manufactured.
#include "conversation_copy.hpp"
#include "strata/program/conversation_restore.hpp"
#include <cstring>
#include <functional>
#include <future>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#if defined(__unix__)
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {
bool issue_error = false, wait_error = false, record_error = false, dependency_error = false, device_error = false;
int submits = 0, waits = 0;
std::vector<std::function<void()>> pending;
void require(bool ok) { if (!ok) { std::fputs("lifetime assertion failed\n", stderr); std::abort(); } }
}
namespace strata::core {
void conversation_checkpoint_join(const void*) {}
}
cudaError_t cudaGetDevice(int* d) { *d = 0; return device_error ? cudaErrorUnknown : cudaSuccess; }
cudaError_t cudaSetDevice(int) { return cudaSuccess; }
cudaError_t cudaGetLastError() { return cudaSuccess; }
const char* cudaGetErrorString(cudaError_t) { return "injected runtime error"; }
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* s, unsigned) { *s = reinterpret_cast<void*>(1); return cudaSuccess; }
cudaError_t cudaStreamSynchronize(cudaStream_t) {
    ++waits;
    if (wait_error) return cudaErrorUnknown;
    for (auto& f : pending) f();
    pending.clear(); return cudaSuccess;
}
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e, unsigned) { *e = reinterpret_cast<void*>(2); return cudaSuccess; }
cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t) { return record_error ? cudaErrorUnknown : cudaSuccess; }
cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned) { return dependency_error ? cudaErrorUnknown : cudaSuccess; }
cudaError_t cudaHostAlloc(void** p, size_t n, unsigned) { *p = std::malloc(n); return *p ? cudaSuccess : cudaErrorMemoryAllocation; }
cudaError_t cudaFreeHost(void* p) { std::free(p); return cudaSuccess; }
cudaError_t cudaMemcpyAsync(void* d, const void* s, size_t n, cudaMemcpyKind, cudaStream_t) {
    ++submits; pending.push_back([=] { std::memcpy(d, s, n); });
    return issue_error ? cudaErrorUnknown : cudaSuccess;
}
using namespace strata::core::conversation_detail;

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "issue-drain";
    issue_error = mode == "issue-drain" || mode == "issue-unknown" || mode == "queued-issue-drain" ||
                  mode == "readback-issue-drain";
    wait_error = mode == "issue-unknown" || mode == "wait-unknown" || mode == "readback-unknown";
    record_error = mode == "record-drain" || mode == "record-unknown";
    dependency_error = mode == "dependency-drain";
    if (mode == "record-unknown") wait_error = true;
    const bool terminal = wait_error || mode == "async-join-failed" ||
                          mode == "async-join-throws" || mode == "async-join-throws-unknown";
    auto run = [&] {
        if (mode.find("async-join") == 0) {
            auto worker = std::async(std::launch::async, [&]() -> cudaError_t {
                if (mode == "async-join-throws") throw std::runtime_error("injected worker exception");
                if (mode == "async-join-throws-unknown") throw 42;
                return terminal ? cudaErrorUnknown : cudaSuccess;
            });
            join_copy_or_exit(worker, "async image regression");
            require(!worker.valid());
            return;
        }
        std::vector<unsigned char> src(64, 0xa5), dst(64, 0);
        if (mode == "pre-submit") {
            device_error = true;
            require(copy_nonblocking(dst.data(), src.data(), src.size()) != cudaSuccess);
            require(submits == 0 && waits == 0 && dst.front() == 0);
            return;
        }
        if (record_error || dependency_error) {
            require(copy_nonblocking(dst.data(), src.data(), src.size(), true) == cudaSuccess);
            require(waits == 0 && !pending.empty());
            require(order_after_queued() != cudaSuccess);
            require(pending.empty() && dst == src && waits == 1);
            return;
        }
        // Exercise the real admission/mutation gate for whole, borrowed and borrowed->whole paths.
        for (int kind = 0; kind < 3; ++kind) {
            using namespace strata::core;
            ConversationCache cache{1 << 20, 4};
            SavedConversation image; image.live.ids = {1, 2}; image.live.gdn = src;
            cache.put(std::move(image));
            const auto selection = cache.best(std::vector<int32_t>{1, 2, 3}, {}, true);
            strata::program::ConversationRestoreAttempt attempt;
            std::string error;
            require(attempt.prepare(cache, selection.serial, kind != 0, nullptr, 0,
                [](const int*, size_t, std::string&) { return ConversationCopyPreparation::ready; }, error)
                    == ConversationCopyPreparation::ready);
            require(attempt.begin_mutation());
            if (kind == 2) require(attempt.take_donor(cache, selection.serial));
            std::fill(dst.begin(), dst.end(), 0);
            const auto result = attempt.apply([&] {
                return copy_nonblocking(dst.data(), src.data(), src.size(), mode == "queued-issue-drain",
                                        mode.find("readback") == 0) == cudaSuccess;
            });
            require(result == !issue_error);
            require(attempt.finish() == !issue_error);
            require(pending.empty());
#if defined(STRATA_USE_HIP)
            // HIP's landing arena deliberately does not publish failed readback bytes.
            if (mode == "readback-issue-drain") require(dst == std::vector<unsigned char>(64, 0));
            else
#endif
                require(dst == src);
            // Failed apply must suppress the next operation, not just return an error locally.
            int decoded = 0;
            (void) attempt.apply([&] { ++decoded; return true; });
            require(decoded == (issue_error ? 0 : 1));
        }
    };
#if defined(__unix__)
    if (terminal) {
        int fds[2]; require(pipe(fds) == 0);
        pid_t pid = fork(); require(pid >= 0);
        if (pid == 0) {
            close(fds[0]);
            struct Owners { int fd; ~Owners() { char x = 'F'; (void) write(fd, &x, 1); } } owners{fds[1]};
            run();
            char resumed = 'R'; (void) write(fds[1], &resumed, 1);
            std::exit(9);
        }
        close(fds[1]); int status = 0; require(waitpid(pid, &status, 0) == pid);
        char byte; require(read(fds[0], &byte, 1) == 0); close(fds[0]);
        require(WIFEXITED(status) && WEXITSTATUS(status) == 1);
    } else
#endif
        run();
    std::printf("conversation lifetime %s PASS\n", mode.c_str());
}
