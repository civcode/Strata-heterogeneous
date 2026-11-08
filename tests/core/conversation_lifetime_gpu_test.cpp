// Valid native transfers + link wrappers reporting errors AFTER real accepted work. No device faults.
#include "conversation_copy.hpp"
#include <cstring>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <sys/wait.h>
#include <unistd.h>

#if defined(STRATA_USE_HIP)
using cudaMemcpyKind = hipMemcpyKind;
constexpr auto cudaErrorUnknown = hipErrorUnknown;
#define REAL(n) __real_hip##n
#define WRAP(n) __wrap_hip##n
#else
#define REAL(n) __real_cuda##n
#define WRAP(n) __wrap_cuda##n
#endif
namespace { std::string mode; bool armed = false; int injected = 0, drains = 0; }
extern "C" {
cudaError_t REAL(MemcpyAsync)(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t);
cudaError_t REAL(StreamSynchronize)(cudaStream_t);
cudaError_t REAL(EventRecord)(cudaEvent_t, cudaStream_t);
cudaError_t WRAP(MemcpyAsync)(void* d, const void* s, size_t n, cudaMemcpyKind kind, cudaStream_t st) {
    const auto e = REAL(MemcpyAsync)(d, s, n, kind, st);
    if (e == cudaSuccess && armed && mode.find("issue") != std::string::npos) { ++injected; return cudaErrorUnknown; }
    return e;
}
cudaError_t WRAP(StreamSynchronize)(cudaStream_t st) {
    const auto e = REAL(StreamSynchronize)(st);
    if (armed) ++drains;
    if (e == cudaSuccess && armed && mode.find("unknown") != std::string::npos) { ++injected; return cudaErrorUnknown; }
    return e;
}
cudaError_t WRAP(EventRecord)(cudaEvent_t ev, cudaStream_t st) {
    const auto e = REAL(EventRecord)(ev, st);
    if (e == cudaSuccess && armed && mode.find("record") != std::string::npos) { ++injected; return cudaErrorUnknown; }
    return e;
}
}
static void check(bool ok, const char* what) { if (!ok) { std::fprintf(stderr, "FAIL %s\n", what); std::exit(2); } }

int main(int argc, char** argv) {
    mode = argc > 1 ? argv[1] : "success";
    const bool child = argc > 2;
    if (mode.find("unknown") != std::string::npos && !child) {
        int fds[2]; check(pipe(fds) == 0, "pipe");
        const pid_t pid = fork(); check(pid >= 0, "fork");
        if (pid == 0) {
            close(fds[0]); const std::string fd = std::to_string(fds[1]);
            execl(argv[0], argv[0], mode.c_str(), fd.c_str(), static_cast<char*>(nullptr));
            std::_Exit(2);
        }
        close(fds[1]); int status = 0; check(waitpid(pid, &status, 0) == pid, "wait child");
        char b; const auto bytes = read(fds[0], &b, 1); close(fds[0]);
        if (WIFEXITED(status) && WEXITSTATUS(status) == 77) return 77;
        check(bytes == 0 && WIFEXITED(status) && WEXITSTATUS(status) == 1, "no owner destruction/resume before terminal exit");
        std::printf("%s native no-unwind PASS\n", mode.c_str()); return 0;
    }
    int count = 0;
    const auto initialized = cudaGetDeviceCount(&count);
    if (initialized != cudaSuccess || !count) { std::fprintf(stderr, "no GPU: %s\n", cudaGetErrorString(initialized)); return 77; }
    check(cudaSetDevice(0) == cudaSuccess, "select device");
    cudaDeviceProp prop{}; check(cudaGetDeviceProperties(&prop, 0) == cudaSuccess, "device identity");
    std::printf("device0 %s\n", prop.name);
    struct Owners {
        void *src = nullptr, *dst = nullptr, *a = nullptr, *b = nullptr;
        int fd = -1;
        ~Owners() {
            if (fd >= 0) { char x = 'F'; (void) write(fd, &x, 1); }
            if (src) cudaFreeHost(src);
            if (dst) cudaFreeHost(dst);
            if (a) cudaFree(a);
            if (b) cudaFree(b);
        }
    } owner;
    if (child) owner.fd = std::atoi(argv[2]);
    constexpr size_t n = 65537;
    check(cudaMallocHost(&owner.src, n) == cudaSuccess && cudaMallocHost(&owner.dst, n) == cudaSuccess &&
          cudaMalloc(&owner.a, n) == cudaSuccess && cudaMalloc(&owner.b, n) == cudaSuccess, "allocate operands");
    auto* src = static_cast<unsigned char*>(owner.src);
    for (size_t i = 0; i < n; ++i) src[i] = static_cast<unsigned char>(i * 37 + 19);
    using namespace strata::core::conversation_detail;
    cudaStream_t st = nullptr; check(private_stream(st) == cudaSuccess, "stream preflight");
    cudaEvent_t ev = nullptr; check(private_event(ev) == cudaSuccess, "event preflight");
    armed = true;
    cudaError_t e;
    if (mode.find("record") != std::string::npos) {
        check(copy_nonblocking(owner.a, src, n, true) == cudaSuccess, "queued H2D");
        e = order_after_queued();
    } else if (mode.find("readback") != std::string::npos) {
        armed = false; check(copy_nonblocking(owner.a, src, n) == cudaSuccess, "readback source"); armed = true;
        e = copy_nonblocking(owner.dst, owner.a, n, false, true);
    } else {
        e = copy_nonblocking(owner.a, src, n);
    }
    armed = false;
    check(e == (mode == "success" ? cudaSuccess : cudaErrorUnknown), "caller-visible result");
    if (mode != "success") check(injected > 0 && drains > 0, "injection and positive error drain");
    // Every recoverable return has established operand completion. Verify H2D + D2D + D2H bytes.
    check(copy_nonblocking(owner.b, owner.a, n) == cudaSuccess, "D2D");
    check(copy_nonblocking(owner.dst, owner.b, n, false, true) == cudaSuccess, "D2H");
    check(std::memcmp(owner.src, owner.dst, n) == 0, "exact transfer bytes");
    std::printf("%s injected=%d drains=%d exact H2D/D2D/D2H PASS\n", mode.c_str(), injected, drains);
}
