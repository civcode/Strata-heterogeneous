// Production ExpertCache startup/lifecycle tests. Synthetic errors follow valid
// work, never a device fault. Fork/exec occurs before HIP initialization.
#include "strata/core/expert_cache.hpp"
#include <hip/hip_runtime_api.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

namespace {
constexpr size_t N = 4096;
std::string mode, phase;
std::map<void*, bool> owners; // true = caller source/output; false = cache allocation
bool tracking = false, pending = false, device_done = false, open_failed = false;
int cache_device = 0, copies = 0, waits = 0, injections = 0, stage_allocs = 0, device_waits = 0;
void* readback = nullptr;
hipStream_t submitted = nullptr;
void require(bool yes, const char* why) { if (!yes) throw std::runtime_error(why); }
void injected() { ++injections; std::puts("INJECTED"); }
bool unknown() { return mode.find("unknown") != std::string::npos; }
void release(void* p) {
    const auto it = owners.find(p);
    if (it == owners.end()) return;
    int current = -1;
    require(hipGetDevice(&current) == hipSuccess, "release device");
    if (pending || (!it->second && (phase == "close" || open_failed) &&
                    (!device_done || current != cache_device))) {
        std::fprintf(stderr, "release phase=%s device_done=%d pending=%d\n", phase.c_str(), int(device_done), int(pending));
        std::puts("EARLY_RELEASE"); std::_Exit(2);
    }
    if (it->second || phase == "close" || open_failed) std::puts("OWNER_RELEASE");
    owners.erase(it);
}
void delete_readback(void* p) {
    if (p != readback || !readback) return;
    if (pending) { std::puts("EARLY_READBACK_DELETE"); std::_Exit(2); }
    std::puts("READBACK_RELEASE"); readback = nullptr;
}
}
extern "C" {
void __real__ZdlPv(void*) noexcept;
void __real__ZdlPvm(void*, size_t) noexcept;
void __wrap__ZdlPv(void* p) noexcept { delete_readback(p); __real__ZdlPv(p); }
void __wrap__ZdlPvm(void* p, size_t n) noexcept { delete_readback(p); __real__ZdlPvm(p, n); }
hipError_t __real_hipMalloc(void**, size_t);
hipError_t __real_hipHostMalloc(void**, size_t, unsigned);
hipError_t __real_hipFree(void*);
hipError_t __real_hipFreeHost(void*);
hipError_t __real_hipMemset(void*, int, size_t);
hipError_t __real_hipMemcpy(void*, const void*, size_t, hipMemcpyKind);
hipError_t __real_hipMemcpyAsync(void*, const void*, size_t, hipMemcpyKind, hipStream_t);
hipError_t __real_hipStreamSynchronize(hipStream_t);
hipError_t __real_hipDeviceSynchronize();
hipError_t __wrap_hipMalloc(void** p, size_t n) {
    if (tracking && mode == "open-alloc") { injected(); open_failed = true; return hipErrorOutOfMemory; }
    const auto e = __real_hipMalloc(p, n);
    if (tracking && e == hipSuccess) {
        require(hipGetDevice(&cache_device) == hipSuccess, "allocation device");
        owners[*p] = false;
    }
    return e;
}
hipError_t __wrap_hipHostMalloc(void** p, size_t n, unsigned f) {
    if (tracking) {
        ++stage_allocs;
        if ((mode == "open-stage" && stage_allocs == 1) || (mode == "open-sized-stage" && stage_allocs == 2)) {
            injected(); open_failed = true; return hipErrorOutOfMemory;
        }
    }
    const auto e = __real_hipHostMalloc(p, n, f);
    if (tracking && e == hipSuccess) owners[*p] = false;
    return e;
}
hipError_t __wrap_hipFree(void* p) { release(p); return __real_hipFree(p); }
hipError_t __wrap_hipFreeHost(void* p) { release(p); return __real_hipFreeHost(p); }
hipError_t __wrap_hipMemset(void* p, int value, size_t n) {
    const auto e = __real_hipMemset(p, value, n);
    if (tracking && e == hipSuccess && mode.find("open-memset") == 0) {
        injected(); open_failed = true; pending = true; return hipErrorUnknown;
    }
    return e;
}
hipError_t __wrap_hipMemcpy(void* d, const void* s, size_t n, hipMemcpyKind k) {
    if (phase == "operation" && ((k == hipMemcpyHostToDevice && mode.find("blocking-") == 0) ||
                                  (k == hipMemcpyDeviceToHost && mode.find("readback-") == 0))) {
        const auto e = __real_hipMemcpyAsync(d, s, n, k, nullptr);
        if (e != hipSuccess) return e;
        if (k == hipMemcpyDeviceToHost) readback = d; // actual production verification vector
        pending = true; submitted = nullptr; ++copies; injected(); return hipErrorUnknown;
    }
    return __real_hipMemcpy(d, s, n, k);
}
hipError_t __wrap_hipMemcpyAsync(void* d, const void* s, size_t n, hipMemcpyKind k, hipStream_t stream) {
    const auto e = __real_hipMemcpyAsync(d, s, n, k, stream);
    if (e == hipSuccess && phase == "operation") {
        ++copies; pending = true; submitted = stream;
        const bool fail = (mode.find("async-") == 0 || mode.find("queued-") == 0) ? copies == 1 :
                          mode.find("parts-") == 0 && copies == 2;
        if (fail && mode != "async-success") { injected(); return hipErrorUnknown; }
    }
    return e;
}
hipError_t __wrap_hipStreamSynchronize(hipStream_t stream) {
    const auto e = __real_hipStreamSynchronize(stream);
    if (e == hipSuccess && phase == "operation") {
        require(stream == submitted, "drained the wrong stream");
        ++waits;
        if (unknown() || ((mode == "sync-drain") && waits == 1)) { injected(); return hipErrorUnknown; }
        pending = false;
    }
    return e;
}
hipError_t __wrap_hipDeviceSynchronize() {
    const auto e = __real_hipDeviceSynchronize();
    if (e == hipSuccess && (phase == "close" || open_failed)) {
        ++device_waits;
        if (unknown()) { injected(); return hipErrorUnknown; }
        int device = -1; require(hipGetDevice(&device) == hipSuccess, "fence device");
        if (device == cache_device) { device_done = true; pending = false; }
        else {
            std::puts("WRONG_DEVICE_FENCE");
            // Diagnostic safety only: finish valid work on the owner device,
            // but do not count this extra fixture fence as production proof.
            require(hipSetDevice(cache_device) == hipSuccess, "diagnostic owner device");
            require(__real_hipDeviceSynchronize() == hipSuccess, "diagnostic safety fence");
            require(hipSetDevice(device) == hipSuccess, "diagnostic entry restore");
        }
    }
    return e;
}
}
namespace {
struct Caller {
    uint8_t *source = nullptr, *out = nullptr;
    Caller() {
        require(hipHostMalloc(&source, 2 * N, hipHostMallocPortable) == hipSuccess, "source owner");
        require(hipHostMalloc(&out, 2 * N, hipHostMallocPortable) == hipSuccess, "output owner");
        owners[source] = owners[out] = true;
        for (size_t i = 0; i < 2 * N; ++i) source[i] = uint8_t(i * 17 + i / 257 + 3);
        std::fill(out, out + 2 * N, 0);
    }
    ~Caller() { if (out) (void) hipHostFree(out); if (source) (void) hipHostFree(source); }
};
int child_run() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count < (mode == "device-scope" ? 2 : 1)) return 77;
    cache_device = mode == "device-scope" ? 1 : 0;
    require(hipSetDevice(cache_device) == hipSuccess, "cache device");
    Caller caller;
    strata::core::ExpertCache cache;
    std::string err;
    const auto close_checked = [&] {
        int entry = -1, restored = -1;
        require(hipGetDevice(&entry) == hipSuccess, "close entry device");
        phase = "close"; cache.close();
        require(hipGetDevice(&restored) == hipSuccess && restored == entry, "close restored caller device");
    };
    phase = "open"; tracking = true;
    const bool opened = mode == "open-sized-stage" ? cache.open_sized({N, 2 * N}, 1, 2, err) :
                                                     cache.open(2, 1, 2, N, err);
    tracking = false;
    if (mode.find("open-") == 0) {
        require(!opened && !cache.valid() && injections > 0, "open refusal / injection");
        for (const auto& item : owners) require(item.second, "failed open leaked cache allocation");
        require(mode == "open-alloc" || device_done, "open cleanup missed device fence");
        mode = "reopen"; open_failed = false;
        phase = "reopen"; tracking = true;
        require(cache.open(2, 1, 2, N, err), err.c_str()); tracking = false;
        require(cache.fill_slot_blocking(0, caller.source, err) && cache.verify_slot(0, caller.source, err), "reopen exact bytes");
    } else {
        require(opened && cache.valid(), err.c_str());
        require(__real_hipDeviceSynchronize() == hipSuccess, "initial zeroing");
        require(cache.admit(0, 0) == 0 && cache.slot_of(0, 0) == 0, "initial residency");
        require(cache.fill_slot_blocking(0, caller.source, err) && cache.verify_slot(0, caller.source, err), "startup exact bytes");
        copies = waits = 0; phase = "operation";
        if (mode == "success") {
            require(cache.fill_slot_queued(1, caller.source, err), err.c_str());
            require(waits == 0, "queued fill added a success wait");
            require(cache.sync_queued(err) && waits == 1, "one startup batch wait");
            require(cache.verify_slot(1, caller.source, err), "queued exact bytes");
        } else if (mode == "verify-null") {
            require(!cache.verify_slot(0, nullptr, err) && err.find("null") != std::string::npos && waits == 0, "null verify refusal");
        } else if (mode.find("blocking-") == 0) {
            require(!cache.fill_slot_blocking(1, caller.source, err), "blocking issue returned success");
            require(waits == 1 && !pending && cache.fills() == 1, "blocking failure did not drain / published fill");
        } else if (mode.find("readback-") == 0) {
            require(!cache.verify_slot(0, caller.source, err), "readback error returned success");
            require(waits == 1 && !pending && !readback, "verification vector not drained before destruction");
        } else if (mode.find("parts-") == 0) {
            const strata::core::BlobPart pieces[] = {{caller.source, N / 4, 0}, {caller.source + N / 4, N / 4, N / 4},
                                                    {caller.source + N / 2, N / 2, N / 2}};
            require(!cache.fill_slot_queued_parts(1, pieces, 3, err), "partial issue returned success");
            require(copies == 2 && waits == 1 && !pending && cache.fills() == 1, "partial upload did not drain");
            require(__real_hipMemcpy(caller.out, cache.device_slot(1), N, hipMemcpyDeviceToHost) == hipSuccess, "partial readback");
            for (size_t i = 0; i < N; ++i) require(caller.out[i] == (i < N / 2 ? caller.source[i] : 0), "partial upload bytes");
        } else if (mode.find("sync-") == 0 || mode.find("queued-") == 0) {
            const bool issued = cache.fill_slot_queued(1, caller.source, err);
            if (mode.find("sync-") == 0) {
                require(issued && !cache.sync_queued(err) && waits == 2 && !pending, "first sync error / second drain");
            } else require(!issued && waits == 1 && !pending && cache.fills() == 1, "queued issue drain");
        } else {
            hipStream_t stream = nullptr;
            require(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) == hipSuccess, "nonblocking stream");
            if (mode.find("async-") == 0) {
                const bool ok = cache.fill_slot(1, caller.source, stream, err);
                if (mode == "async-success") {
                    require(ok && waits == 0, "async success acquired a wait");
                    require(hipStreamSynchronize(stream) == hipSuccess, "caller orders async success");
                } else require(!ok && waits == 1 && !pending, "async issue drain");
            } else {
                if (mode == "graph-reader") {
                    hipGraph_t graph = nullptr; hipGraphExec_t exec = nullptr; hipGraphNode_t node = nullptr;
                    require(hipGraphCreate(&graph, 0) == hipSuccess, "graph create");
                    require(hipGraphAddMemcpyNode1D(&node, graph, nullptr, 0, caller.out, cache.device_slot(0), N,
                                                  hipMemcpyDeviceToHost) == hipSuccess, "graph cache reader");
                    require(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0) == hipSuccess, "graph instantiate");
                    require(hipGraphLaunch(exec, stream) == hipSuccess, "graph launch"); pending = true;
                    close_checked();
                    require(!pending && device_done && std::equal(caller.out, caller.out + N, caller.source), "close drained active graph reader");
                    require(hipGraphExecDestroy(exec) == hipSuccess && hipGraphDestroy(graph) == hipSuccess, "graph cleanup");
                } else {
                    require(hipMemcpyAsync(caller.out, cache.device_slot(0), N, hipMemcpyDeviceToHost, stream) == hipSuccess, "cache reader");
                    if (mode == "device-scope") require(hipSetDevice(0) == hipSuccess, "restore caller device");
                    close_checked();
                    require(!pending && device_done && std::equal(caller.out, caller.out + N, caller.source), "close drained reader");
                    if (mode == "device-scope") std::puts("DEVICE_SCOPE_PASS");
                }
            }
            require(hipStreamDestroy(stream) == hipSuccess, "stream cleanup");
        }
        if (injections) require(!err.empty() || mode == "close-unknown", "original error not returned");
    }
    close_checked(); close_checked();
    require(!cache.valid() && device_done, "closed cache");
    // Every nonfatal path must remain reopenable and preserve variable-slot bytes.
    mode = "reopen"; phase = "reopen"; tracking = true;
    require(cache.open_sized({N, 2 * N}, 1, 2, err), err.c_str()); tracking = false;
    require(cache.fill_slot_blocking(1, caller.source, err, 2 * N) && cache.verify_slot(1, caller.source, err, 2 * N), "sized reopen exact bytes");
    device_done = false; close_checked();
    for (const auto& item : owners) require(item.second, "cache allocation leaked");
    std::puts("CACHE_PASS"); return 0;
}
}
int main(int argc, char** argv) {
    try {
        require(argc >= 2, "scenario required"); mode = argv[1];
        if (argc == 3 && std::string(argv[2]) == "child") return child_run();
        int fds[2]; require(pipe(fds) == 0, "pipe"); const pid_t pid = fork(); require(pid >= 0, "fork");
        if (!pid) {
            close(fds[0]); if (dup2(fds[1], STDOUT_FILENO) < 0) std::_Exit(3); close(fds[1]);
            execl(argv[0], argv[0], argv[1], "child", (char*) nullptr); std::_Exit(3);
        }
        close(fds[1]); std::string output; std::array<char, 1024> buf{}; ssize_t n;
        while ((n = read(fds[0], buf.data(), buf.size())) > 0) output.append(buf.data(), size_t(n));
        close(fds[0]); int status = 0; require(waitpid(pid, &status, 0) == pid && WIFEXITED(status), "child exit");
        std::fputs(output.c_str(), stdout);
        if (WEXITSTATUS(status) == 77) return 77;
        require(WEXITSTATUS(status) == (unknown() ? 1 : 0), "unexpected child status");
        if (unknown()) require(output.find("INJECTED") != std::string::npos && output.find("ERR ") != std::string::npos &&
                               output.find("RELEASE") == std::string::npos && output.find("CACHE_PASS") == std::string::npos,
                               "unknown completion unwound an owner");
        else require(output.find("CACHE_PASS") != std::string::npos && output.find("EARLY_") == std::string::npos, "cache path incomplete");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 2; }
}
