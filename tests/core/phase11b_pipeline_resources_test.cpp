// CPU-only fake HIP/CUDA runtime: mixed-device pipeline/refill ownership.
#include "strata/core/phase11b_pipeline_resources.hpp"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace {
int device = 0, gets = 0, switches = 0, syncs[3] = {}, released[3] = {}, failed = 0;
bool completed[3] = {};
int sequence[20] = {}, nseq = 0;
void require(bool v, const char* why) {
    if (!v) { std::fprintf(stderr, "ASSERTION_FAILED %s\n", why); std::_Exit(98); }
}
struct Dependent {
    ~Dependent() { require(nseq == 0, "dependent released after registered handles"); }
};
void* handle(int d, int slot) {
    return reinterpret_cast<void*>(static_cast<uintptr_t>(0x1000 + d * 0x100 + slot));
}
int dev_of(void* p) {
    return static_cast<int>((reinterpret_cast<uintptr_t>(p) - 0x1000) / 0x100);
}
void release(void* p, int kind) {
    const int owner = dev_of(p);
    require(owner == device && completed[owner], "wrong owner or missing completion fence");
    require(nseq < 20, "release sequence overflow");
    sequence[nseq++] = kind;
    ++released[kind];
    std::printf("RELEASE %d owner=%d %p\n", kind, owner, p);
}
}
extern "C" cudaError_t cudaGetDevice(int* d) {
    if (failed == 1) return cudaErrorUnknown;
    ++gets; *d = device; return cudaSuccess;
}
extern "C" cudaError_t cudaSetDevice(int d) {
    if (failed == 2 || (failed == 7 && d == 0)) return cudaErrorUnknown;
    ++switches; device = d; return cudaSuccess;
}
extern "C" cudaError_t cudaDeviceSynchronize(void) {
    if (failed == 3) return cudaErrorUnknown;
    ++syncs[device]; completed[device] = true; return cudaSuccess;
}
extern "C" cudaError_t cudaFree(void* p) {
    if (failed == 4) return cudaErrorUnknown;
    release(p, 0); return cudaSuccess;
}
extern "C" cudaError_t cudaStreamDestroy(cudaStream_t p) {
    if (failed == 5) return cudaErrorUnknown;
    release(reinterpret_cast<void*>(p), 1); return cudaSuccess;
}
extern "C" cudaError_t cudaEventDestroy(cudaEvent_t p) {
    if (failed == 6) return cudaErrorUnknown;
    release(reinterpret_cast<void*>(p), 2); return cudaSuccess;
}
extern "C" const char* cudaGetErrorString(cudaError_t e) {
    return e == cudaSuccess ? "success" : "test injected failure";
}
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const std::string mode = argv[1];
    if (mode == "fail-track") failed = 1;
    if (mode == "fail-set") failed = 2;
    if (mode == "fail-sync") failed = 3;
    if (mode == "fail-free") failed = 4;
    if (mode == "fail-stream") failed = 5;
    if (mode == "fail-event") failed = 6;
    if (mode == "fail-restore") failed = 7;
    {
        strata::core::Phase11bPipelineResources p;
        Dependent dependent;
        if (mode == "fail-overflow") {
            for (int i = 0; i < 97; ++i) p.track_buffer(handle(0, i));
        } else if (mode == "fail-invalid") {
            device = 64;
            p.track_buffer(handle(0, 1));
        } else if (mode == "empty") {
            // Nothing registered; no runtime calls on destruction.
        } else if (mode == "partial" || mode == "fail-restore") {
            device = 1; p.track_buffer(handle(1, 1)); device = 0;
        } else {
            p.track_stream(reinterpret_cast<cudaStream_t>(handle(0, 1)));
            device = 1;
            p.track_stream(reinterpret_cast<cudaStream_t>(handle(1, 2)));
            p.track_buffer(handle(1, 3));
            device = 0;
            p.track_buffer(handle(0, 4));
            p.track_event(reinterpret_cast<cudaEvent_t>(handle(0, 5)));
            device = 1;
            p.track_event(reinterpret_cast<cudaEvent_t>(handle(1, 6)));
            device = 0;
            if (mode == "fail-get") failed = 1; // destructor query, after registering
        }
    }
    if (mode == "empty")
        require(gets == 0 && switches == 0 && !nseq, "empty registry touched runtime");
    else if (mode == "partial")
        require(syncs[1] == 1 && released[0] == 1 && device == 0, "partial owner cleanup");
    else {
        require(released[0] == 2 && released[1] == 2 && released[2] == 2, "all handles");
        require(syncs[0] == 1 && syncs[1] == 1, "fence once per device");
        require(device == 0, "caller device not restored");
        require(sequence[0] == 2 && sequence[1] == 2 &&
                sequence[2] == 0 && sequence[3] == 0 &&
                sequence[4] == 1 && sequence[5] == 1, "reverse creation order");
    }
    std::puts("PASSED");
    return 0;
}
