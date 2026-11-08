// Fake-runtime proof that raw later-stage resources are destroyed only after
// dependent member destructors, under the owner device fence.
#include "strata/core/layer_split_stage_owner.hpp"
#include "strata/core/layer_split_stage_resources.hpp"
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

namespace {
int current = 0, syncs = 0, frees = 0, destroys = 0, failure = 0;
void require(bool yes, const char* why) {
    if (!yes) { std::fprintf(stderr, "FAILED %s\n", why); std::exit(22); }
}
struct Dependent {
    ~Dependent() {
        require(frees == 0 && destroys == 0, "dependent destroyed after raw resources");
    }
};
struct Stage {
    int dev = 1;
    bool cleanup_armed = true;
    strata::core::LayerSplitStageResources raw; // destroyed after dependent
    Dependent dependent;
};
} // namespace

extern "C" cudaError_t cudaGetDevice(int* dev) { *dev = current; return cudaSuccess; }
extern "C" cudaError_t cudaSetDevice(int dev) { current = dev; return cudaSuccess; }
extern "C" cudaError_t cudaDeviceSynchronize(void) {
    require(current == 1, "synchronized wrong GPU"); ++syncs; return cudaSuccess;
}
extern "C" const char* cudaGetErrorString(cudaError_t error) {
    return error == cudaSuccess ? "success" : "injected failure";
}
extern "C" cudaError_t cudaFree(void* pointer) {
    require(current == 1 && syncs == 1, "free before owner fence");
    if (failure == 1) return cudaErrorUnknown;
    ++frees; std::printf("FREE %p\n", pointer); return cudaSuccess;
}
extern "C" cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    require(current == 1 && syncs == 1, "stream release before owner fence");
    if (failure == 2) return cudaErrorUnknown;
    ++destroys; std::printf("STREAM %p\n", stream); return cudaSuccess;
}
extern "C" cudaError_t cudaEventDestroy(cudaEvent_t event) {
    require(current == 1 && syncs == 1, "event release before owner fence");
    if (failure == 3) return cudaErrorUnknown;
    ++destroys; std::printf("EVENT %p\n", event); return cudaSuccess;
}
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const std::string mode = argv[1];
    if (mode == "owner-same") current = 1;
    if (mode == "fail-free") failure = 1;
    if (mode == "fail-stream") failure = 2;
    if (mode == "fail-event") failure = 3;
    auto stage = std::unique_ptr<Stage, strata::core::LayerSplitStageDelete<Stage>>(new Stage);
    if (mode == "empty") stage->cleanup_armed = false;
    else {
        auto& r = stage->raw;
        r.weight_arena = reinterpret_cast<void*>(0x101);
        r.session_arena = reinterpret_cast<void*>(0x102);
        r.residency = reinterpret_cast<void*>(0x103);
        r.mrope = reinterpret_cast<void*>(0x104);
        r.stream = reinterpret_cast<void*>(0x201);
        r.adapt_stream = reinterpret_cast<void*>(0x202);
        r.adapt_event = reinterpret_cast<void*>(0x301);
    }
    stage.reset();
    if (mode == "empty") require(syncs == 0 && frees == 0 && destroys == 0, "empty stage");
    else require(syncs == 1 && frees == 4 && destroys == 3, "seven handles released");
    require(current == (mode == "owner-same" ? 1 : 0), "caller device not restored");
    std::puts("PASSED"); return 0;
}
