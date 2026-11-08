// CPU-only fake runtime tests for the primary early-resource owner.
#include "strata/core/layer_split_stage_owner.hpp"
#include "strata/core/primary_early_resources.hpp"
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

namespace {
int current = 0, syncs = 0, frees = 0, streams = 0, failure = 0;
bool owner_complete = false;
void require(bool ok, const char* reason) {
    if (!ok) { std::fprintf(stderr, "FAILED %s\n", reason); std::_Exit(99); }
}
struct Dependent {
    ~Dependent() { require(frees == 0 && streams == 0, "freed before dependent"); }
};
struct Stage : strata::core::PrimaryEarlyResources { Dependent dependent; };
} // namespace

extern "C" cudaError_t cudaGetDevice(int* p) {
    if (failure == 1) return cudaErrorUnknown;
    *p = current; return cudaSuccess;
}
extern "C" cudaError_t cudaSetDevice(int d) {
    if ((failure == 2 && d == 1) || (failure == 6 && d == 0)) return cudaErrorUnknown;
    current = d; return cudaSuccess;
}
extern "C" cudaError_t cudaDeviceSynchronize(void) {
    if (failure == 3) return cudaErrorUnknown;
    require(current == 1, "synchronized wrong GPU");
    ++syncs; owner_complete = true; return cudaSuccess;
}
extern "C" cudaError_t cudaFree(void* p) {
    require(p != nullptr && current == 1 && owner_complete, "free before owner fence");
    if (failure == 4) return cudaErrorUnknown;
    ++frees; std::printf("FREE %p\n", p); return cudaSuccess;
}
extern "C" cudaError_t cudaStreamDestroy(cudaStream_t p) {
    require(p != nullptr && current == 1 && owner_complete, "stream before owner fence");
    if (failure == 5) return cudaErrorUnknown;
    ++streams; std::printf("STREAM %p\n", p); return cudaSuccess;
}
extern "C" const char* cudaGetErrorString(cudaError_t e) {
    return e == cudaSuccess ? "success" : "injected error";
}
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const std::string mode = argv[1];
    if (mode == "get-fail") failure = 1;
    if (mode == "select-fail") failure = 2;
    if (mode == "sync-fail") failure = 3;
    if (mode == "free-fail") failure = 4;
    if (mode == "stream-fail") failure = 5;
    if (mode == "restore-fail") failure = 6;
    if (mode == "owner-same") current = 1;
    using Owner = std::unique_ptr<Stage, strata::core::LayerSplitStageDelete<Stage>>;
    Owner p(new Stage);
    p->dev = 1;
    if (mode != "empty") {
        p->cleanup_armed = true;
        p->weight_arena = reinterpret_cast<void*>(0x1001);
        p->vision_table = reinterpret_cast<void*>(0x1002);
        p->ple_embedding = reinterpret_cast<void*>(0x1003);
        p->ple_scratch = reinterpret_cast<void*>(0x1004);
        p->expert_partials = reinterpret_cast<void*>(0x1005);
        p->main_stream = reinterpret_cast<cudaStream_t>(0x2001);
    }
    p.reset();
    if (mode == "empty") require(syncs == 0 && frees == 0 && streams == 0, "empty owner touched GPU");
    else require(syncs == 1 && frees == 5 && streams == 1, "all six releases");
    require(current == (mode == "owner-same" ? 1 : 0), "caller GPU restored");
    std::puts("PASSED");
}
