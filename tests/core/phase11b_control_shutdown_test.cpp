// CPU-only fake GPU regression for control-vector tables and the penalty history
// GPU buffer. The real HIP/CUDA runtime and model are not linked.
#include "strata/kernels/cvec_owner.hpp"
#include "strata/core/device_owned_buffer.hpp"
#include "strata/core/layer_split_stage_owner.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

namespace {
int current_device = 0;
int sync_count = 0;
int frees = 0;
int set_count = 0;
int fail = 0;
bool synchronized[4] = {};

void require(bool good, const char* message) {
    if (!good) {
        std::fprintf(stderr, "ASSERTION_FAILED %s\n", message);
        std::exit(50);
    }
}
struct Tables {
    float* dir = nullptr;
    float* s = nullptr;
    int* on = nullptr;
};
} // namespace

extern "C" cudaError_t cudaGetDevice(int* d) {
    if (fail == 1) return cudaErrorUnknown;
    *d = current_device;
    return cudaSuccess;
}
extern "C" cudaError_t cudaSetDevice(int d) {
    if (fail == 2) return cudaErrorUnknown;
    current_device = d;
    ++set_count;
    return cudaSuccess;
}
extern "C" cudaError_t cudaDeviceSynchronize(void) {
    if (fail == 3) return cudaErrorUnknown;
    ++sync_count;
    synchronized[current_device] = true;
    return cudaSuccess;
}
extern "C" cudaError_t cudaFree(void* p) {
    require(p != nullptr, "null free");
    require(synchronized[current_device], "free without owner completion");
    const auto address = reinterpret_cast<std::uintptr_t>(p);
    if (address == 0x101)
        require(current_device == 0 || sync_count == 1, "wrong GPU for primary table");
    else if (address >= 0x201 && address <= 0x204)
        require(current_device == 1, "wrong GPU for second-stage table/history");
    if (fail == 4) return cudaErrorUnknown;
    ++frees;
    std::printf("FREE device=%d address=%p\n", current_device, p);
    return cudaSuccess;
}
extern "C" const char* cudaGetErrorString(cudaError_t e) {
    return e == cudaSuccess ? "success" : "fake error";
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const std::string mode = argv[1];
    if (mode == "fail-get") fail = 1;
    if (mode == "fail-set") fail = 2;
    if (mode == "fail-sync") fail = 3;
    if (mode == "fail-free") fail = 4;

    if (mode == "tables-empty") {
        Tables t[3]{};
        strata::kernels::detail::cvec_release_devices(t);
        require(frees == 0 && sync_count == 0 && set_count == 0, "empty cleanup called GPU");
    } else if (mode == "tables" || mode.rfind("fail-", 0) == 0) {
        Tables t[3]{};
        t[0].dir = reinterpret_cast<float*>(0x101);
        t[1].dir = reinterpret_cast<float*>(0x202);
        t[1].s = reinterpret_cast<float*>(0x203);
        t[1].on = reinterpret_cast<int*>(0x204);
        strata::kernels::detail::cvec_release_devices(t);
        require(frees == 4 && current_device == 0 && sync_count == 2,
                "table owners/free count");
        require(!t[0].dir && !t[1].dir && !t[1].s && !t[1].on,
                "released entries must be cleared");
    } else if (mode == "partial") {
        current_device = 1;
        Tables t{reinterpret_cast<float*>(0x202), nullptr, reinterpret_cast<int*>(0x204)};
        strata::kernels::detail::cvec_release_here(t);
        require(frees == 2 && sync_count == 1 && !t.dir && !t.on,
                "partial table cleanup");
    } else if (mode == "buffer" || mode == "buffer-same") {
        if (mode == "buffer-same") current_device = 1;
        using Buffer = strata::core::DeviceOwnedBuffer<int>;
        std::unique_ptr<Buffer, strata::core::LayerSplitStageDelete<Buffer>> owned(new Buffer);
        owned->dev = 1;
        owned->cleanup_armed = true;
        owned->ptr = reinterpret_cast<int*>(0x201);
        owned.reset();
        require(frees == 1 && sync_count == 1 && current_device == (mode == "buffer" ? 0 : 1),
                "history buffer owner and previous device");
    } else {
        return 2;
    }
    std::puts("PASSED");
    return 0;
}
