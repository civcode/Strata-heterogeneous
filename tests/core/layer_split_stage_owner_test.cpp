// CPU-only fake-CUDA regression for GpuStage's custom unique_ptr deleter.
// Each case runs in its own process because uncertain GPU completion must
// exit without deleting the stage's live resource-owning subobjects.
#include "strata/core/layer_split_stage_owner.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

namespace {
int current_device = 0;
int requested_owner = 1;
bool fail_get = false, fail_owner_select = false, fail_sync = false, fail_restore = false;
bool synchronized_owner = false;
int owner_switches = 0, restore_switches = 0, syncs = 0, releases = 0;

void require(bool ok, const char* msg) {
    if (!ok) { std::fprintf(stderr, "ASSERTION_FAILED %s\n", msg); std::exit(20); }
}

struct DeviceResource {
    int device = 1;
    bool allocated = true;
    ~DeviceResource() {
        if (!allocated) return;
        // A normal unique_ptr<GpuStage> deletes this with primary 0 current.
        if (current_device != device || !synchronized_owner) {
            std::fprintf(stderr, "WRONG_DEVICE_RELEASE current=%d owner=%d sync=%d\n",
                         current_device, device, int(synchronized_owner));
            std::fflush(stderr);
            std::_Exit(91); // expected failure of the old ownership policy
        }
        ++releases;
        std::printf("MEMBER_RELEASE device=%d\n", current_device);
    }
};
struct Stage {
    int dev = requested_owner;
    bool cleanup_armed = true;
    DeviceResource resource;
    explicit Stage(bool armed = true) : cleanup_armed(armed), resource{dev, armed} {}
};
} // namespace

extern "C" cudaError_t cudaGetDevice(int* device) {
    std::puts("DEVICE_QUERY");
    if (fail_get) return cudaErrorUnknown;
    *device = current_device;
    return cudaSuccess;
}
extern "C" cudaError_t cudaSetDevice(int device) {
    std::printf("DEVICE_SELECT %d\n", device);
    if ((device == requested_owner && fail_owner_select) ||
        (device == 0 && fail_restore)) return cudaErrorUnknown;
    current_device = device;
    if (device == requested_owner) ++owner_switches;
    else ++restore_switches;
    return cudaSuccess;
}
extern "C" cudaError_t cudaDeviceSynchronize(void) {
    std::printf("DEVICE_SYNC %d\n", current_device);
    ++syncs;
    if (fail_sync) return cudaErrorUnknown;
    if (current_device == requested_owner) synchronized_owner = true;
    return cudaSuccess;
}
extern "C" const char* cudaGetErrorString(cudaError_t status) {
    return status == cudaSuccess ? "success" : "injected unknown";
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const std::string mode = argv[1];
    if (mode == "owner-same") current_device = requested_owner;
    if (mode == "get-fail") fail_get = true;
    if (mode == "select-fail") fail_owner_select = true;
    if (mode == "sync-fail") fail_sync = true;
    if (mode == "restore-fail") fail_restore = true;
    if (mode == "missing-owner") requested_owner = -1;

    if (mode == "negative-plain") {
        // Control: old GpuStage storage was a default-deleted unique_ptr.
        std::unique_ptr<Stage> old(new Stage());
        old.reset();  // DeviceResource fails with 91: *no* correct owner fence
        require(false, "old default deletion unexpectedly succeeded");
    } else {
        using Owned = std::unique_ptr<Stage, strata::core::LayerSplitStageDelete<Stage>>;
        const bool armed = mode != "no-resources";
        Owned stage(new Stage(armed));
        stage.reset();
    }
    if (mode == "owner-switch") {
        require(current_device == 0 && syncs == 1 && owner_switches == 1 && restore_switches == 1 &&
                releases == 1, "switch, completion, members, and restore ordering");
    } else if (mode == "owner-same") {
        require(current_device == 1 && syncs == 1 && owner_switches == 0 && restore_switches == 0 &&
                releases == 1, "already-selected owner is preserved");
    } else if (mode == "no-resources") {
        require(syncs == 0 && owner_switches == 0 && restore_switches == 0 && releases == 0,
                "empty preflight refusal must not select or synchronize a GPU");
    } else {
        require(false, "terminal failure incorrectly returned to caller");
    }
    std::puts("PASSED");
    return 0;
}
