#pragma once

#include "strata/core/transfer_terminal.hpp"
#include <cuda_runtime.h>

namespace strata::core {

// A GPU buffer kept alive through later-stage member destructors. Hold with
// LayerSplitStageDelete so the owning GPU is synchronized and selected first.
// The buffer itself makes no device switches and never aliases stage.raw.
template<class T>
struct DeviceOwnedBuffer {
    int dev = -1;
    bool cleanup_armed = false;
    T* ptr = nullptr;
    DeviceOwnedBuffer() = default;
    DeviceOwnedBuffer(const DeviceOwnedBuffer&) = delete;
    DeviceOwnedBuffer& operator=(const DeviceOwnedBuffer&) = delete;
    ~DeviceOwnedBuffer() noexcept {
        if (ptr) {
            const cudaError_t e = cudaFree(ptr);
            if (e != cudaSuccess)
                transfer_terminal("device-owned buffer release", cudaGetErrorString(e));
        }
    }
};

} // namespace strata::core
