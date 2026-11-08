#pragma once

#include "strata/core/transfer_terminal.hpp"
#include <cuda_runtime.h>

namespace strata::core::detail {

// Teardown-only: all mapped host operands may be read on the primary and helper
// devices. A per-helper stream wait does NOT fence primary-stream copy_rows/combine.
inline void remote_opt_check(cudaError_t status, const char* phase) noexcept {
    if (status != cudaSuccess) core::transfer_terminal(phase, cudaGetErrorString(status));
}

inline void remote_opt_select(int device, const char* phase) noexcept {
    remote_opt_check(cudaSetDevice(device), phase);
}

inline void remote_opt_complete(int device, const char* phase) noexcept {
    remote_opt_select(device, phase);
    remote_opt_check(cudaDeviceSynchronize(), phase);
}

} // namespace strata::core::detail
