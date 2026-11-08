#pragma once

#include "strata/core/transfer_terminal.hpp"
#include <cuda_runtime.h>

namespace strata::core::detail {

// The peer copies sometimes borrow caller-owned mapped x/out and ExpertSource
// blobs. An issue error does not prove that the DMA was rejected: drain before
// returning, and never unwind owners if completion remains unknown.
inline void peer_check(cudaError_t status, const char* phase) noexcept {
    if (status != cudaSuccess) core::transfer_terminal(phase, cudaGetErrorString(status));
}

inline void peer_drain_or_exit(cudaStream_t stream, const char* phase) noexcept {
    peer_check(cudaStreamSynchronize(stream), phase);
}

inline bool peer_event_ready(cudaEvent_t event, const char* phase) noexcept {
    const cudaError_t status = cudaEventQuery(event);
    if (status == cudaSuccess) return true;
    if (status == cudaErrorNotReady) return false;
    peer_check(status, phase);
    return false;
}

// Unlike the old best-effort On scope, a failed device selection or restoration
// cannot silently run teardown on the wrong device. Use only on ownership paths.
struct PeerDeviceScope {
    int previous = -1;
    explicit PeerDeviceScope(int device) noexcept {
        peer_check(cudaGetDevice(&previous), "peer experts get device");
        peer_check(cudaSetDevice(device), "peer experts select device");
    }
    ~PeerDeviceScope() noexcept {
        peer_check(cudaSetDevice(previous), "peer experts restore device");
    }
    PeerDeviceScope(const PeerDeviceScope&) = delete;
    PeerDeviceScope& operator=(const PeerDeviceScope&) = delete;
};

} // namespace strata::core::detail
