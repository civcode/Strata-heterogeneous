#pragma once

#include "strata/core/transfer_terminal.hpp"
#include <array>
#include <cstddef>
#include <cuda_runtime.h>

namespace strata::core {

// A fixed-capacity engine-scope owner for resources created by pipelined
// serving. Construct before verifier/stage locals, so dependent GPU graphs
// unwind before release. Record the ACTUAL device at successful/partial create.
class Phase11bPipelineResources {
public:
    Phase11bPipelineResources() = default;
    Phase11bPipelineResources(const Phase11bPipelineResources&) = delete;
    Phase11bPipelineResources& operator=(const Phase11bPipelineResources&) = delete;
    ~Phase11bPipelineResources() noexcept {
        if (count_ == 0) return; // no runtime calls for an unused registry
        int previous = -1;
        check(cudaGetDevice(&previous), "pipeline cleanup query device");
        int current = previous;
        std::array<bool, kMaxDevices> completed{};
        // Fence ALL owners before releasing anything: work on one GPU may
        // still read another GPU's allocations after its local queue drains.
        for (std::size_t i = count_; i != 0; --i) {
            const Entry& e = entries_[i - 1];
            if (completed[static_cast<std::size_t>(e.device)]) continue;
            if (current != e.device) {
                check(cudaSetDevice(e.device), "pipeline cleanup select owner");
                current = e.device;
            }
            check(cudaDeviceSynchronize(), "pipeline cleanup completion");
            completed[static_cast<std::size_t>(e.device)] = true;
        }
        // Reverse creation order: refill event/stream, drafter events,
        // snapshots and side streams, rows, then stage streams.
        for (std::size_t i = count_; i != 0; --i) {
            const Entry& e = entries_[i - 1];
            if (current != e.device) {
                check(cudaSetDevice(e.device), "pipeline cleanup select owner");
                current = e.device;
            }
            switch (e.kind) {
                case Kind::buffer:
                    check(cudaFree(e.handle), "pipeline cleanup buffer");
                    break;
                case Kind::stream:
                    check(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(e.handle)), "pipeline cleanup stream");
                    break;
                case Kind::event:
                    check(cudaEventDestroy(reinterpret_cast<cudaEvent_t>(e.handle)), "pipeline cleanup event");
                    break;
            }
        }
        if (current != previous)
            check(cudaSetDevice(previous), "pipeline cleanup restore device");
    }

    void track_buffer(void* handle) noexcept { add(Kind::buffer, handle); }
    void track_stream(cudaStream_t handle) noexcept { add(Kind::stream, reinterpret_cast<void*>(handle)); }
    void track_event(cudaEvent_t handle) noexcept { add(Kind::event, reinterpret_cast<void*>(handle)); }

private:
    enum class Kind { buffer, stream, event };
    struct Entry { void* handle = nullptr; int device = -1; Kind kind = Kind::buffer; };
    static constexpr std::size_t kMaxEntries = 96;
    static constexpr std::size_t kMaxDevices = 64;
    std::array<Entry, kMaxEntries> entries_{};
    std::size_t count_ = 0;

    void add(Kind kind, void* handle) noexcept {
        if (handle == nullptr) return;
        if (count_ == kMaxEntries)
            transfer_terminal("pipeline resource registry", "capacity exhausted");
        int device = -1;
        check(cudaGetDevice(&device), "pipeline track resource device");
        if (device < 0 || static_cast<std::size_t>(device) >= kMaxDevices)
            transfer_terminal("pipeline track resource device", "invalid device ordinal");
        entries_[count_++] = {handle, device, kind};
    }
    static void check(cudaError_t status, const char* label) noexcept {
        if (status != cudaSuccess)
            transfer_terminal(label, cudaGetErrorString(status));
    }
};

} // namespace strata::core
