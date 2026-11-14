#pragma once
// CUDA-local mapped egress only. The transport never registers these pages
// with HIP. Native Verifier owns the publish kernel and producer completion;
// this adapter owns mapped host storage and checks publication after drain.
#include <cuda_runtime.h>
#include "strata/platform/cpu_relax.hpp"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace strata::heterogeneous {
class CudaMappedEgress {
public:
    CudaMappedEgress() = default;
    CudaMappedEgress(const CudaMappedEgress&) = delete;
    CudaMappedEgress& operator=(const CudaMappedEgress&) = delete;
    // Release explicitly on the owning device AFTER graph destruction/drain.
    bool allocate(size_t bytes) noexcept {
        if (!bytes || bytes > 16u * 1024u * 1024u || host_ || flag_host_) return false;
        if (cudaHostAlloc(&host_, bytes, cudaHostAllocMapped) != cudaSuccess ||
            cudaHostGetDevicePointer(reinterpret_cast<void**>(&device_), host_, 0) != cudaSuccess ||
            cudaHostAlloc(reinterpret_cast<void**>(&flag_host_), 64, cudaHostAllocMapped) != cudaSuccess ||
            cudaHostGetDevicePointer(reinterpret_cast<void**>(&flag_device_), flag_host_, 0) != cudaSuccess)
            return false;
        std::memset(host_, 0, bytes);
        std::memset(flag_host_, 0, 64);
        capacity_ = bytes;
        return true;
    }
    float* device_payload() const noexcept { return device_; }
    uint32_t* host_flag() const noexcept { return flag_host_; }
    uint32_t* device_flag() const noexcept { return flag_device_; }
    bool reset() noexcept {
        if (!capacity_ || !flag_host_) return false;
        *(volatile uint32_t*)flag_host_ = 0;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        strata_store_fence();
        return true;
    }
    bool copy_completed(void* dst, size_t bytes) const noexcept {
        if (!dst || !bytes || bytes > capacity_ || !flag_host_ ||
            *(volatile const uint32_t*)flag_host_ != 1) return false;
        strata_load_fence();
        std::atomic_thread_fence(std::memory_order_acquire);
        std::memcpy(dst, host_, bytes);
        return true;
    }
    bool release() noexcept {
        // Do not cudaFree the local device aliases: they belong to host storage.
        if (flag_host_ && cudaFreeHost(flag_host_) != cudaSuccess) return false;
        flag_host_ = flag_device_ = nullptr;
        if (host_ && cudaFreeHost(host_) != cudaSuccess) return false;
        host_ = nullptr;
        device_ = nullptr;
        capacity_ = 0;
        return true;
    }
private:
    void* host_ = nullptr;
    float* device_ = nullptr;
    uint32_t *flag_host_ = nullptr, *flag_device_ = nullptr;
    size_t capacity_ = 0;
};
} // namespace strata::heterogeneous
