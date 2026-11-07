#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

namespace strata::core::detail {
// Serialized by RemoteExperts::read_mu_. GPU writes only this owner's arena, never
// borrowed caller storage. The cache owner must outlive unknown_completion().
class RemoteReadback {
public:
    RemoteReadback() = default;
    RemoteReadback(const RemoteReadback&) = delete;
    RemoteReadback& operator=(const RemoteReadback&) = delete;
    ~RemoteReadback() {
        // release() is called on the owning device after the owner's cleanup fence.
        if (data_) std::fprintf(stderr, "strata remote: retaining %zu readback bytes until process exit\n", capacity_);
    }
    cudaError_t copy(const uint8_t* const* src, uint8_t* const* dst, const size_t* bytes,
                     size_t n, cudaStream_t stream) {
        if (error_ != cudaSuccess) return error_;
        size_t total = 0;
        for (size_t i = 0; i < n; ++i) if (src[i] && dst[i]) {
            if (bytes[i] > std::numeric_limits<size_t>::max() - total) return cudaErrorInvalidValue;
            total += bytes[i];
        }
        if (!total) return cudaSuccess;
        if (total > capacity_) {
            void* fresh = nullptr;
            const auto e = cudaHostAlloc(&fresh, total, cudaHostAllocPortable);
            if (e != cudaSuccess) return e; // no submission, old idle arena remains valid
            if (data_) (void) cudaFreeHost(data_);
            data_ = static_cast<uint8_t*>(fresh);
            capacity_ = total;
        }
        size_t offset = 0;
        for (size_t i = 0; i < n; ++i) if (src[i] && dst[i]) {
            unknown_ = true; // even an unsuccessful API may have accepted work
            const auto e = cudaMemcpyAsync(data_ + offset, src[i], bytes[i], cudaMemcpyDeviceToHost, stream);
            if (e != cudaSuccess) {
                error_ = e;
                if (cudaStreamSynchronize(stream) == cudaSuccess) unknown_ = false;
                return error_;
            }
            offset += bytes[i];
        }
        error_ = cudaStreamSynchronize(stream);
        if (error_ != cudaSuccess) return error_;
        unknown_ = false;
        offset = 0;
        for (size_t i = 0; i < n; ++i) if (src[i] && dst[i]) {
            std::memcpy(dst[i], data_ + offset, bytes[i]);
            offset += bytes[i];
        }
        return cudaSuccess;
    }
    bool failed() const { return error_ != cudaSuccess; }
    bool unknown_completion() const { return unknown_; }
    size_t capacity() const { return capacity_; }
    // Only the owner can establish the device and a fence covering all submitted reads.
    void release_completed() {
        unknown_ = false;
        if (data_) (void) cudaFreeHost(data_);
        data_ = nullptr;
        capacity_ = 0;
        // Error is deliberately sticky: release is not context recovery.
    }
private:
    uint8_t* data_ = nullptr;
    size_t capacity_ = 0;
    cudaError_t error_ = cudaSuccess;
    bool unknown_ = false;
};
} // namespace strata::core::detail
