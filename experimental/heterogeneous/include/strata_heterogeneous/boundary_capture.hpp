#pragma once

// Opt-in byte evidence from the real CUDA/HIP verifier transfer buffers.
// This is not an inference kernel and performs no floating-point arithmetic.
#include <cuda_runtime.h>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <unistd.h>
#include <vector>

namespace strata::heterogeneous {
inline bool capture_device_handoff(const void* device, const void* pinned,
                                   size_t bytes, size_t stage, uint64_t request,
                                   uint64_t window, const char* direction,
                                   std::string& error) {
    const char* prefix = std::getenv("STRATA_HETERO_BOUNDARY_CAPTURE_PREFIX");
    if (!prefix || !*prefix) return true;
    if (!device || !pinned || !bytes || bytes > 16u * 1024u * 1024u ||
        !direction || (std::strcmp(direction, "in") != 0 &&
                       std::strcmp(direction, "out") != 0)) {
        error = "boundary capture: invalid pointer, size or direction";
        return false;
    }

    // Caller has already waited for the production-stream fence. Readback
    // observes device bytes without changing what the verifier computes.
    std::vector<uint8_t> actual(bytes);
    const cudaError_t result = cudaMemcpy(actual.data(), device, bytes,
                                          cudaMemcpyDeviceToHost);
    if (result != cudaSuccess) {
        error = std::string("boundary capture: GPU readback failed: ") +
                cudaGetErrorString(result);
        return false;
    }
    if (std::memcmp(actual.data(), pinned, bytes) != 0) {
        error = "boundary capture: GPU readback differs from pinned transfer bytes";
        return false;
    }
    const std::string filename = std::string(prefix) + ".stage" +
        std::to_string(stage) + ".r" + std::to_string(request) + ".w" +
        std::to_string(window) + "." + direction + ".f32";
    const int fd = ::open(filename.c_str(), O_WRONLY | O_CREAT | O_EXCL |
                          O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd < 0) {
        error = std::string("boundary capture: exclusive open failed: ") +
                std::strerror(errno);
        return false;
    }
    size_t offset = 0;
    int failure = 0;
    while (offset < bytes) {
        const ssize_t n = ::write(fd, actual.data() + offset, bytes - offset);
        if (n > 0) offset += static_cast<size_t>(n);
        else if (n < 0 && errno == EINTR) continue;
        else { failure = n < 0 ? errno : EIO; break; }
    }
    if (::close(fd) != 0 && !failure) failure = errno;
    if (failure) {
        (void)::unlink(filename.c_str());
        error = std::string("boundary capture: write failed: ") +
                std::strerror(failure);
        return false;
    }
    std::fprintf(stderr, "STRATA_HANDOFF_BYTES stage=%zu request=%llu window=%llu "
                 "direction=%s bytes=%zu verified=device-vs-pinned\n",
                 stage, static_cast<unsigned long long>(request),
                 static_cast<unsigned long long>(window), direction, bytes);
    return true;
}
} // namespace strata::heterogeneous
