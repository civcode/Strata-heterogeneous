#pragma once

// Host-only Linux memfd handoff. One mutable, MAP_SHARED slot is passed to
// each exec-isolated worker with SCM_RIGHTS over its existing local socket.
// No pinned GPU buffers, CUDA/HIP pointers, or cross-runtime events involved.
#include "strata_heterogeneous/host.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::heterogeneous {

class SharedActivation {
public:
    SharedActivation() = default;
    ~SharedActivation() noexcept;
    SharedActivation(const SharedActivation&) = delete;
    SharedActivation& operator=(const SharedActivation&) = delete;
    SharedActivation(SharedActivation&& other) noexcept;
    SharedActivation& operator=(SharedActivation&& other) noexcept;

    static SharedActivation create();
    static SharedActivation receive(int socket_fd, int timeout_ms = default_io_timeout_ms);
    void send(int socket_fd, int timeout_ms = default_io_timeout_ms) const;

    static constexpr size_t capacity = max_payload_bytes;
    // All data is local, CPU-visible and explicitly staged. Do not interpret
    // the mapped address as CUDA/HIP-accessible memory.
    void store(const std::vector<float>& values);
    std::vector<float> load(size_t count) const;
    uint32_t checksum(size_t bytes) const;
    uint8_t* data() { return bytes_; }
    const uint8_t* data() const { return bytes_; }
    bool valid() const { return fd_ >= 0 && bytes_ != nullptr; }

private:
    int fd_ = -1;
    uint8_t* bytes_ = nullptr;
    explicit SharedActivation(int fd);
    void release() noexcept;
};

} // namespace strata::heterogeneous
