#pragma once

// Runtime-neutral binding of a dedicated, independently registered memfd.
// The CPU view is used only for the existing finite/checksum admission; the
// device alias and ready word are resolved by this worker's own runtime.
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace strata::heterogeneous {
struct DirectVerifiedBinding {
    const uint8_t* transport = nullptr;
    float* device_payload = nullptr;
    uint32_t* host_ready = nullptr;
    uint32_t* device_ready = nullptr;
    size_t capacity = 0;

    explicit operator bool() const noexcept {
        return transport && device_payload && host_ready && device_ready && capacity &&
               capacity % sizeof(float) == 0;
    }
    bool matches(const uint8_t* data, size_t bytes) const noexcept {
        return static_cast<bool>(*this) && data == transport && bytes &&
               bytes <= capacity && bytes % sizeof(float) == 0;
    }
    // Only called after stream drain, or before the next producer starts;
    // never read the word to replace a stream-completion proof.
    bool published() const noexcept {
        if (!static_cast<bool>(*this)) return false;
        const auto ready = *static_cast<const volatile uint32_t*>(host_ready);
        std::atomic_thread_fence(std::memory_order_acquire);
        return ready == 1;
    }
};
} // namespace strata::heterogeneous
