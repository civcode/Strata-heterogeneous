#pragma once
// Host-side publication into HIP-owned coherent mapped memory. No runtime
// pointers are passed across the vendor boundary. The caller has admitted
// the exact window and prelaunched a native bounded wait on ready.
#include "strata/platform/cpu_relax.hpp"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace strata::heterogeneous {
inline bool publish_receiver_input(void* mapped, size_t capacity, volatile uint32_t* ready,
                                   const void* payload, size_t bytes) noexcept {
    if (!mapped || !ready || !payload || !bytes || bytes > capacity || *ready != 0)
        return false;
    std::memcpy(mapped, payload, bytes);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    strata_store_fence();
    *ready = 1;
    strata_store_fence();
    return true;
}
} // namespace strata::heterogeneous
