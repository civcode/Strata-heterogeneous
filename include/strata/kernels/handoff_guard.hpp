#pragma once

#include <cstdint>

// Guarded native publication ABI. The counter lives in producer device memory;
// only the ready word is shared across runtimes. At most 64 blocks contribute.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRATA_HANDOFF_HD __host__ __device__
#else
#define STRATA_HANDOFF_HD
#endif
namespace strata::kernels::handoff_guard {
inline constexpr uint32_t pending = 0;
inline constexpr uint32_t ready = 1;
inline constexpr uint32_t nonfinite = 2;
inline constexpr uint32_t aborted = UINT32_MAX;
inline constexpr uint32_t rejected = 1; // consumer-local error word
inline constexpr uint32_t block_mask = 0xffffu;
inline constexpr uint32_t bad_block = 0x10000u;
inline constexpr uint32_t max_blocks = 64;
static_assert(max_blocks < block_mask && max_blocks < bad_block);

// Check the exponent bits, not a floating point comparison that fast-math
// could optimize away. Sign, subnormals and signed zero are immaterial.
STRATA_HANDOFF_HD constexpr bool finite_f32_bits(uint32_t bits) noexcept {
    return (bits & 0x7f800000u) != 0x7f800000u;
}
STRATA_HANDOFF_HD constexpr uint32_t contribution(bool bad) noexcept {
    return 1u + (bad ? bad_block : 0u);
}
STRATA_HANDOFF_HD constexpr bool last_block(uint32_t previous, uint32_t blocks) noexcept {
    return (previous & block_mask) + 1u == blocks;
}
STRATA_HANDOFF_HD constexpr uint32_t verdict(uint32_t previous, bool bad) noexcept {
    return ((previous >> 16u) || bad) ? nonfinite : ready;
}
STRATA_HANDOFF_HD constexpr bool accepted(uint32_t word) noexcept {
    return word == ready;
}
} // namespace strata::kernels::handoff_guard
#undef STRATA_HANDOFF_HD
