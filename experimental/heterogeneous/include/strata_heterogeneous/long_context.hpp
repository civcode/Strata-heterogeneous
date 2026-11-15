#pragma once
#include <cstddef>
#include <cstdint>

namespace strata::heterogeneous {
inline constexpr uint32_t legacy_workload_context = 32768;
inline constexpr uint32_t max_long_context = 262144;
inline constexpr uint32_t min_long_context = 2048;
inline constexpr uint32_t long_context_alignment = 256;
inline constexpr uint64_t long_context_min_reserve = 1ull << 30;
inline constexpr int long_prefill_timeout_ms = 1800000;
inline constexpr size_t long_context_command_bytes = 3 * 1024 * 1024;

constexpr bool long_context_cells_valid(uint64_t cells) noexcept {
    return cells >= min_long_context && cells <= max_long_context && cells % long_context_alignment == 0;
}
constexpr bool long_context_stage_valid(uint64_t cells, uint64_t reserve, uint64_t budget,
                                       uint64_t max_t, uint64_t slots, uint64_t layers) noexcept {
    return long_context_cells_valid(cells) && reserve >= long_context_min_reserve &&
           budget == 0 && max_t == 8 && layers > 0 && layers <= 48 && slots == layers * 512;
}
// Additional memory is estimated before allocation; subtraction avoids overflow.
constexpr bool capacity_fits(uint64_t free, uint64_t reserve, uint64_t additional) noexcept {
    return reserve <= free && additional <= free - reserve;
}
constexpr size_t resident_command_cap(uint32_t context, bool workload, bool extended) noexcept {
    return extended ? long_context_command_bytes : workload && context > 8192 ? 384 * 1024 : 65536;
}
} // namespace strata::heterogeneous
