#pragma once

#include <cstdint>

namespace strata::heterogeneous {

// Canonical Q2_0: 48 layers, 512 routed experts per layer.
constexpr uint64_t canonical_stage_expert_slots(uint32_t first, uint32_t end) noexcept {
    return first < end && end <= 48 ? uint64_t(end - first) * 512u : 0;
}
constexpr bool valid_native_expert_slots(uint32_t first, uint32_t end,
                                         uint32_t requested) noexcept {
    return requested > 0 && requested <= canonical_stage_expert_slots(first, end);
}

} // namespace strata::heterogeneous
