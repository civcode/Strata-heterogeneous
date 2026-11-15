#pragma once

#include "strata_heterogeneous/direct_handoff_registration.hpp"
#include "strata_heterogeneous/direct_verified_binding.hpp"
#include "strata_heterogeneous/shared.hpp"

namespace strata::heterogeneous {
inline DirectVerifiedBinding register_verified_slot(DirectHandoffRegistration& registration,
        const SharedActivation& slot, int ordinal, const Plan& plan) {
    if (!slot.valid() || !direct_verified_plan(plan))
        throw std::invalid_argument("invalid direct verified shared slot or plan");
    const auto layout = DirectHandoffLayout::make(plan.geometry.hc, plan.geometry.n_embd, 8);
    registration.open_slot(slot.descriptor(), ordinal, layout, SharedActivation::capacity);
    const auto host = registration.host();
    const auto alias = registration.local_alias(ordinal);
    return {slot.data(), alias.payload(), host.ready(), alias.ready(), layout.payload_bytes};
}
} // namespace strata::heterogeneous
