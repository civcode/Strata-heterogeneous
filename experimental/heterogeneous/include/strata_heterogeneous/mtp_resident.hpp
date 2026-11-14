#pragma once
#include "mtp_generation.hpp"

namespace strata::heterogeneous {
// The resident diagnostic uses the already-negotiated generation policy for
// every request. The wire reset frame need not carry a new shape or budget.
inline constexpr uint64_t mtp_resident_max_requests=8;
inline constexpr bool mtp_resident_next_request(uint64_t previous,uint64_t next) noexcept {
    return previous<mtp_resident_max_requests && next==previous+1;
}
inline bool mtp_resident_shape(const MtpGenerationPolicy& policy,size_t prompt,unsigned budget) noexcept {
    return policy.valid() && prompt==policy.prompt_rows && budget==policy.max_new;
}
} // namespace strata::heterogeneous
