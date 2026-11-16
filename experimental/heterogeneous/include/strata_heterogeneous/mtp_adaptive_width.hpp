#pragma once

#include <algorithm>
#include <cstdint>

namespace strata::heterogeneous {

// Request-local controller for the existing native draft limit. No target
// state or token IDs are owned here. Only a committed offer supplies feedback.
// Two poor offers reduce the cap; two full accepted offers increase it.
// At zero, use native tail prefill (not draft with max_drafts=0) for two
// committed anchor-only rounds, then probe one candidate again.
class MtpAdaptiveWidth {
    unsigned width_ = 4, good_ = 0, bad_ = 0, cooldown_ = 0;
public:
    static constexpr unsigned max_width = 4;
    unsigned width() const noexcept { return width_; }
    unsigned limit(int64_t remaining_candidates) const noexcept {
        return remaining_candidates <= 0 ? 0u :
            static_cast<unsigned>(std::min<int64_t>(width_, remaining_candidates));
    }
    bool observe(unsigned budget, unsigned offered, unsigned accepted) noexcept {
        if (budget > width_ || offered > budget || accepted > offered) return false;
        // A short remaining request must not be interpreted as rejection.
        if (budget != width_) return true;
        if (width_ == 0) {
            if (++cooldown_ == 2) {
                width_ = 1;
                cooldown_ = good_ = bad_ = 0;
            }
            return true;
        }
        // Confidence/stop clipping may shorten the offer. Only submitted
        // candidates can fail verification, and only full-cap success grows it.
        if (offered == width_ && accepted == offered) {
            bad_ = 0;
            if (++good_ == 2) {
                if (width_ < max_width) ++width_;
                good_ = 0;
            }
        } else if (offered && 2 * accepted < offered) {
            good_ = 0;
            if (++bad_ == 2) {
                --width_;
                bad_ = cooldown_ = 0;
            }
        } else {
            good_ = bad_ = 0;
        }
        return true;
    }
};

} // namespace strata::heterogeneous
