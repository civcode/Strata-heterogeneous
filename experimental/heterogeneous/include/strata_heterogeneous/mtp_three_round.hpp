#pragma once
#include <array>
#include <cstdint>
#include <cstddef>
#include "mtp_fresh_replay.hpp"

namespace strata::heterogeneous {
inline constexpr std::array<int32_t,7> mtp_three_prompt{248045,846,198,2523,513,32767,264};
inline constexpr unsigned mtp_diagnostic_rounds=3;
inline constexpr unsigned mtp_three_provisional_max=mtp_fresh_replay_max+4;
constexpr bool mtp_three_shape(unsigned windows,unsigned rounds,bool fresh,bool repeat,bool reprime,bool variable=false) noexcept {
    return (variable ? windows>=1 && windows<=mtp_three_prompt.size() : windows==mtp_three_prompt.size()) && rounds==mtp_diagnostic_rounds && !fresh && !repeat && !reprime;
}
// Keep the full greedy prefix or refuse the transaction. Reserve one anchor
// for each remaining round; never shorten a target-authoritative acceptance.
constexpr bool mtp_three_commit_fits(int64_t pos,unsigned keep,unsigned round,unsigned prompt_windows=mtp_three_prompt.size()) noexcept {
    return prompt_windows>=1 && prompt_windows<=mtp_three_prompt.size() &&
           round<mtp_diagnostic_rounds && pos>=int64_t(prompt_windows+round) &&
           pos<=int64_t(mtp_fresh_replay_max) && keep>=1 && keep<=5 &&
           pos+keep+(mtp_diagnostic_rounds-1-round)<=mtp_fresh_replay_max;
}
constexpr bool mtp_three_window_fits(int64_t pos,std::size_t rows,unsigned round,unsigned prompt_windows=mtp_three_prompt.size()) noexcept {
    return rows>=2 && rows<=5 && mtp_three_commit_fits(pos,1,round,prompt_windows) &&
           pos+int64_t(rows)<=mtp_three_provisional_max;
}
} // namespace strata::heterogeneous
