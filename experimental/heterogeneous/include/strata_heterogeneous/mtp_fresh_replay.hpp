#pragma once
#include <string_view>

namespace strata::heterogeneous {
inline constexpr unsigned mtp_fresh_replay_max = 12;

// Only explicit, isolated fresh replay can exceed the old eight-window cap.
constexpr bool mtp_probe_window_count(unsigned windows, unsigned relay, bool fresh) noexcept {
    return windows > 0 && (fresh ? relay == 0 && windows <= mtp_fresh_replay_max : windows <= 8);
}
constexpr bool mtp_fresh_replay_shape(unsigned rows, unsigned keep, bool capture, bool repeat) noexcept {
    return rows == 1 && keep == 1 && capture && !repeat;
}
// Worker environment values are canonical decimal strings, not arbitrary atoi input.
constexpr unsigned mtp_probe_parse_windows(std::string_view value) noexcept {
    if (value.empty() || value.size() > 2 || value.front() == '0') return 0;
    unsigned n = 0;
    for (char c : value) {
        if (c < '0' || c > '9') return 0;
        n = n * 10 + unsigned(c - '0');
    }
    return n <= mtp_fresh_replay_max ? n : 0;
}
} // namespace strata::heterogeneous
