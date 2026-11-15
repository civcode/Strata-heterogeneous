#pragma once

// Request-local policy for native greedy MTP serving. Independent of the
// fixed-shape qualification policy (which must remain unchanged).
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace strata::heterogeneous {
struct MtpRequestPolicy {
    static constexpr uint32_t initial_capacity = 2048;
    static constexpr uint32_t max_candidates = 4;
    uint32_t context = 0, max_new = 0, spec = 0, vocab = 0;
    std::vector<int32_t> prompt, stops;

    bool token(int32_t id) const noexcept { return id >= 0 && uint32_t(id) < vocab; }
    bool stop(int32_t id) const noexcept {
        return std::find(stops.begin(), stops.end(), id) != stops.end();
    }
    bool valid() const noexcept {
        return context == initial_capacity && vocab == 248320 &&
            max_new > 0 && max_new <= 256 && spec >= 1 && spec <= max_candidates &&
            !prompt.empty() && prompt.size() < context &&
            max_new <= context - prompt.size() && !stops.empty() && stops.size() <= 64 &&
            std::all_of(prompt.begin(), prompt.end(), [this](auto id){return token(id);}) &&
            std::all_of(stops.begin(), stops.end(), [this](auto id){return token(id);});
    }
    // Keep one final generated token un-fed, as ordinary greedy generation does.
    int64_t feed_end() const noexcept { return int64_t(prompt.size()) + max_new - 1; }
    bool live(int64_t pos, int32_t anchor) const noexcept {
        return pos >= int64_t(prompt.size()) && pos < feed_end() && token(anchor) && !stop(anchor);
    }
    // Native draft may write its entire configured chain, even if the target
    // window is later clipped. Stay within the same conservative five-cell
    // cursor guard as the proven native relay. Near the boundary, verify the
    // anchor alone, and prime the deferred tail using its actual successor.
    bool can_offer(int64_t pos, int32_t anchor) const noexcept {
        return live(pos, anchor) && pos <= int64_t(context) - 5;
    }
    bool prompt_window(int64_t pos, const std::vector<int32_t>& ids) const noexcept {
        return pos >= 0 && pos < int64_t(prompt.size()) && !ids.empty() && ids.size() <= 8 &&
            ids.size() <= prompt.size() - size_t(pos) &&
            std::equal(ids.begin(), ids.end(), prompt.begin() + pos);
    }
    std::vector<int32_t> clip_offer(int64_t pos, int32_t anchor,
                                    const std::vector<int32_t>& candidates) const {
        if (!can_offer(pos, anchor) || candidates.size() > spec ||
            !std::all_of(candidates.begin(), candidates.end(), [this](auto id){return token(id);}))
            throw std::invalid_argument("invalid native MTP service offer");
        std::vector<int32_t> out{anchor};
        for (auto id : candidates) {
            if (stop(id) || int64_t(out.size()) >= feed_end() - pos) break;
            out.push_back(id);
        }
        return out;
    }
    bool decode_window(int64_t pos, int32_t anchor, const std::vector<int32_t>& ids) const noexcept {
        return live(pos, anchor) && !ids.empty() && ids.front() == anchor &&
            ids.size() <= size_t(spec) + 1 && int64_t(ids.size()) <= feed_end() - pos &&
            (ids.size() == 1 || can_offer(pos, anchor)) &&
            std::all_of(ids.begin(), ids.end(), [this](auto id){return token(id) && !stop(id);});
    }
};
} // namespace strata::heterogeneous
