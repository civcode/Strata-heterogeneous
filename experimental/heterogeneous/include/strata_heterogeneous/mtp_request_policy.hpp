#pragma once

// Request-local policy for native greedy MTP serving. Independent of the
// fixed-shape qualification policy (which must remain unchanged).
#include "strata_heterogeneous/long_context.hpp"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <stdexcept>
#include <vector>

namespace strata::heterogeneous {
struct MtpRequestPolicy {
    static constexpr uint32_t initial_capacity = 2048;
    static constexpr uint32_t max_candidates = 4;
    static constexpr uint32_t standard_capacity = 8192;
    static constexpr uint32_t max_capacity = max_long_context;
    static constexpr size_t bulk_prompt_limit = max_capacity-1;
    static constexpr bool capacity_valid(uint64_t cells) noexcept {
        return long_context_cells_valid(cells);
    }
    static constexpr bool requires_long_context(uint64_t cells) noexcept {
        return cells>standard_capacity;
    }
    // Extended mode admits overlap and the probe runtime only for the
    // resident service, never for diagnostic MTP or direct/armed paths.
    static constexpr bool long_worker_mode_valid(bool extended, bool service, bool overlap,
                                                  bool arm, bool direct, bool probe) noexcept {
        return !extended || (!arm && !direct && (service || (!overlap && !probe)));
    }
    static constexpr uint32_t bulk_chunk_limit = 64;
    static constexpr uint64_t prime_headroom_bytes = 512ull<<20;
    uint32_t context = 0, max_new = 0, spec = 0, vocab = 0;
    std::vector<int32_t> prompt, stops;

    bool token(int32_t id) const noexcept { return id >= 0 && uint32_t(id) < vocab; }
    bool stop(int32_t id) const noexcept {
        return std::find(stops.begin(), stops.end(), id) != stops.end();
    }
    bool valid() const noexcept {
        return capacity_valid(context) && vocab == 248320 &&
            max_new > 0 && max_new <= 256 && spec >= 1 && spec <= max_candidates &&
            !prompt.empty() && prompt.size() < context &&
            max_new <= context - prompt.size() && !stops.empty() && stops.size() <= 64 &&
            std::all_of(prompt.begin(), prompt.end(), [this](auto id){return token(id);}) &&
            std::all_of(stops.begin(), stops.end(), [this](auto id){return token(id);});
    }
    // Bulk remains within the exact admitted service capacity. The final prompt token is always
    // evaluated by the ordinary target verifier/head.
    bool bulk_prefix(const std::vector<int32_t>& ids, uint32_t chunk) const noexcept {
        return valid() && prompt.size() >= 2 && prompt.size() <= bulk_prompt_limit &&
            chunk >= 1 && chunk <= bulk_chunk_limit && ids.size() == prompt.size()-1 &&
            std::equal(ids.begin(), ids.end(), prompt.begin());
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

// Service metadata is validated once per endpoint and owned as an immutable
// snapshot. A null pointer means no admitted request. Never call valid() to
// rediscover admission in a per-window hot path (it scans the entire prompt).
inline std::unique_ptr<const MtpRequestPolicy> admit_mtp_service_request(MtpRequestPolicy policy) {
    // Public --spec 4 (three drafts) and legacy four-draft qualification.
    if ((policy.spec != 3 && policy.spec != MtpRequestPolicy::max_candidates) || !policy.valid()) return {};
    return std::make_unique<const MtpRequestPolicy>(std::move(policy));
}
} // namespace strata::heterogeneous
