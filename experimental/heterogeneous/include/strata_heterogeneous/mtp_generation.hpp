#pragma once

// Bounded actual greedy generation qualification, not general admission.
// Native drafter produces the complete candidate list. Limit only the
// submitted window, before either target backend sees it. Always leave room
// for the final un-fed output token, and never submit a speculative stop ID.
#include <algorithm>
#include <charconv>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::heterogeneous {
struct MtpGenerationPolicy {
    static constexpr unsigned max_prompt=7, max_feed=12, max_candidates=4;
    unsigned prompt_rows=0, max_new=0;
    std::vector<int32_t> stops;
    bool enabled() const noexcept { return prompt_rows!=0; }
    bool valid() const noexcept {
        if (!prompt_rows || prompt_rows>max_prompt || !max_new || max_new>max_feed ||
            prompt_rows+max_new-1>max_feed || stops.empty() || stops.size()>64) return false;
        for (auto id: stops) if (id<0 || id>=248320) return false;
        return true;
    }
    unsigned feed_end() const noexcept { return prompt_rows+max_new-1; }
    bool stop(int32_t id) const noexcept { return std::find(stops.begin(),stops.end(),id)!=stops.end(); }
    bool can_draft(int64_t pos) const noexcept {
        return valid() && pos>=int64_t(prompt_rows) && pos<=int64_t(feed_end());
    }
    bool live(int64_t pos,int32_t anchor) const noexcept {
        return can_draft(pos) && pos<int64_t(feed_end()) && !stop(anchor);
    }
    // The terminal diagnostic returns the complete offer, unsubmitted.
    // For a live offer, its anchor is retained even when all drafts are cut.
    size_t candidate_count(int64_t pos,int32_t anchor,const std::vector<int32_t>& ids) const {
        if (!can_draft(pos) || ids.empty() || ids.size()>max_candidates)
            throw std::invalid_argument("invalid bounded MTP generation offer");
        if (!live(pos,anchor)) return ids.size();
        const auto cap=std::min(ids.size(),size_t(int64_t(feed_end())-pos-1));
        size_t count=0;
        while (count<cap && !stop(ids[count])) ++count;
        return count;
    }
    bool live_window(int64_t pos,const std::vector<int32_t>& ids) const noexcept {
        return !ids.empty() && ids.size()<=max_candidates+1 && live(pos,ids.front()) &&
               int64_t(ids.size())<=int64_t(feed_end())-pos &&
               std::none_of(ids.begin(),ids.end(),[this](int32_t id){return stop(id);});
    }
};

inline MtpGenerationPolicy mtp_generation_policy(unsigned prompt,const std::string& budget,const std::string& csv) {
    MtpGenerationPolicy result;
    if (budget=="0" && csv.empty()) return result;
    const auto number=[](const std::string& value) {
        unsigned n=0;
        const auto r=std::from_chars(value.data(),value.data()+value.size(),n);
        if (value.empty() || (value.size()>1 && value.front()=='0') || r.ec!=std::errc{} ||
            r.ptr!=value.data()+value.size()) throw std::invalid_argument("invalid MTP generation integer");
        return n;
    };
    result.prompt_rows=prompt;
    result.max_new=number(budget);
    if (csv.size()>512) throw std::invalid_argument("MTP generation stop list too large");
    size_t begin=0;
    for (;;) {
        const auto end=csv.find(',',begin);
        const unsigned id=number(csv.substr(begin,end==std::string::npos?end:end-begin));
        if (id>=248320) throw std::invalid_argument("invalid MTP generation stop ID");
        result.stops.push_back(static_cast<int32_t>(id));
        if (end==std::string::npos) break;
        begin=end+1;
    }
    if (!result.valid()) throw std::invalid_argument("invalid bounded MTP generation policy");
    return result;
}
} // namespace strata::heterogeneous
