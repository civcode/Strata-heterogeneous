#pragma once

// Host-only publication of the target-authoritative accepted sequence.
// The caller MUST have drained every physical stage's prefix commit.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace strata::heterogeneous {
struct MtpStreamBatch {
    bool terminal = false;
    std::vector<int32_t> tokens;
};

class MtpStreamCursor {
public:
    MtpStreamBatch advance(size_t prompt, uint32_t budget, int64_t pos,
                           const std::vector<int32_t>& generated, int32_t next,
                           const std::vector<int32_t>& stops) {
        if (!prompt || !budget || budget>256 || pos<0 ||
            pos>int64_t(prompt)+budget-1 || finished_ ||
            (position_>=0 && pos<=position_))
            throw std::logic_error("MTP streaming checkpoint order or budget mismatch");
        MtpStreamBatch batch;
        if (pos<int64_t(prompt)) {
            if (!generated.empty() || !published_.empty())
                throw std::logic_error("MTP streaming output before prompt commit");
        } else {
            // generated contains only accepted feedback inputs. Its final
            // successor (next) is the already verified prediction, not fed yet.
            if (generated.size()!=size_t(pos-int64_t(prompt)) ||
                published_.size()>generated.size() ||
                !std::equal(published_.begin(),published_.end(),generated.begin()) ||
                next<0)
                throw std::logic_error("MTP streaming committed/published prefix mismatch");
            batch.tokens.assign(generated.begin()+published_.size(),generated.end());
            batch.terminal=std::find(stops.begin(),stops.end(),next)!=stops.end();
            if (!batch.terminal) batch.tokens.push_back(next);
            if (batch.tokens.size()>5 || published_.size()+batch.tokens.size()>budget ||
                std::any_of(batch.tokens.begin(),batch.tokens.end(),[&](int32_t id) {
                    return id<0 || std::find(stops.begin(),stops.end(),id)!=stops.end();
                }))
                throw std::logic_error("MTP streaming batch outside output budget");
        }
        published_.insert(published_.end(),batch.tokens.begin(),batch.tokens.end());
        position_=pos;
        finished_=batch.terminal || published_.size()==budget;
        return batch;
    }
private:
    std::vector<int32_t> published_;
    int64_t position_=-1;
    bool finished_=false;
};
} // namespace strata::heterogeneous
