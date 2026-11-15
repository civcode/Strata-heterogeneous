#pragma once

// Backend-local MTP pairing only. Native MtpDrafter owns all math/KV.
// Every residual R[p] pairs with the ACTUAL token at p+1, not a rejected
// candidate or an argmax from a teacher-forced prompt. The last accepted
// residual is borrowed only until the next verifier run (or final draft).
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::heterogeneous {
class MtpPrefixCursor {
public:
    MtpPrefixCursor(size_t stride, int64_t context, int64_t vocab, size_t max_rows)
        : stride_(stride), max_rows_(max_rows), context_(context), vocab_(vocab) {}
    MtpPrefixCursor(const MtpPrefixCursor&) = delete;
    MtpPrefixCursor& operator=(const MtpPrefixCursor&) = delete;

    // Bounded seed [0,N-1) from completed terminal native
    // prompt chunks, leaving token N-1 for the ordinary verifier. Every
    // successor is known from the actual prompt. Prime consumes and fences
    // the borrowed R rows before return; no temporary tail pointer survives.
    template<class Prime>
    bool seed_bulk_prompt_chunk(const std::vector<int32_t>& prompt,int64_t pos,
                                const float* residuals,size_t rows,Prime&& prime,std::string& error) {
        if (state_!=State::ready || !stride_ || !max_rows_ || max_rows_>8 || context_<=0 || vocab_<=0 ||
            tail_ || !offered_.empty() || !residuals || !rows || rows>max_rows_ ||
            prompt.size()<2 || prompt.size()>7 || int64_t(prompt.size())>=context_ ||
            pos!=next_ || pos<0 || pos>=int64_t(prompt.size()-1) ||
            rows>prompt.size()-1-size_t(pos) ||
            (bulk_prompt_.empty() ? next_!=0 : bulk_prompt_!=prompt))
            return fail(error,"MTP bulk seed identity or extent refused");
        for (auto id:prompt) if (id<0 || id>=vocab_) return fail(error,"MTP bulk seed token outside vocabulary");
        if (bulk_prompt_.empty()) bulk_prompt_=prompt;
        if (!prime(residuals,bulk_prompt_.data()+size_t(pos)+1,rows,pos,error))
            return fail(error,"MTP bulk seed prime failed");
        next_+=int64_t(rows);
        return true;
    }

    // Callback must synchronously drain native prefill before returning.
    template<class Prime>
    bool begin(int64_t pos, const std::vector<int32_t>& tokens, Prime&& prime, std::string& error) {
        if (state_ != State::ready || !stride_ || max_rows_ == 0 || max_rows_ > 8 ||
            context_ <= 0 || vocab_ <= 0 || pos != next_ || tokens.empty() ||
            tokens.size() > max_rows_ || pos > context_ - static_cast<int64_t>(tokens.size()))
            return fail(error, "MTP prefix begin identity/extent mismatch");
        for (auto token : tokens)
            if (token < 0 || token >= vocab_) return fail(error, "MTP token outside vocabulary");
        if (!bulk_prompt_.empty()) {
            if (pos!=int64_t(bulk_prompt_.size()-1) || tokens.size()!=1 || tokens.front()!=bulk_prompt_.back())
                return fail(error,"MTP bulk seed must finish with the actual final prompt token");
            bulk_prompt_.clear();
        }
        if (!offered_.empty()) {
            if (tokens != offered_) return fail(error,"MTP relay did not consume exact offered window");
            offered_.clear();
        }
        if (tail_) {
            if (next_ <= 0 || !prime(tail_, tokens.data(), 1, next_ - 1, error))
                return fail(error, "MTP deferred tail prime failed");
            tail_ = nullptr;
        }
        tokens_ = tokens;
        state_ = State::begun;
        return true;
    }

    template<class Prime>
    bool committed(const float* residuals, size_t keep, Prime&& prime, std::string& error) {
        if (state_ != State::begun || !residuals || !keep || keep > tokens_.size())
            return fail(error, "MTP accepted prefix mismatch");
        // The final token's successor is not known yet. In particular,
        // tokens_[keep] is a rejected speculative row when keep < size.
        if (keep > 1 && !prime(residuals, tokens_.data() + 1, keep - 1, next_, error))
            return fail(error, "MTP accepted rows prime failed");
        tail_ = residuals + (keep - 1) * stride_;
        next_ += static_cast<int64_t>(keep);
        tokens_.clear();
        state_ = State::ready;
        return true;
    }

    // One-shot qualification only. Caller supplies the verified next token
    // and native draft_first, with its OWN scratch R buffer. No rearm/reset
    // after a final speculative draft; the separate draft KV may be ahead.
    template<class Draft>
    bool finish(int32_t next_token, Draft&& draft, std::string& error) {
        if (state_ != State::ready || !tail_ || next_ <= 0 || next_ >= context_ ||
            next_token < 0 || next_token >= vocab_)
            return fail(error, "MTP final draft identity mismatch");
        if (!draft(tail_, next_token, next_ - 1, error)) return fail(error, "MTP first draft failed");
        tail_ = nullptr;
        state_ = State::sealed;
        return true;
    }
    // Native draft_first writes the accepted tail paired with the verified
    // anchor and may write speculative future cells. On the next commit,
    // native prefill/draft overwrite those cells with actual accepted pairs
    // before reading them. No physical suffix zeroing is claimed; qualify
    // native causal overwrite against independent fresh-prefix replay.
    template<class Draft>
    bool offer(int32_t anchor, std::vector<int32_t>& output, Draft&& draft, std::string& error, bool allow_anchor_only=false) {
        if (state_ != State::ready || !tail_ || !offered_.empty() ||
            next_ <= 0 || next_ > context_ - 5 || anchor < 0 || anchor >= vocab_)
            return fail(error,"MTP relay offer identity or context mismatch");
        std::vector<int32_t> candidates;
        if (!draft(tail_,anchor,next_-1,candidates,error) || (!allow_anchor_only && candidates.empty()) || candidates.size()>4)
            return fail(error,"MTP native offer failed");
        for (auto id:candidates) if (id<0 || id>=vocab_) return fail(error,"MTP candidate outside vocabulary");
        offered_={anchor};
        offered_.insert(offered_.end(),candidates.begin(),candidates.end());
        output=offered_;
        tail_=nullptr; // already primed by draft_first
        return true;
    }
    int64_t committed_position() const noexcept { return next_; }
private:
    enum class State { ready, begun, sealed, failed };
    State state_ = State::ready;
    size_t stride_, max_rows_;
    int64_t context_, vocab_, next_ = 0;
    const float* tail_ = nullptr;
    std::vector<int32_t> tokens_, offered_, bulk_prompt_;
    bool fail(std::string& error, const char* message) {
        state_ = State::failed;
        tail_ = nullptr;
        offered_.clear();
        bulk_prompt_.clear();
        if (error.empty()) error = message;
        return false;
    }
};
} // namespace strata::heterogeneous
