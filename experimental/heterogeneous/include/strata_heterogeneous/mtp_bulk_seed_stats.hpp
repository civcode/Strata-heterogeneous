#pragma once

#include "strata_heterogeneous/mtp_prefix_cursor.hpp"
#include <chrono>
#include <cstdint>

namespace strata::heterogeneous {

// Request-local, terminal-owner wall time. No extra GPU synchronization,
// per-row allocations, or device readback. Only completed chunks contribute.
struct MtpBulkSeedStats {
    using Clock = std::chrono::steady_clock;
    uint64_t chunks=0, batch_calls=0, batch_rows=0, narrow_calls=0, narrow_rows=0;
    uint64_t declines=0, callback_ns=0, batch_ns=0, narrow_ns=0, decline_ns=0;

    uint64_t rows() const noexcept { return batch_rows+narrow_rows; }

    static uint64_t elapsed(Clock::time_point start) noexcept {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now()-start).count());
    }

    template<class Prime,class Batch>
    bool seed(MtpPrefixCursor& cursor,int64_t pos,const float* residuals,size_t count,
              Prime&& prime,Batch&& batch,std::string& error) {
        // Keep tentative accounting private until the cursor adopts the
        // entire chunk. Failure cannot look like a successful partial seed.
        MtpBulkSeedStats delta;
        const auto start=Clock::now();
        const bool ok=cursor.seed_bound_bulk_prompt_chunk(pos,residuals,count,
            [&](const float* r,const int32_t* next,size_t n,int64_t cell,std::string& e) {
                const auto t=Clock::now();
                const bool done=prime(r,next,n,cell,e);
                delta.narrow_ns+=elapsed(t);
                if (done) { ++delta.narrow_calls; delta.narrow_rows+=n; }
                return done;
            },
            [&](const float* r,const int32_t* next,size_t n,int64_t cell,std::string& e) {
                const auto t=Clock::now();
                const auto result=batch(r,next,n,cell,e);
                const uint64_t ns=elapsed(t);
                if (result==MtpBulkSeedResult::completed) {
                    ++delta.batch_calls; delta.batch_rows+=n; delta.batch_ns+=ns;
                } else if (result==MtpBulkSeedResult::declined) {
                    ++delta.declines; delta.decline_ns+=ns;
                }
                return result;
            },error);
        if (!ok) return false;
        ++chunks;
        batch_calls+=delta.batch_calls; batch_rows+=delta.batch_rows; batch_ns+=delta.batch_ns;
        narrow_calls+=delta.narrow_calls; narrow_rows+=delta.narrow_rows; narrow_ns+=delta.narrow_ns;
        declines+=delta.declines; decline_ns+=delta.decline_ns;
        callback_ns+=elapsed(start);
        return true;
    }
};
} // namespace strata::heterogeneous
