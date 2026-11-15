#include "strata_heterogeneous/mtp_bulk_seed_stats.hpp"
#include <array>
#include <stdexcept>
#include <iostream>
using namespace strata::heterogeneous;
int main() {
    unsigned checks=0;
    auto check=[&](bool ok){++checks;if(!ok)throw std::runtime_error("bulk seed stats");};
    for (bool batching : {false,true}) {
        MtpPrefixCursor cursor(2,2048,248320,8);
        MtpBulkSeedStats stats;
        std::string err;
        std::vector<int32_t> prompt(77,10);
        std::array<float,128> r{};
        check(cursor.start_bulk_prompt(prompt,err));
        auto prime=[](auto...){return true;};
        auto batch=[&](const float*,const int32_t*,size_t n,int64_t,std::string&) {
            return batching && n>=32 ? MtpBulkSeedResult::completed : MtpBulkSeedResult::declined;
        };
        check(stats.seed(cursor,0,r.data(),32,prime,batch,err));
        check(stats.seed(cursor,32,r.data(),32,prime,batch,err));
        check(stats.seed(cursor,64,r.data(),12,prime,batch,err));
        check(stats.rows()==76 && stats.chunks==3 && cursor.committed_position()==76);
        check(stats.batch_rows==(batching?64u:0u) && stats.batch_calls==(batching?2u:0u));
        check(stats.narrow_rows==(batching?12u:76u) && stats.narrow_calls==(batching?2u:10u));
        check(stats.declines==(batching?1u:3u));
        check(stats.callback_ns>=stats.batch_ns+stats.narrow_ns+stats.decline_ns);
        check(cursor.begin(76,{10},prime,err));
        check(cursor.committed(r.data(),1,prime,err));
        check(stats.rows()==76); // decode/final prompt uses separate narrow path
        stats={};
        check(stats.rows()==0 && stats.chunks==0 && stats.callback_ns==0 &&
              stats.batch_ns==0 && stats.decline_ns==0 && stats.narrow_ns==0 &&
              stats.batch_calls==0 && stats.narrow_calls==0 && stats.declines==0);
    }
    for(bool hard_batch : {false,true}) {
        MtpPrefixCursor cursor(2,2048,248320,8); MtpBulkSeedStats stats;
        std::array<float,128> r{}; std::string err;
        check(cursor.start_bulk_prompt(std::vector<int32_t>(65,10),err));
        check(stats.seed(cursor,0,r.data(),32,[](auto...){return true;},
                         [](auto...){return MtpBulkSeedResult::completed;},err));
        const auto before=stats;
        unsigned calls=0;
        check(!stats.seed(cursor,32,r.data(),32,
            [&](auto...){return ++calls==1;},
            [&](auto...){return hard_batch?MtpBulkSeedResult::failed:MtpBulkSeedResult::declined;},err));
        check(calls==(hard_batch?0u:2u));
        check(stats.rows()==32 && stats.chunks==1 && cursor.committed_position()==32);
        check(stats.batch_calls==before.batch_calls && stats.batch_rows==before.batch_rows &&
              stats.narrow_rows==before.narrow_rows && stats.narrow_calls==before.narrow_calls &&
              stats.declines==before.declines && stats.callback_ns==before.callback_ns &&
              stats.batch_ns==before.batch_ns && stats.narrow_ns==before.narrow_ns && stats.decline_ns==before.decline_ns);
        err.clear();
        check(!stats.seed(cursor,32,r.data(),32,[](auto...){return true;},
                          [](auto...){return MtpBulkSeedResult::completed;},err));
        check(stats.rows()==32 && stats.chunks==1);
    }
    std::cout<<"PASS "<<checks<<" bulk seed accounting assertions\n";
}
