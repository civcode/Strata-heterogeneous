#include "strata_heterogeneous/mtp_request_policy.hpp"
#include <iostream>
#include <stdexcept>
using strata::heterogeneous::MtpRequestPolicy;
int main() {
    unsigned checks = 0;
    auto check = [&](bool ok) { ++checks; if (!ok) throw std::runtime_error("MTP request policy"); };
    for (unsigned p : {1u, 7u, 32u, 511u, 1792u, 2043u, 2047u}) {
        for (unsigned budget : {1u, 2u, 7u, 64u, 256u}) {
            for (unsigned spec = 1; spec <= 4; ++spec) {
                MtpRequestPolicy policy{2048,budget,spec,248320,std::vector<int32_t>(p,10),{99,100}};
                check(policy.valid() == (p + budget <= 2048));
                if (!policy.valid()) continue;
                check(policy.prompt_window(0,{10}));
                check(!policy.prompt_window(0,{11}));
                check(!policy.prompt_window(p,{10}));
                for (int64_t pos = p; pos <= policy.feed_end(); ++pos) {
                    check(policy.live(pos,11) == (pos < policy.feed_end()));
                    check(!policy.live(pos,99));
                    check(policy.can_offer(pos,11) == (pos < policy.feed_end() && pos <= 2043));
                    check(policy.decode_window(pos,11,{11}) == policy.live(pos,11));
                    if (!policy.can_offer(pos,11)) continue;
                    std::vector<int32_t> drafts(spec,12);
                    for (unsigned eos=0; eos<=spec; ++eos) {
                        auto proposed=drafts;
                        if (eos<spec) proposed[eos]=99;
                        auto offer=policy.clip_offer(pos,11,proposed);
                        check(offer.size() == 1 + std::min<size_t>(eos,policy.feed_end()-pos-1));
                        check(policy.decode_window(pos,11,offer));
                    }
                    check(policy.clip_offer(pos,11,{}).size()==1);
                    bool rejected=false;
                    try {(void)policy.clip_offer(pos,11,{-1});} catch(const std::invalid_argument&) { rejected=true; }
                    check(rejected);
                }
            }
        }
    }
    MtpRequestPolicy p{2048,8,4,248320,{1,2,3},{99}};
    check(p.valid() && p.prompt_window(1,{2,3}) && !p.prompt_window(1,{2,4}));
    check(!p.decode_window(3,10,{11,12}) && !p.decode_window(3,10,{10,99}));
    p.context=262144; check(!p.valid()); // no unqualified capacity promotion
    p.context=2048; p.spec=5; check(!p.valid());
    std::cout << "PASS " << checks << " request-policy assertions\n";
}
