#include "strata_heterogeneous/mtp_request_policy.hpp"
#include <iostream>
#include <stdexcept>
#include <type_traits>
using strata::heterogeneous::MtpRequestPolicy;
using strata::heterogeneous::admit_mtp_service_request;
int main() {
    unsigned checks = 0;
    auto check = [&](bool ok) { ++checks; if (!ok) throw std::runtime_error("MTP request policy"); };
    for (unsigned bits=0;bits<64;++bits) {
        bool extended=bits&1, service=bits&2, overlap=bits&4, arm=bits&8, direct=bits&16, probe=bits&32;
        check(MtpRequestPolicy::long_worker_mode_valid(extended,service,overlap,arm,direct,probe)==
              (!extended || (!arm && !direct && (service || (!overlap && !probe)))));
    }
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
    for (unsigned cap : {2048u,2304u,4096u,8192u,8448u,32768u,262144u}) {
        MtpRequestPolicy edge{cap,4,4,248320,std::vector<int32_t>(cap-4,10),{99}};
        check(edge.valid());
        check(MtpRequestPolicy::requires_long_context(cap)==(cap>8192));
        const std::vector<int32_t> prefix(cap-5,10);
        check(edge.bulk_prefix(prefix,64));
        check(!edge.can_offer(cap-5,11)); // before prompt
        check(edge.live(cap-4,11) && !edge.can_offer(cap-4,11));
        check(edge.decode_window(cap-4,11,{11}));
        check(!edge.decode_window(cap-4,11,{11,12}));
        edge.max_new=5; check(!edge.valid());
    }
    for (unsigned cap : {0u,2047u,2049u,8193u,262143u,262145u,524288u}) {
        p.context=cap; check(!p.valid());
    }
    p.context=2048; p.spec=5; check(!p.valid());
    // Verify the real service admission helper owns an immutable snapshot,
    // including at positions far beyond the historical 8K limit.
    {
        constexpr unsigned cap=262144;
        MtpRequestPolicy input{cap,256,4,248320,std::vector<int32_t>(cap-256,10),{99}};
        input.prompt.back()=17;
        auto admitted=admit_mtp_service_request(input);
        static_assert(std::is_same_v<decltype(*admitted),const MtpRequestPolicy&>);
        check(bool(admitted));
        const auto prompt_size=admitted->prompt.size();
        input.prompt[0]=-1; input.prompt.back()=19; input.stops[0]=11; input.max_new=1;
        check(admitted->prompt.front()==10 && admitted->prompt.back()==17 && admitted->max_new==256);
        check(admitted->prompt_window(prompt_size-1,{17}));
        check(!admitted->prompt_window(prompt_size-1,{19}));
        check(!admit_mtp_service_request(input));
        input.prompt[0]=10; input.spec=3;
        check(input.valid() && !admit_mtp_service_request(input));
        for (int64_t pos=prompt_size; pos<admitted->feed_end(); ++pos) {
            check(admitted->decode_window(pos,11,{11}));
            check(!admitted->decode_window(pos,99,{99}));
            if (admitted->can_offer(pos,11)) {
                const auto ids=admitted->clip_offer(pos,11,{12,13,99,14});
                check(ids==std::vector<int32_t>({11,12,13}));
            } else {
                check(pos>int64_t(cap)-5 && !admitted->decode_window(pos,11,{11,12}));
            }
        }
        admitted.reset();
        check(!admitted);
        admitted=admit_mtp_service_request({cap,8,4,248320,{7,8,9},{99}});
        check(admitted && admitted->prompt==std::vector<int32_t>({7,8,9}) && admitted->max_new==8);
    }
    std::cout << "PASS " << checks << " request-policy assertions\n";
}
