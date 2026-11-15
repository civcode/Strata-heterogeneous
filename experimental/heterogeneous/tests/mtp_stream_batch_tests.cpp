#include "strata_heterogeneous/mtp_stream_batch.hpp"
#include <iostream>
#include <stdexcept>
using strata::heterogeneous::MtpStreamCursor;
int main() {
    unsigned checks=0;
    auto check=[&](bool ok) {++checks;if(!ok)throw std::runtime_error("MTP stream cursor check");};
    auto fails=[&](auto f) {bool rejected=false;try{f();}catch(const std::logic_error&){rejected=true;}check(rejected);};
    const std::vector<int32_t> stops{99};
    for (unsigned prompt:{1u,3u,9u}) for(unsigned budget=1;budget<=16;++budget)
        for(unsigned count=0;count<=budget;++count) for(unsigned keep=1;keep<=5;++keep) {
            MtpStreamCursor cursor;
            if(prompt>1)check(cursor.advance(prompt,budget,prompt-1,{},123,stops).tokens.empty());
            const bool stopped=count<budget;
            const unsigned fed=stopped?count:count-1;
            std::vector<int32_t> outputs;
            for(unsigned n=0;n<count;++n)outputs.push_back(10+static_cast<int32_t>(n));
            auto first=cursor.advance(prompt,budget,prompt,{},outputs.empty()?99:outputs.front(),stops);
            check(first.terminal==outputs.empty());
            std::vector<int32_t> collected=first.tokens;
            unsigned f=0;
            while(f<fed) {
                const unsigned n=std::min(keep,fed-f);
                f+=n;
                auto batch=cursor.advance(prompt,budget,prompt+f,
                    std::vector<int32_t>(outputs.begin(),outputs.begin()+f),
                    f==count?99:outputs.at(f),stops);
                check(batch.tokens.size()==n-(stopped&&f==count));
                check(batch.terminal==(stopped&&f==count));
                collected.insert(collected.end(),batch.tokens.begin(),batch.tokens.end());
            }
            check(collected==outputs);
        }
    MtpStreamCursor c;
    check(c.advance(2,8,2,{},10,stops).tokens==std::vector<int32_t>{10});
    fails([&]{c.advance(2,8,3,{11},12,stops);}); // changed already visible anchor
    fails([&]{c.advance(2,8,2,{},10,stops);}); // replay
    fails([&]{c.advance(2,8,8,{10,11,12,13,14,15},16,stops);}); // oversized batch
    check(c.advance(2,8,4,{10,11},99,stops).tokens==std::vector<int32_t>{11});
    fails([&]{c.advance(2,8,5,{10,11,12},13,stops);}); // after EOS
    std::cout << "MTP stream cursor: PASS " << checks << " assertions\n";
}
