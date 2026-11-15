#include "strata_heterogeneous/mtp_request_policy.hpp"
#include "strata_heterogeneous/mtp_prefix_cursor.hpp"
#include <array>
#include <iostream>
#include <tuple>
using namespace strata::heterogeneous;
int main() {
    unsigned checks=0;
    auto check=[&](bool ok){++checks;if(!ok)throw std::runtime_error("bulk prompt assertion");};
    for (unsigned n=1;n<=9;++n) {
        std::vector<int32_t> prompt;
        for(unsigned i=0;i<n;++i) prompt.push_back(static_cast<int32_t>(10+i*3));
        MtpRequestPolicy policy{2048,8,4,248320,prompt,{99}};
        std::vector<int32_t> prefix(prompt.begin(),prompt.end()-1);
        for (unsigned chunk=0;chunk<=9;++chunk)
            check(policy.bulk_prefix(prefix,chunk)==(n>=2 && n<=7 && chunk>=1 && chunk<=8));
        check(!policy.bulk_prefix(prompt,3));
        if(!prefix.empty()) {auto changed=prefix;changed.back()=5;check(!policy.bulk_prefix(changed,3));}
        if(n<2 || n>7) continue;
        for(unsigned chunk=1;chunk<=8;++chunk) {
            MtpPrefixCursor cursor(2,2048,248320,8);
            std::string err;
            std::vector<std::tuple<float,int32_t,int64_t>> seen;
            auto prime=[&](const float* r,const int32_t* next,size_t rows,int64_t pos,std::string&){
                for(size_t j=0;j<rows;++j) seen.emplace_back(r[2*j],next[j],pos+int64_t(j));
                return true;
            };
            std::array<float,16> r{};
            for(unsigned pos=0;pos<n-1;) {
                unsigned rows=std::min(chunk,n-1-pos);
                for(unsigned j=0;j<rows;++j)r[2*j]=float(pos+j+1);
                check(cursor.seed_bulk_prompt_chunk(prompt,pos,r.data(),rows,prime,err));
                pos+=rows;
            }
            check(cursor.committed_position()==n-1 && seen.size()==n-1);
            for(unsigned i=0;i<n-1;++i)check(seen[i]==std::make_tuple(float(i+1),prompt[i+1],int64_t(i)));
            r[0]=123.f;
            check(cursor.begin(n-1,{prompt.back()},prime,err));
            check(cursor.committed(r.data(),1,prime,err));
            check(seen.size()==n-1 && cursor.committed_position()==n);
            std::vector<int32_t> offer;
            check(cursor.offer(91,offer,[&](const float* row,int32_t anchor,int64_t cell,std::vector<int32_t>& out,std::string&){
                check(row[0]==123.f && anchor==91 && cell==int64_t(n-1));out={92};return true;
            },err));
            check(offer==std::vector<int32_t>({91,92}));
        }
    }
    for(int fault=0;fault<5;++fault) {
        MtpPrefixCursor cursor(2,2048,248320,8);
        std::string err;
        std::array<float,6> r{};
        auto prime=[](auto...){return true;};
        std::vector<int32_t> prompt{10,11,12,13};
        check(cursor.seed_bulk_prompt_chunk(prompt,0,r.data(),1,prime,err));
        if(fault==0)check(!cursor.seed_bulk_prompt_chunk(prompt,0,r.data(),1,prime,err));
        if(fault==1)check(!cursor.seed_bulk_prompt_chunk({10,11,12,14},1,r.data(),2,prime,err));
        if(fault==2)check(!cursor.seed_bulk_prompt_chunk(prompt,1,r.data(),3,prime,err));
        if(fault==3)check(!cursor.begin(1,{11},prime,err));
        if(fault==4)check(!cursor.seed_bulk_prompt_chunk(prompt,1,r.data(),2,[](auto...){return false;},err));
        check(!cursor.seed_bulk_prompt_chunk(prompt,1,r.data(),2,prime,err));
    }
    std::cout<<"PASS "<<checks<<" bulk prompt policy/pairing assertions\n";
}
