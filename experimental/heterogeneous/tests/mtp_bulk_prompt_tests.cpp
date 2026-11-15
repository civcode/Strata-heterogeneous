#include "strata_heterogeneous/mtp_request_policy.hpp"
#include "strata_heterogeneous/mtp_prefix_cursor.hpp"
#include <array>
#include <iostream>
#include <tuple>
using namespace strata::heterogeneous;
int main() {
    unsigned checks=0;
    auto check=[&](bool ok){++checks;if(!ok)throw std::runtime_error("bulk prompt assertion");};
    for (unsigned capacity : {2048u,8192u}) {
    for (unsigned n : {1u,2u,3u,7u,8u,9u,63u,64u,65u,129u,2040u,2047u,2048u,2049u,4097u,8187u,8191u,8192u}) {
        std::vector<int32_t> prompt;
        for(unsigned i=0;i<n;++i) prompt.push_back(static_cast<int32_t>(10+i*3));
        MtpRequestPolicy policy{capacity,1,4,248320,prompt,{99}};
        std::vector<int32_t> prefix(prompt.begin(),prompt.end()-1);
        for (unsigned chunk=0;chunk<=65;++chunk)
            check(policy.bulk_prefix(prefix,chunk)==(n>=2 && n<capacity && chunk>=1 && chunk<=64));
        check(!policy.bulk_prefix(prompt,3));
        if(!prefix.empty()) {auto changed=prefix;changed.back()=5;check(!policy.bulk_prefix(changed,3));}
        if(n<2 || n>=capacity) continue;
        for(unsigned chunk : {1u,3u,8u,9u,31u,32u,63u,64u}) {
            MtpPrefixCursor cursor(2,capacity,248320,8);
            std::string err;
            std::vector<std::tuple<float,int32_t,int64_t>> seen;
            auto prime=[&](const float* r,const int32_t* next,size_t rows,int64_t pos,std::string&){
                check(rows>=1 && rows<=8);
                for(size_t j=0;j<rows;++j) seen.emplace_back(r[2*j],next[j],pos+int64_t(j));
                return true;
            };
            std::array<float,128> r{};
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
            if(n>capacity-5) {
                // Physical tail guard: never draft beyond capacity.
                check(!cursor.offer(91,offer,[](auto...){return true;},err));
                continue;
            }
            check(cursor.offer(91,offer,[&](const float* row,int32_t anchor,int64_t cell,std::vector<int32_t>& out,std::string&){
                check(row[0]==123.f && anchor==91 && cell==int64_t(n-1));out={92};return true;
            },err));
            check(offer==std::vector<int32_t>({91,92}));
        }
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
    // Exercise the real one-time binding path at the full metadata ceiling.
    // This is CPU-only: no long inference request, no quadratic rescanning.
    {
        const unsigned capacity=262144, n=capacity-1;
        std::vector<int32_t> prompt(n);
        for(unsigned j=0;j<n;++j) prompt[j]=int32_t(10+j%101);
        MtpPrefixCursor cursor(2,capacity,248320,8);
        std::string err; std::array<float,128> residual{};
        check(cursor.start_bulk_prompt(prompt,err));
        unsigned consumed=0;
        auto prime=[&](const float* r,const int32_t* next,size_t rows,int64_t cell,std::string&) {
            check(rows>=1 && rows<=8 && cell==consumed);
            for(size_t j=0;j<rows;++j) {
                check(r[j*2]==float(consumed+1) && next[j]==prompt[consumed+1]);
                ++consumed;
            }
            return true;
        };
        for(unsigned pos=0;pos<n-1;) {
            unsigned rows=std::min(64u,n-1-pos);
            for(unsigned j=0;j<rows;++j) residual[j*2]=float(pos+j+1);
            check(cursor.seed_bound_bulk_prompt_chunk(pos,residual.data(),rows,prime,err));
            pos+=rows;
        }
        check(consumed==n-1 && cursor.committed_position()==n-1);
        check(cursor.begin(n-1,{prompt.back()},prime,err));
        check(cursor.committed(residual.data(),1,prime,err));
        check(cursor.committed_position()==n);
        std::vector<int32_t> offer;
        check(!cursor.offer(91,offer,[](auto...){return true;},err));
    }
    // Binding is immutable, and a seed must not proceed without it.
    {
        MtpPrefixCursor cursor(2,262144,248320,8); std::string err;
        std::array<float,2> r{};
        check(!cursor.seed_bound_bulk_prompt_chunk(0,r.data(),1,[](auto...){return true;},err));
        check(!cursor.start_bulk_prompt({10,11},err));
    }
    {
        MtpPrefixCursor cursor(2,262144,248320,8); std::string err;
        check(cursor.start_bulk_prompt({10,11},err));
        check(!cursor.start_bulk_prompt({10,12},err));
    }
    // A later subbatch failure may have written draft cells, but must
    // poison the cursor and never permit adoption or retry of that prefix.
    {
        MtpPrefixCursor cursor(2,2048,248320,8);
        std::string err; std::array<float,32> r{};
        std::vector<int32_t> prompt(17,10); unsigned calls=0;
        check(!cursor.seed_bulk_prompt_chunk(prompt,0,r.data(),16,[&](auto...){return ++calls==1;},err));
        check(calls==2 && cursor.committed_position()==0);
        check(!cursor.seed_bulk_prompt_chunk(prompt,0,r.data(),16,[](auto...){return true;},err));
        check(!cursor.begin(0,{10},[](auto...){return true;},err));
    }
    // Exercise partial accepts, stale speculative suffixes and the physical
    // tail at real numeric 32K/256K positions, with a synthetic draft ledger.
    // This checks pairing/order only; it does not execute attention kernels.
    for (unsigned capacity : {32768u,262144u}) {
        const unsigned n=capacity-9;
        std::vector<int32_t> actual(n);
        for(unsigned j=0;j<n;++j) actual[j]=int32_t(1000+j%137);
        auto policy=admit_mtp_service_request({capacity,9,4,248320,actual,{99}});
        check(bool(policy));
        MtpPrefixCursor cursor(2,capacity,248320,8);
        std::string err;
        std::array<float,128> r{};
        std::vector<std::pair<float,int32_t>> cells(capacity,{-1.f,-1});
        auto prime=[&](const float* rows,const int32_t* next,size_t count,int64_t cell,std::string&) {
            check(count>=1 && count<=8 && cell>=0 && cell+int64_t(count)<=capacity);
            for(size_t j=0;j<count;++j) cells[cell+j]={rows[2*j],next[j]};
            return true;
        };
        check(cursor.start_bulk_prompt(actual,err));
        for(unsigned pos=0;pos<n-1;) {
            const unsigned count=std::min(32u,n-1-pos);
            for(unsigned j=0;j<count;++j) r[2*j]=float(pos+j+1);
            check(cursor.seed_bound_bulk_prompt_chunk(pos,r.data(),count,prime,err));
            pos+=count;
        }
        check(cursor.begin(n-1,{actual.back()},prime,err));
        r[0]=float(n);
        check(cursor.committed(r.data(),1,prime,err));
        unsigned offers=0,anchors=0;
        while (int64_t(actual.size())<policy->feed_end()) {
            const auto pos=int64_t(actual.size());
            const auto anchor=int32_t(20000+10*(pos-n));
            std::vector<int32_t> ids{anchor};
            const bool speculate=policy->can_offer(pos,anchor);
            if(speculate) {
                ++offers;
                check(cursor.offer(anchor,ids,[&](const float* row,int32_t token,int64_t cell,
                                                     std::vector<int32_t>& out,std::string&) {
                    check(cell==pos-1 && row[0]==float(pos) && token==anchor && cell+4<capacity);
                    cells[cell]={row[0],token};
                    for(int j=1;j<=4;++j) cells[cell+j]={-999.f,anchor+j};
                    const auto clipped=policy->clip_offer(pos,anchor,{anchor+1,anchor+2,anchor+3,anchor+4});
                    out.assign(clipped.begin()+1,clipped.end());
                    return true;
                },err));
            } else ++anchors;
            check(policy->decode_window(pos,anchor,ids));
            check(cursor.begin(pos,ids,prime,err)); // deferred tail still owns its old row here
            for(size_t j=0;j<ids.size();++j) r[2*j]=float(pos+j+1);
            const size_t keep=speculate?2:1;
            check(ids.size()>=keep);
            check(cursor.committed(r.data(),keep,prime,err));
            actual.insert(actual.end(),ids.begin(),ids.begin()+keep);
            check(cursor.committed_position()==int64_t(actual.size()));
            for(size_t j=size_t(pos-1);j+1<actual.size();++j)
                check(cells[j]==std::make_pair(float(j+1),actual[j+1]));
        }
        check(offers==3 && anchors==2 && actual.size()==capacity-1);
        for(size_t j=0;j+1<actual.size();++j)
            check(cells[j]==std::make_pair(float(j+1),actual[j+1]));
    }
    std::cout<<"PASS "<<checks<<" bulk prompt policy/pairing assertions\n";
}
