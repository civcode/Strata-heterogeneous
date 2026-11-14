#include "strata_heterogeneous/mtp_prefix_cursor.hpp"
#include "strata_heterogeneous/mtp_fresh_replay.hpp"
#include "strata_heterogeneous/mtp_three_round.hpp"
#include "strata_heterogeneous/mtp_generation.hpp"
#include "strata_heterogeneous/mtp_resident.hpp"
#include "strata_heterogeneous/speculation.hpp"
#include <array>
#include <iostream>
#include <stdexcept>
#include <tuple>
using strata::heterogeneous::MtpPrefixCursor;
int main() {
    int n=0;
    auto check=[&](bool ok){++n; if(!ok) throw std::runtime_error("MTP prefix cursor assertion");};
    using namespace strata::heterogeneous;
    check(mtp_probe_window_count(8,1,false));
    check(!mtp_probe_window_count(9,0,false));
    check(mtp_probe_window_count(9,0,true));
    check(mtp_probe_window_count(12,0,true));
    check(!mtp_probe_window_count(13,0,true));
    check(!mtp_probe_window_count(9,1,true));
    check(!mtp_probe_window_count(0,0,true));
    check(mtp_fresh_replay_shape(1,1,true,false));
    check(!mtp_fresh_replay_shape(2,1,true,false));
    check(!mtp_fresh_replay_shape(1,1,false,false));
    check(!mtp_fresh_replay_shape(1,1,true,true));
    for (auto x : {"", "0", "01", "13", "99", "1x", " 9", "+9", "999"}) check(!mtp_probe_parse_windows(x));
    for (auto x : {"1", "8", "9", "10", "11", "12"}) check(mtp_probe_parse_windows(x));
    check(mtp_three_shape(7,3,false,false,false));
    check(!mtp_three_shape(7,2,false,false,false));
    check(!mtp_three_shape(7,3,true,false,false));
    check(mtp_three_commit_fits(7,2,0));
    check(!mtp_three_commit_fits(7,4,0));
    check(mtp_three_window_fits(11,5,2));
    check(!mtp_three_window_fits(12,5,2));
    for (unsigned round=0;round<5;++round)
        for (int64_t pos=-1;pos<18;++pos)
            for (unsigned keep=0;keep<7;++keep)
                check(mtp_three_commit_fits(pos,keep,round)==
                    (round<3 && pos>=7+int64_t(round) && keep>=1 && keep<=5 && pos+keep+2-int64_t(round)<=12));
    for(unsigned n=0;n<=9;++n) {
        check(mtp_three_shape(n,3,false,false,false,true)==(n>=1 && n<=7));
        for(unsigned round=0;round<5;++round)
            for(int64_t pos=-1;pos<18;++pos)
                for(unsigned keep=0;keep<7;++keep)
                    check(mtp_three_commit_fits(pos,keep,round,n)==
                        (n>=1 && n<=7 && round<3 && pos>=int64_t(n+round) &&
                         keep>=1 && keep<=5 && pos+keep+2-int64_t(round)<=12));
    }
    // Exercise every bounded prompt/length combination, including terminal
    // diagnostic (full list) and zero surviving candidates (anchor only).
    for(unsigned p=0;p<=9;++p) for(unsigned budget=0;budget<=14;++budget) {
        MtpGenerationPolicy policy{p,budget,{99}};
        check(policy.valid()==(p>=1 && p<=7 && budget>=1 && budget<=12 && p+budget-1<=12));
        if(!policy.valid()) continue;
        for(int pos=-1;pos<=14;++pos) for(int stop=0;stop<6;++stop) {
            std::vector<int32_t> ids{10,11,12,13}; if(stop<4) ids[stop]=99;
            const int32_t anchor=stop==4?99:1;
            const bool live=pos>=int(p) && pos<int(p+budget-1) && anchor!=99;
            check(policy.live(pos,anchor)==live);
            if(policy.can_draft(pos)) {
                size_t want=4;
                if(live) {want=std::min<size_t>(4,p+budget-1-pos-1);if(stop<4)want=std::min(want,size_t(stop));}
                check(policy.candidate_count(pos,anchor,ids)==want);
                std::vector<int32_t> offered{anchor};offered.insert(offered.end(),ids.begin(),ids.begin()+want);
                check(policy.live_window(pos,offered)==live);
            }
        }
    }
    check(mtp_generation_policy(6,"7","99,12").valid());
    check(!mtp_generation_policy(1,"0","").enabled());
    for (const auto& bad : std::vector<std::pair<std::string,std::string>>{
        {"","99"},{"01","99"},{"2",""},{"2","99,"},{"2","-1"},{"2","248320"},{"13","99"},{"0","99"}}) {
        bool refused=false;
        try {(void)mtp_generation_policy(6,bad.first,bad.second);} catch (const std::invalid_argument&) {refused=true;}
        check(refused);
    }
    check(greedy_accepted_prefix({2724,579,23513,1830,364},{579,5713,1,1,1})==2);
    check(greedy_accepted_prefix({264},{2724})==1);
    for(uint64_t prev=0;prev<=10;++prev) for(uint64_t next=0;next<=10;++next)
        check(mtp_resident_next_request(prev,next)==(prev<8 && next==prev+1));
    check(!mtp_resident_next_request(UINT64_MAX,0));
    for(unsigned p=0;p<=9;++p) for(unsigned budget=0;budget<=14;++budget)
        for(unsigned q=0;q<=9;++q) for(unsigned cap=0;cap<=14;++cap) {
            MtpGenerationPolicy policy{p,budget,{248044,248046}};
            check(mtp_resident_shape(policy,q,cap)==(p>=1 && p<=7 && budget>=1 && budget<=12 &&
                   p+budget-1<=12 && p==q && budget==cap));
        }
    check(!mtp_resident_shape({6,7,{}},6,7));
    std::string e;
    std::array<float,8> r{11,12,21,22,31,32,41,42};
    using Row=std::tuple<float,int32_t,int64_t>;
    std::vector<Row> seen;
    auto prime=[&](const float* rows,const int32_t* next,size_t count,int64_t pos,std::string&){
        for(size_t i=0;i<count;++i) seen.emplace_back(rows[2*i],next[i],pos+int64_t(i));
        return true;
    };
    MtpPrefixCursor cursor(2,32,100,8);
    check(cursor.begin(0,{5,6,77,78},prime,e));
    check(cursor.committed(r.data(),2,prime,e));
    check(seen==std::vector<Row>{{11,6,0}});
    // Row 1 pairs with actual successor 9, not rejected 77.
    check(cursor.begin(2,{9,10,11,12},prime,e));
    check(seen==std::vector<Row>{{11,6,0},{21,9,1}});
    r[0]=51; r[2]=61;
    check(cursor.committed(r.data(),1,prime,e));
    check(cursor.committed_position()==3 && seen.size()==2);
    check(cursor.finish(42,[&](const float* row,int32_t token,int64_t pos,std::string&){
        check(row[0]==51 && token==42 && pos==2); return true;
    },e));
    check(!cursor.begin(3,{42},prime,e));
    for (int fault=0;fault<4;++fault) {
        e.clear(); MtpPrefixCursor bad(2,32,100,8);
        if(fault==0) check(!bad.begin(1,{5},prime,e));
        if(fault==1) check(!bad.begin(0,{100},prime,e));
        if(fault==2) {check(bad.begin(0,{5},prime,e));check(!bad.committed(r.data(),2,prime,e));}
        if(fault==3) {check(bad.begin(0,{5,6},prime,e));
            check(!bad.committed(r.data(),2,[](auto...){return false;},e));}
        e.clear(); check(!bad.begin(0,{5},prime,e));
    }
    e.clear(); MtpPrefixCursor boundary(2,2,100,8);
    check(boundary.begin(0,{5,6},prime,e) && boundary.committed(r.data(),2,prime,e));
    check(!boundary.finish(7,[](auto...){return true;},e));
    e.clear(); seen.clear();
    MtpPrefixCursor relay(2,32,100,8);
    r={11,12,21,22,31,32,41,42};
    auto draft=[&](const float* row,int32_t token,int64_t pos,std::vector<int32_t>& out,std::string&){
        seen.emplace_back(row[0],token,pos);
        out={23,24,25,26}; return true;
    };
    std::vector<int32_t> offer;
    check(relay.begin(0,{5,6,77,78},prime,e) && relay.committed(r.data(),2,prime,e));
    check(relay.offer(9,offer,draft,e) && offer==std::vector<int32_t>({9,23,24,25,26}));
    check(seen==std::vector<Row>({{11,6,0},{21,9,1}}));
    check(relay.begin(2,offer,prime,e) && seen.size()==2);
    r={51,52,61,62,71,72,81,82};
    check(relay.committed(r.data(),1,prime,e) && relay.committed_position()==3);
    check(relay.offer(42,offer,draft,e) && seen.back()==Row(51,42,2));
    check(relay.begin(3,offer,prime,e));
    check(relay.committed(r.data(),2,prime,e) && seen.back()==Row(51,23,3));
    check(relay.offer(99,offer,draft,e) && seen.back()==Row(61,99,4));
    check(!relay.begin(5,{99,23,24,25,27},prime,e)); // no altered/replayed candidate
    for (int failure=0;failure<3;++failure) {
        e.clear(); MtpPrefixCursor bad(2,32,100,8);
        check(bad.begin(0,{1},prime,e) && bad.committed(r.data(),1,prime,e));
        if(failure==0) {
            check(bad.offer(2,offer,draft,e));
            check(!bad.offer(2,offer,draft,e));
        } else {
            check(!bad.offer(2,offer,[&](auto,auto,auto,auto& ids,auto&){
                ids=failure==1?std::vector<int32_t>{100}:std::vector<int32_t>{};
                return true;
            },e));
        }
    }
    // Generation can submit the anchor even if every native candidate
    // was clipped by the stop/length policy. Ordinary diagnostics cannot.
    e.clear();
    MtpPrefixCursor only(2,32,100,8);
    check(only.begin(0,{1},prime,e) && only.committed(r.data(),1,prime,e));
    auto clipped=[](const float*,int32_t,int64_t,std::vector<int32_t>& ids,std::string&){ids.clear();return true;};
    check(only.offer(2,offer,clipped,e,true) && offer==std::vector<int32_t>{2});
    check(only.begin(1,offer,prime,e) && only.committed(r.data(),1,prime,e));
    check(only.offer(3,offer,draft,e) && offer.front()==3);
    e.clear();
    MtpPrefixCursor strict(2,32,100,8);
    check(strict.begin(0,{1},prime,e) && strict.committed(r.data(),1,prime,e));
    check(!strict.offer(2,offer,clipped,e));
    std::cout << "PASS " << n << " MTP prefix pairing CPU assertions\n";
}
