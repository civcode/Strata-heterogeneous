#include "strata_heterogeneous/mtp_prefix_cursor.hpp"
#include "strata_heterogeneous/mtp_fresh_replay.hpp"
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
    std::cout << "PASS " << n << " MTP prefix pairing CPU assertions\n";
}
