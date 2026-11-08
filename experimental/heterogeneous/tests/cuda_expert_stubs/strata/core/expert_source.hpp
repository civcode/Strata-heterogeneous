#pragma once
#include "strata/core/verify.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include <cstdint>
#include <string>
#include <vector>
#include <utility>
#include <stdexcept>
namespace strata::core {
namespace fake_expert{
inline bool fail_open=false,fail_profile=false,missing_blob=false,fail_dispatch=false,throw_dispatch=false;
inline int source_opens=0,source_closes=0,blobs=0,profile_reads=0,dispatches=0;
inline int64_t profile_slots=4;
inline std::vector<std::pair<int32_t,int32_t>> profile_pairs{{0,1},{4,2},{1,3},{7,0}};
inline void reset(){fail_open=fail_profile=missing_blob=fail_dispatch=throw_dispatch=false;
 source_opens=source_closes=blobs=profile_reads=dispatches=0;profile_slots=4;
 profile_pairs={{0,1},{4,2},{1,3},{7,0}};}
}
inline bool read_expert_profile(const std::string&,int64_t,int64_t,
 std::vector<std::pair<int32_t,int32_t>>& p,int64_t& slots,std::string& err){++fake_expert::profile_reads;
 if(fake_expert::fail_profile){err="bad STRP profile";return false;}
 p=fake_expert::profile_pairs;slots=fake_expert::profile_slots;return true;}
class FileExpertSource {
 bool opened_=false;
 std::vector<uint8_t> bytes_=std::vector<uint8_t>(64);
public:
 bool open(const std::string&,int64_t,int64_t,std::string& err){++fake_expert::source_opens;
 if(fake_expert::fail_open){err="invalid experts.bin";return false;}opened_=true;
 for(size_t i=0;i<bytes_.size();++i)bytes_[i]=static_cast<uint8_t>(i);
 return true;}
 const uint8_t* blob(int64_t,int64_t){++fake_expert::blobs;
 return opened_&&!fake_expert::missing_blob?bytes_.data():nullptr;}
 void close(){if(opened_){++fake_expert::source_closes;opened_=false;}}
 ~FileExpertSource(){close();}
};
struct ExpertDispatch {
 FileExpertSource* src=nullptr;
 strata::kernels::cpu::ExpertPool* pool=nullptr;
 int64_t n_expert=512;
 const int32_t* host_res=nullptr;
 const uint8_t* cache_base=nullptr;
 int64_t cache_blob=0;
 const uint64_t* cache_slot_off=nullptr;
 void* plan=nullptr;
 int64_t layers=0,fail_layer=-1,fail_expert=-1;
 bool failed=false;
 const char* fail=nullptr;
};
inline void set_tail_skip_rank(int){}
inline void expert_pool_dispatch_multi(ExpertDispatch& d,const float*,const int32_t* ids,
 int64_t n_tok,int64_t k,float* out){++fake_expert::dispatches;
 if(fake_expert::throw_dispatch)throw std::runtime_error("native dispatch threw");
 if(fake_expert::fail_dispatch||!d.pool||!d.src||!d.host_res){d.failed=true;d.fail="native dispatch failed";
 d.fail_layer=d.layers;d.fail_expert=ids[0];return;}
 for(int64_t i=0;i<n_tok*k;++i){auto slot=d.host_res[static_cast<size_t>(d.layers*d.n_expert+ids[i])];
 out[i]=slot>=0?0.0f:static_cast<float>(ids[i]+1);}
 ++d.layers;
}
}
