#pragma once
#include <cstdint>
#include <string>
namespace strata::kernels::cpu {
inline constexpr int H=2560, NE=512;
struct ExpertLayout{bool native=false;int64_t n_layers=8,n_expert=NE;
 uint64_t blob_bytes(int64_t) const {return 64;}};
namespace fake_layout{inline bool fail=false, native=false;inline int loaded=0;}
inline const ExpertLayout& expert_layout(){static ExpertLayout x;x.native=fake_layout::native;return x;}
inline bool expert_layout_load(const std::string&,int64_t,int64_t,std::string& err){++fake_layout::loaded;
 if(fake_layout::fail){err="bad expert layout";return false;}return true;}
}
