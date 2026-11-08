#pragma once
#include "strata/core/verify.hpp"
#include <cstdint>
namespace strata::core {
inline uint64_t session_bytes(const ModelGeometry&,int64_t,int64_t,int64_t,int64_t){return 8192;}
inline uint64_t session_init(const ModelGeometry& g,int64_t cells,int64_t,void*,SessionState& ss,
                             int64_t lo,int64_t hi){
    ++fake_resource::session_init_calls;
    ss.max_cells=cells;ss.layer_lo=lo;ss.layer_hi=hi;ss.qsa_ord0=lo/4;ss.qsa_alloc=1;
    ss.ple_hist=static_cast<float*>(std::malloc(4096));
    ss.qsa_states=new QsaState[static_cast<size_t>((g.n_layers+3)/4)];
    for(int64_t j=0;j<(g.n_layers+3)/4;++j){ss.qsa_states[j].owned=true;++fake_resource::qsa_live;}
    ss.qsa_states[ss.qsa_ord0].owns_rope=true;
    ss.qsa_states[ss.qsa_ord0].cos_tab=reinterpret_cast<float const*>(0x1234);
    return fake_resource::fail_session_init?0:8192;
}
inline void session_zero(SessionState& s,const ModelGeometry&,const float*,void*) {
    ++fake_resource::zero_calls;s.ple_prev[0]=s.ple_prev[1]=-1;s.ple_token=-1;}

inline void session_release(SessionState& s) {++fake_resource::session_releases;
    std::free(s.ple_hist);s.ple_hist=nullptr;
    for(int64_t j=0;j<s.qsa_alloc;++j)if(s.qsa_states[j].owns_rope){
        ++fake_resource::rope_releases;s.qsa_states[j].owns_rope=false;}
}
}
