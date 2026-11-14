#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>
namespace strata::kernels { constexpr int kVerifyMaxT=8; }
namespace strata::core {
// Fail-closed M2.17 CPU-only test-double contract.
struct StageMicroProbe {
    uint32_t n_embd = 0, n_ff = 0, conv_channels = 0, conv_kernel = 0;
    std::vector<float> data;
    std::vector<uint8_t> input_q8, swiglu_q8;
};
struct StageBranchProbe {
    uint32_t n_embd = 0, n_expert = 0, k = 0;
    uint32_t conv_channels = 0, value_heads = 0, value_dim = 0;
    std::vector<float> data;
    std::vector<int32_t> ids;
};
// Fail-closed M2.12 diagnostic contract for CPU-only native test doubles.
struct LayerZeroArithmetic {
    uint32_t n_embd = 0, qkv_channels = 0;
    std::vector<float> embedded, mixed, qkv;
    std::vector<uint8_t> q8_bytes;
};
namespace fake_resource {
inline int pool_calls=0,load_calls=0, session_init_calls=0, zero_calls=0;
inline int rope_releases=0,session_releases=0,qsa_live=0,provider_destroys=0;
inline bool fail_native_size=false, fail_load=false, fail_session_init=false;
inline std::set<std::string> skip_seen;
inline void reset(){pool_calls=load_calls=session_init_calls=zero_calls=0;
    rope_releases=session_releases=qsa_live=provider_destroys=0;
    fail_native_size=fail_load=fail_session_init=false;skip_seen.clear();}
}
struct ModelGeometry {int64_t n_layers=8,hc=2,n_embd=3,n_vocab=100;};
struct QsaState {
    bool owns_rope=false;
    const float* cos_tab=nullptr;
    bool owned=false;
    ~QsaState(){if(owned)--fake_resource::qsa_live;}
};
struct SessionState {
    int64_t layer_lo=0,layer_hi=0,max_cells=0,qsa_ord0=0,qsa_alloc=0;
    QsaState* qsa_states=nullptr;
};
struct VerifyHits {const int32_t* d_res=nullptr;};
class Verifier {
public:
    static constexpr int kStage0ProbeLayers = 12;
    static constexpr int kStage0ProbePhases = 4;
    void set_stage_micro_probe(bool) {}
    bool copy_stage_micro_probe(StageMicroProbe&, std::string& e) const {
        e = "stage microprobe unsupported in CPU test double"; return false;
    }
    void set_stage_branch_probe(bool) {}
    bool copy_stage_branch_probe(StageBranchProbe&, std::string& e) const {
        e = "branch probe unavailable in CPU test double"; return false;
    }
    void set_stage0_phase_probe(bool) {}
    bool copy_stage0_phase_probe(std::vector<float>&, std::string& e) const {
        e = "stage0 phase probe unsupported in CPU test double"; return false;
    }
    void set_layer_zero_arithmetic_probe(bool) {}
    bool copy_layer_zero_arithmetic(LayerZeroArithmetic&, std::string& e) const {
        e="layer-zero probe unsupported in CPU stub"; return false;
    }
    void set_taps(const std::vector<int>&) {}
    bool gather_taps_host(float*, int, int, std::string& err) {
        err = "tap collection is not implemented by this test double";
        return false;
    }
};
using PoolMultiFn=void(*)(void*,const float*,const int32_t*,int64_t,int64_t,float*,int64_t);
class WeightTable {
public:
    static bool pool_bytes(const std::string& dir,uint64_t& out,std::string& error,
                           const std::set<std::string>* skip=nullptr){
        std::ifstream f(dir+"/index.txt");if(!f){error="index unavailable";return false;}
        std::string line;uint64_t align=0,full=0;out=0;
        while(std::getline(f,line)){
            if(line.rfind("# align ",0)==0){
                std::istringstream h(line);std::string mark,a,p,t;uint64_t nt;
                h>>mark>>a>>align>>p>>full>>t>>nt;
            } else if(!line.empty()&&line[0]!='#') {
                std::istringstream in(line);std::string name;int file,kind;
                uint64_t off,src,doff,dbytes;in>>name>>file>>kind>>off>>src>>doff>>dbytes;
                if(skip==nullptr||skip->count(name)==0)out+=((dbytes+align-1)/align)*align;
            }
        }
        if(!align){error="missing align";return false;}
        if(!skip)out=full;
        if(fake_resource::fail_native_size)out+=256;
        return true;
    }
    bool load(const std::string& dir,void* ptr,uint64_t bytes,std::string& error,
              const std::set<std::string>* skip=nullptr){
        ++fake_resource::load_calls; fake_resource::skip_seen=skip?*skip:std::set<std::string>{};
        uint64_t expect=0;
        if(!pool_bytes(dir,expect,error,skip)||!ptr||bytes!=expect||fake_resource::fail_load){
            error="injected weight load failure";return false;
        }
        return true;
    }
};
}
