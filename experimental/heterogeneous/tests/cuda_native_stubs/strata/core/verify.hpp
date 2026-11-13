#pragma once
#include <cuda_runtime.h>
#include "strata/kernels/ple.hpp"
#include <map>
#include <cstdint>
#include <cstddef>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>
namespace strata::kernels { constexpr int kVerifyMaxT=8; }
namespace strata::core {
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
inline bool bad_conv=false,missing_ple_value=false;
inline std::set<std::string> skip_seen;
inline void reset(){pool_calls=load_calls=session_init_calls=zero_calls=0;
    rope_releases=session_releases=qsa_live=provider_destroys=0;
    fail_native_size=fail_load=fail_session_init=false;
    bad_conv=missing_ple_value=false;skip_seen.clear();}
}
struct ModelGeometry {int64_t n_layers=8,hc=2,n_embd=3,n_vocab=100;};
struct PleRun {
    strata::kernels::PleTable* table=nullptr;
    strata::kernels::PleWeights w{};
    const int32_t* token=nullptr;
    const int32_t* prev=nullptr;
    float* hist=nullptr;
    float* emb_dev=nullptr;
    float* emb_host=nullptr;
    float* scratch=nullptr;
    strata::kernels::PleConsts consts{};
    bool ready() const {return table && table->is_open() && token && prev && hist &&
        emb_dev && emb_host && scratch &&
        (w.key_native_data || w.key_bf16 || (w.key_codes && w.key_scales)) &&
        w.value_bf16 && w.norm_key && w.norm_query && w.norm_conv && w.conv1d_f16;}
};
struct QsaState {
    bool owns_rope=false;
    const float* cos_tab=nullptr;
    bool owned=false;
    ~QsaState(){if(owned)--fake_resource::qsa_live;}
};
struct SessionState {
    int64_t layer_lo=0,layer_hi=0,max_cells=0,qsa_ord0=0,qsa_alloc=0;
    QsaState* qsa_states=nullptr;
    float* ple_hist=nullptr;
    int32_t ple_prev[2]={-1,-1};
    int32_t ple_token=-1;
    PleRun ple{};
};
struct VerifyHits {const int32_t* d_res=nullptr;};
class Verifier {
public:
    static constexpr int kStage0ProbeLayers = 12;
    static constexpr int kStage0ProbePhases = 4;
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
};
using PoolMultiFn=void(*)(void*,const float*,const int32_t*,int64_t,int64_t,float*,int64_t);
enum class WeightKind {Verbatim=0,Bf16InF32=1,F32=2,F16InF32=3};
struct WeightRef {
    const void* data=nullptr;
    uint64_t bytes=0, codes_bytes=0,scales_bytes=0;
    int64_t elements=0;
    WeightKind kind=WeightKind::Verbatim;
    int code_bits=0;
    const void* native_data=nullptr;
    void* native_q8_1=nullptr;
    int native_type=-1;
    const void* hc_q8=nullptr;
    bool quantized() const {return code_bits!=0;}
};
class WeightTable {
    std::map<std::string,WeightRef> table_;
    static const void* fakeptr(){ static unsigned char data[65536]{}; return data; }
public:
    const WeightRef* find(const std::string& name) const {
        const auto it=table_.find(name);return it==table_.end()?nullptr:&it->second;
    }
    const std::map<std::string,WeightRef>& all() const {return table_;}
    std::map<std::string,WeightRef>& modifiable() {return table_;}
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
        const auto* storage=fakeptr();
        auto put=[&](std::string name, WeightKind kind, int bits, uint64_t bytes, int64_t elems){
            WeightRef w;w.data=storage;w.kind=kind;w.code_bits=bits;
            w.bytes=bytes;w.elements=elems;
            if(bits){w.codes_bytes=512;w.scales_bytes=256;}
            table_.emplace(std::move(name),w);
        };
        put("blk.0.attn_qkv.weight",WeightKind::Verbatim,8,800,100);
        put("blk.1.ple_key.weight",WeightKind::Verbatim,2,768,100);
        if(!fake_resource::missing_ple_value)
            put("blk.1.ple_value.weight",WeightKind::Verbatim,0,32,16);
        put("blk.1.ple_norm_key.weight",WeightKind::F32,0,64,16);
        put("blk.1.ple_norm_query.weight",WeightKind::F32,0,64,16);
        put("blk.1.ple_norm_conv.weight",WeightKind::F32,0,64,16);
        put("blk.1.ple_conv1d.weight",fake_resource::bad_conv ? WeightKind::F32 : WeightKind::Verbatim,0,
            fake_resource::bad_conv ? 64 : 32,16);
        return true;
    }
};
}
