#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
namespace strata::kernels { constexpr int kVerifyMaxT = 8; }
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
struct WeightTable {};
struct ModelGeometry { int64_t n_layers=4, hc=2, n_embd=3, n_vocab=100; };
struct SessionState { int64_t layer_lo=3, layer_hi=4, max_cells=32; };
struct VerifyHits {};
using PoolMultiFn=void (*)(void*,const float*,const int32_t*,int64_t,int64_t,float*,int64_t);
namespace fake_final {
inline bool init_fails=false, run_fails=false, commit_fails=false;
inline bool copy_fails=false, wait_fails=false, nonfinite=false, nan_logit=false;
inline bool guard_on=false;
inline bool publish=false, sampling=false;
inline int runs=0, commits=0, aborts=0, keep=0, logits_copies=0;
inline const float* input=nullptr;
inline float* output=nullptr;
inline int64_t pos=0, begin=0, end=0;
inline int rows=0;
inline std::vector<int32_t> tokens;
inline void reset() {
    init_fails=run_fails=commit_fails=copy_fails=wait_fails=nonfinite=nan_logit=guard_on=false;
    publish=sampling=false;
    runs=commits=aborts=keep=logits_copies=0;
    input=nullptr;output=nullptr;pos=begin=end=rows=0;tokens.clear();
}
}
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
    static bool nan_guard_on() {return fake_final::guard_on;}
    void set_stage(int64_t b,int64_t e,const float* in,float* out) {
        fake_final::begin=b;fake_final::end=e;fake_final::input=in;fake_final::output=out;
    }
    void set_handoff_rows(int n){fake_final::rows=n;}
    void set_next(Verifier*,void*){}
    void set_handoff_flags(uint32_t*,uint32_t*,uint32_t*,uint32_t*){}
    void set_always_publish(bool b){fake_final::publish=b;}
    void set_head_sampling(bool b){fake_final::sampling=b;}
    bool init(const WeightTable&,const ModelGeometry&,SessionState&,const VerifyHits&,const void*,int,std::string&){return !fake_final::init_fails;}
    void* stream(){return nullptr;}
    int64_t vocab() const{return 100;}
    bool nonfinite_in(int t) const {return fake_final::nonfinite && t==0;}
    bool copy_logits(int t,float* host) const {
        ++fake_final::logits_copies;
        if(fake_final::copy_fails)return false;
        for(int i=0;i<100;++i)host[i]=float(t*100+i)/100.f;
        if(fake_final::nan_logit && t==0) {
            uint32_t bits=0x7fc12345u;
            std::memcpy(host,&bits,4);
        }
        return true;
    }
    bool run(int n,const int32_t* t,int64_t pos,PoolMultiFn pool,void* user,
             int32_t* out,std::string&) {
        ++fake_final::runs; fake_final::pos=pos;
        fake_final::tokens.assign(t,t+n);
        pool(user,nullptr,nullptr,n,0,nullptr,fake_final::begin);
        for(int i=0;i<n;++i)out[i]=t[i]+1;
        return !fake_final::run_fails;
    }
    bool commit(int k,std::string&) {++fake_final::commits;fake_final::keep=k;return !fake_final::commit_fails;}
    bool wait_commit(std::string&) {return !fake_final::wait_fails;}
    bool release_gpu_waits(int) {++fake_final::aborts;return true;}
};
}
