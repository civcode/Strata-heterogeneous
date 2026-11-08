#include "strata_heterogeneous/cuda_stage_resources.hpp"
#include <cuda_runtime.h>
#include "strata/core/verify.hpp"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unistd.h>

using namespace strata::heterogeneous;
namespace core = strata::core;
namespace fr = strata::core::fake_resource;
namespace {
int assertions=0;
void req(bool ok,const char* what){++assertions;if(!ok)throw std::runtime_error(what);}
struct Fixture {
    std::string dir;
    Fixture(){ char tmp[]="/tmp/strata_m114_XXXXXX";const char* p=::mkdtemp(tmp);
        if (!p) throw std::runtime_error("mkdtemp");
        dir = p;
        std::ofstream out(dir+"/index.txt");
        out<<"# align 256 pool 2560 tensors 10\n";
        auto row=[&](std::string name,int i){out<<name<<" 0 0 0 10 "<<i*256
             <<" 10 16 1 0 0 0 0 0 0 0 0 0 0\n";};
        row("token_embd.weight",0);row("output.weight",1);
        for(int i=0;i<8;++i)row("blk."+std::to_string(i)+".weight",i+2);
    }
    ~Fixture(){std::remove((dir+"/index.txt").c_str());::rmdir(dir.c_str());}
};
struct Experts final : CudaStageExperts {
    core::VerifyHits hits() const override {return {};}
    void run(const float*,const int32_t*,int64_t n,int64_t,float*,int64_t) override {fr::pool_calls+=static_cast<int>(n);}
    ~Experts() override {++fr::provider_destroys;}
};
Plan plan(){Plan p;p.total_layers=8;p.model_fingerprint=123;p.geometry={2,3};p.stages={{Backend::cuda,0,0,4},{Backend::cuda,1,4,8}};return p;}
CudaStageLoadOptions options(const Fixture& f) {
    CudaStageLoadOptions o;o.pack_dir=f.dir;o.max_cells=32;o.experts_per_token=10;
    o.max_t=8;o.timeout_ms=20;o.free_reserve_bytes=0;return o;
}
void reset(){req(fake_cuda::allocations==0,"leaked device allocations");
    fake_cuda::reset();fr::reset();}
}
int main(){
try {
    Fixture f;std::string e;
    for(size_t stage:{size_t(0),size_t(1)}){
        reset();
        CudaStageResources resources(core::ModelGeometry{},123);
        auto o=options(f);
        req(!resources.ready()&&!resources.loaded(),"fresh owner");
        req(resources.load(plan(),stage,o,e),"load stage");
        req(e.empty()&&resources.loaded()&&!resources.ready(),"loaded but no experts");
        req(fake_cuda::current_device==static_cast<int>(stage),"local device ordinal");
        req(resources.weight_bytes()==6*256,"stage compact weight count");
        req(resources.session_bytes()==8192,"carved session size");
        req(resources.session().layer_lo==static_cast<int64_t>(stage*4),"session low");
        req(resources.session().layer_hi==static_cast<int64_t>(stage*4+4),"session high");
        req(resources.weight_plan().skip.size()==4,"skipped foreign layer names");
        req(fr::skip_seen==resources.weight_plan().skip,"single shared skip set");
        req(fake_cuda::allocations==2&&fake_cuda::malloc_calls==2,"two CUDA arenas");
        req(fr::load_calls==1&&fr::session_init_calls==1&&fr::zero_calls==1,"native initialization sequence");
        req(!resources.load(plan(),stage,o,e),"one shot");
        try{(void)resources.make_bindings();throw std::runtime_error("unattached bindings returned");}
        catch(const std::logic_error&){}
        req(!resources.attach_experts({},e),"refuse null provider");
        req(resources.attach_experts(std::make_unique<Experts>(),e),"attach owned provider");
        req(!resources.attach_experts(std::make_unique<Experts>(),e),"refuse replacing provider");
        // The rejected provider dies immediately, not from the owner.
        req(fr::provider_destroys==1,"destroy rejected provider");
        auto bindings=resources.make_bindings();
        req(bindings.model_fingerprint==123&&bindings.max_t==8&&bindings.timeout_ms==20,"bindings identity");
        req(bindings.initial_position==0&&bindings.pool&&bindings.pool_user,"bindings pool");
        bindings.pool(bindings.pool_user,nullptr,nullptr,2,10,nullptr,0);
        req(fr::pool_calls==2,"pool dispatch");
        resources.shutdown();
        req(resources.poisoned()&&!resources.loaded(),"shutdown status");
        req(fake_cuda::allocations==0&&fr::qsa_live==0,"no leaked arena or host session descriptors");
        req(fr::rope_releases==1,"carved primary rope registration released");
        req(fr::session_releases==1&&fr::provider_destroys==2,"release pool and session");
        resources.shutdown();
        req(fr::rope_releases==1&&fr::session_releases==1,"shutdown idempotent");
    }
    // Reject unsafe configuration before contacting CUDA or allocating memory.
    for(int kind=0;kind<15;++kind){
        reset();auto p=plan();auto o=options(f);core::ModelGeometry g;
        uint64_t fp=123;size_t idx=0;
        if(kind==0)p.stages[0].backend=Backend::hip;
        if(kind==1)fp=987;
        if(kind==2)g.n_layers=7;
        if(kind==3)g.hc=4;
        if(kind==4)o.pack_dir="/tmp/m114_nonexistent_pack";
        if(kind==5)o.max_cells=0;
        if(kind==6)o.experts_per_token=-1;
        if(kind==7)o.initial_position=99;
        if(kind==8)o.max_t=9;
        if(kind==9)o.timeout_ms=0;
        if(kind==10)idx=2;
        if(kind==11)fr::fail_native_size=true;
        if(kind==12)o.arena_budget_bytes=1;
        if(kind==13)g.n_vocab=-1;
        if(kind==14)o.initial_position=5;
        CudaStageResources x(g,fp);
        req(!x.load(p,idx,o,e),"reject bad load options/planner/native mismatch");
        req(!x.loaded(),"bad owner unready");
        req(fake_cuda::malloc_calls==0&&fake_cuda::info_calls==0,"preflight CPU only");
        x.shutdown();
        req(fake_cuda::allocations==0,"no failure allocation");
    }
    // Device and loader failures clean up fully and poison one-shot owner.
    for(int kind=0;kind<7;++kind){
        reset();auto o=options(f);
        if(kind==0)fake_cuda::fail_set_at=1;
        if(kind==1)fake_cuda::fail_meminfo=true;
        if(kind==2)fake_cuda::free_bytes=8000;
        if(kind==3)fake_cuda::fail_malloc_at=1;
        if(kind==4)fr::fail_load=true;
        if(kind==5)fake_cuda::fail_malloc_at=2;
        if(kind==6)fr::fail_session_init=true;
        CudaStageResources x(core::ModelGeometry{},123);
        req(!x.load(plan(),1,o,e),"fault must fail");
        req(x.poisoned()&&!x.loaded(),"fault poisons");
        req(fake_cuda::allocations==0&&fr::qsa_live==0,"fault must not leak");
        if(kind==6)req(fr::rope_releases==1,"partial session release");
    }
    reset();{
        CudaStageResources x(core::ModelGeometry{},123);
        auto o=options(f);
        req(x.load(plan(),0,o,e),"destructor ownership load");
        req(x.attach_experts(std::make_unique<Experts>(),e),"destructor ownership attach");
    }
    req(fake_cuda::allocations==0&&fr::provider_destroys==1&&fr::qsa_live==0,"RAII cleanup");
    std::cout<<"PASS: M1.14 stage resource ownership with CPU doubles ("<<assertions
             <<" assertions; no GPU inference)\n";
    return 0;
}catch(const std::exception& ex){std::cerr<<"FAIL after "<<assertions<<" assertions: "<<ex.what()<<'\n';return 1;}
}
