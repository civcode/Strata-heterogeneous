// M1.17 production CUDA native-stage source against CPU-only native/CUDA doubles.
#include "strata_heterogeneous/cuda_native_stage.hpp"
#include "strata/core/session.hpp"
#include <cuda_runtime.h>

#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
using namespace strata::heterogeneous;
namespace core = strata::core;
namespace {
int assertions=0;
void need(bool yes,const char* what){++assertions;if(!yes)throw std::runtime_error(what);}
struct Fixture {
    std::string dir, pack;
    Fixture(){char tmp[]="/tmp/strata-m117-XXXXXX";if(!::mkdtemp(tmp))throw std::runtime_error("mkdtemp");
      dir=tmp;pack=dir+"/pack";std::filesystem::create_directory(pack);
      std::ofstream out(pack+"/index.txt");
      out<<"# align 256 pool 1024 tensors 1\n";
      out<<"blk.0.attn_qkv.weight 0 0 0 1024 0 1024 2560 64 8 0 32 0 0 0 0 0 0 0\n";
    }
    ~Fixture(){std::error_code ignored;std::filesystem::remove_all(dir,ignored);}
};
Plan make_plan(){Plan p;p.total_layers=8;p.model_fingerprint=77;p.geometry={2,3};
 p.stages={{Backend::cuda,0,0,3},{Backend::hip,0,3,8}};return p;}
CudaStageLoadOptions loadopts(const Fixture& f){CudaStageLoadOptions o;
 o.pack_dir=f.pack;o.max_cells=512;o.experts_per_token=10;o.require_native_stage=true;return o;}
CudaNativeStageOptions opts(const Fixture& f){CudaNativeStageOptions o;
 o.gguf_shards={f.dir+"/first.gguf",f.dir+"/ple.gguf"};o.ple_gguf_path=o.gguf_shards.back();return o;}
struct Expert : CudaStageExperts {
 core::VerifyHits hits() const override{return {};}
 void run(const float*,const int32_t*,int64_t,int64_t,float*,int64_t)override{}
};
}
namespace strata::heterogeneous {
bool plan_stage_weights(const Plan&,size_t,const std::string& path,StageWeightPlan& out,std::string& err){
 auto dir=std::filesystem::path(path).parent_path().string();
 uint64_t n=0;if(!core::WeightTable::pool_bytes(dir,n,err))return false;
 out={};out.full_pool_bytes=n;out.selected_pool_bytes=n;out.total_tensors=1;out.selected_tensors=1;
 err.clear();return true;
}
}
int main(){try{
 Fixture f;auto plan=make_plan();core::ModelGeometry geometry;
 fake_cuda::reset();core::fake_resource::reset();core::fake_native::reset();
 // Native stage must not attach to an uninitialized owner.
 {CudaStageResources owner(geometry,77);CudaNativeStage native;std::string err;
  need(!native.open(owner,plan,0,opts(f),err),"unloaded owner refused");
  need(fake_cuda::malloc_calls==0,"unloaded owner did not allocate CUDA");}
 // Valid path, PLE table, and all native aliases are live before bindings.
 {CudaStageResources owner(geometry,77);std::string err;
  need(owner.load(plan,0,loadopts(f),err),"resource owner prepared");
  need(!owner.native_ready(),"native starts unready");
  CudaNativeStage native;
  need(native.open(owner,plan,0,opts(f),err),"native GGUF and PLE prepared");
  need(native.ready()&&native.ple_ready()&&owner.native_ready(),"native/PLE is ready");
  need(native.native_bytes()==4096,"native memory measured");
  need(owner.session().ple.ready(),"actual session PLE fully wired");
  need(owner.session().ple.w.key_native_data!=nullptr,"native key attached");
  need(owner.session().ple.w.conv1d_f16!=nullptr,"F16 conv wired");
  need(owner.attach_experts(std::make_unique<Expert>(),err),"native owner accepts expert pool");
  auto bindings=owner.make_bindings();
  need(bindings.session.ple.ready(),"verifier bindings receive wired PLE session");
  need(bindings.zero_session!=nullptr,"verifier bindings receive session reset");
  const int zero_calls=core::fake_resource::zero_calls;
  bindings.session.ple_prev[0]=11;
  bindings.session.ple_prev[1]=12;
  bindings.session.ple_token=13;
  bindings.zero_session(bindings.session,bindings.geometry,nullptr);
  need(core::fake_resource::zero_calls==zero_calls+1,"session reset dispatches once");
  need(bindings.session.ple_prev[0]==-1&&bindings.session.ple_prev[1]==-1&&
       bindings.session.ple_token==-1,"session reset clears PLE history tokens");
  need(bindings.session.ple.ready(),"session reset preserves PLE wiring");
  native.shutdown();
  need(!owner.native_ready()&&!native.ready(),"native shutdown makes binding unavailable");
  bool refused=false;try{(void)owner.make_bindings();}catch(const std::logic_error&){refused=true;}
  need(refused,"owner cannot publish dangling native bindings");
  owner.shutdown();
  need(fake_cuda::allocations==0,"all CUDA allocations released");
  need(core::fake_resource::qsa_live==0,"no QSA state escaped");}
 // Missing mandatory first-stage native QKV: fail before GPU native allocs.
 {core::fake_native::reset();core::fake_native::no_qkv=true;
  CudaStageResources owner(geometry,77);std::string err;
  need(owner.load(plan,0,loadopts(f),err),"owner prepared for negative QKV");
  CudaNativeStage native;
  need(!native.open(owner,plan,0,opts(f),err),"missing QKV refused");
  need(err.find("attn_qkv")!=std::string::npos,"QKV diagnostic named");
  native.shutdown();owner.shutdown();need(fake_cuda::allocations==0,"missing QKV has no leaked allocations");}
 // A bad native size estimate must fail without pretending to serve weights.
 {core::fake_native::reset();core::fake_native::fail_price=true;
  CudaStageResources owner(geometry,77);std::string err;
  need(owner.load(plan,0,loadopts(f),err),"owner prepared for pricing failure");
  CudaNativeStage native;need(!native.open(owner,plan,0,opts(f),err),"native pricing error refused");
  need(!owner.native_ready(),"price failure never marks ready");
  native.shutdown();owner.shutdown();need(fake_cuda::allocations==0,"price failure cleans up");}
 // Native loading failure uses partial cleanup and leaves the owner poisoned.
 {core::fake_native::reset();core::fake_native::fail_load=true;
  CudaStageResources owner(geometry,77);std::string err;
  need(owner.load(plan,0,loadopts(f),err),"owner prepared for native failure");
  CudaNativeStage native;need(!native.open(owner,plan,0,opts(f),err),"native load error refused");
  need(!owner.native_ready(),"failed native never ready");
  native.shutdown();owner.shutdown();need(fake_cuda::allocations==0,"failed load released CUDA");}
 // Even if GGUF advertises a QKV projection, open must check the actual
 // WeightRef alias after NativeDense::load; CPU metadata alone is insufficient.
 {core::fake_native::reset();core::fake_native::bind_qkv=false;
  CudaStageResources owner(geometry,77);std::string err;
  need(owner.load(plan,0,loadopts(f),err),"owner before unbound QKV");
  CudaNativeStage native;
  need(!native.open(owner,plan,0,opts(f),err),"unbound QKV rejected");
  need(err.find("attn_qkv")!=std::string::npos,"unbound QKV names tensor");
  native.shutdown();owner.shutdown();need(fake_cuda::allocations==0,"unbound QKV cleanup");}
 // PLE F32 convolution is not F16; a cast would silently corrupt the model.
 {core::fake_native::reset();core::fake_resource::bad_conv=true;
  CudaStageResources owner(geometry,77);std::string err;
  need(owner.load(plan,0,loadopts(f),err),"owner before wrong PLE storage");
  CudaNativeStage native;
  need(!native.open(owner,plan,0,opts(f),err),"wrong F16 PLE conv refused");
  native.shutdown();owner.shutdown();need(fake_cuda::allocations==0,"bad PLE conv cleanup");
  core::fake_resource::bad_conv=false;}
 {core::fake_native::reset();core::fake_resource::missing_ple_value=true;
  CudaStageResources owner(geometry,77);std::string err;
  need(owner.load(plan,0,loadopts(f),err),"owner before missing PLE weight");
  CudaNativeStage native;
  need(!native.open(owner,plan,0,opts(f),err),"missing PLE weight refused");
  need(err.find("ple_value")!=std::string::npos,"diagnostic names PLE tensor");
  native.shutdown();owner.shutdown();need(fake_cuda::allocations==0,"missing PLE weight cleanup");
  core::fake_resource::missing_ple_value=false;}
 {core::fake_native::reset();core::fake_native::bad_key=true;
  CudaStageResources owner(geometry,77);std::string err;
  need(owner.load(plan,0,loadopts(f),err),"owner before incompatible native PLE key");
  CudaNativeStage native;
  need(!native.open(owner,plan,0,opts(f),err),"invalid native key type refused");
  native.shutdown();owner.shutdown();need(fake_cuda::allocations==0,"bad PLE native key cleanup");}
 // Missing PLE file and partial PLE CUDA allocations are fail-closed.
 {core::fake_native::reset();CudaStageResources owner(geometry,77);std::string err;
  need(owner.load(plan,0,loadopts(f),err),"owner prepared for PLE failure");
  CudaNativeStage native;auto o=opts(f);o.gguf_shards[1]=f.dir+"/bad-ple.gguf";o.ple_gguf_path=o.gguf_shards[1];
  need(!native.open(owner,plan,0,o,err),"PLE table failure refused");
  native.shutdown();owner.shutdown();need(fake_cuda::allocations==0,"failed PLE cleanup");}
 for(int fail_at:{3,4,5}){
  core::fake_native::reset();fake_cuda::reset();
  CudaStageResources owner(geometry,77);std::string err;
  need(owner.load(plan,0,loadopts(f),err),"owner before allocation fault");
  fake_cuda::fail_malloc_at=fail_at;
  CudaNativeStage native;need(!native.open(owner,plan,0,opts(f),err),"partial PLE allocation refused");
  fake_cuda::fail_malloc_at=0;native.shutdown();owner.shutdown();
  need(fake_cuda::allocations==0,"partial PLE allocations released");
 }
 // Memory accounting includes NativeDense and the PLE scratch; no overflow or implicit overcommit.
 {core::fake_native::reset();fake_cuda::reset();
  CudaStageResources owner(geometry,77);std::string err;
  need(owner.load(plan,0,loadopts(f),err),"owner before budget check");
  CudaNativeStage native;auto o=opts(f);o.total_arena_budget_bytes=1024;
  need(!native.open(owner,plan,0,o,err),"native-inclusive arena budget enforced");
  native.shutdown();owner.shutdown();need(fake_cuda::allocations==0,"budget rejection cleanup");}
 {core::fake_native::reset();fake_cuda::reset();
  CudaStageResources owner(geometry,77);std::string err;
  need(owner.load(plan,0,loadopts(f),err),"owner before free-memory check");
  CudaNativeStage native;auto o=opts(f);
  fake_cuda::free_bytes=4095;
  need(!native.open(owner,plan,0,o,err),"low free memory refused");
  fake_cuda::reset();native.shutdown();owner.shutdown();need(fake_cuda::allocations==0,"memory rejection cleanup");}
 // Wrong shutdown order is a deliberate fail-stop process exit, not UAF.
 {pid_t child=::fork();need(child>=0,"fork for lifetime guard");
  if(child==0){fake_cuda::reset();core::fake_native::reset();
   CudaStageResources owner(geometry,77);std::string err;
   if(!owner.load(plan,0,loadopts(f),err))std::_Exit(5);
   CudaNativeStage native;
   if(!native.open(owner,plan,0,opts(f),err))std::_Exit(6);
   owner.shutdown();std::_Exit(7);
  }
  int status=0;need(::waitpid(child,&status,0)==child,"reaped guard child");
  need(WIFEXITED(status)&&WEXITSTATUS(status)==70,"out-of-order owner shutdown stopped process");}
 std::cout<<"M1.17 CUDA NativeDense and PLE contracts: PASS ("<<assertions<<" assertions; no GPU)\n";
 return 0;
 }catch(const std::exception& e){std::cerr<<"M1.17 FAIL after "<<assertions<<": "<<e.what()<<'\n';return 1;}
}
