#include "strata_heterogeneous/hip_file_experts.hpp"
#include <cuda_runtime.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strata::heterogeneous;
namespace fc = strata::core::fake_cache;
namespace fe = strata::core::fake_expert;
namespace fl = strata::kernels::cpu::fake_layout;
namespace fp = strata::kernels::cpu::fake_pool;
namespace {
int assertions=0;
void require(bool yes,const char* message){++assertions;if(!yes)throw std::runtime_error(message);}
Plan plan(){Plan p;p.total_layers=8;p.model_fingerprint=123;p.geometry={2,2560};
 p.stages={{Backend::hip,0,0,4},{Backend::hip,1,4,8}};return p;}
strata::core::ModelGeometry geom(){return {};}
HipFileExpertOptions opts(){HipFileExpertOptions o;o.pack_dir="trusted-q2_0";
 o.ranked_pairs={{0,1},{4,2},{1,3},{7,0}};o.max_cache_slots=2;o.cpu_workers=2;o.free_reserve_bytes=0;return o;}
void reset(){require(fake_cuda::allocations==0,"leaked HIP allocation from previous test");
 fake_cuda::reset();fe::reset();fc::fail_open=fc::fail_admit=fc::fail_fill=fc::fail_verify=false;
 fc::opens=fc::closes=fc::admitted=fc::fills=fc::verifies=0;fc::live=false;
 fl::fail=fl::native=false;fl::loaded=0;fp::fail_create=false;fp::created=fp::destroyed=0;}
void reject_exit(HipFileExperts& e,int type){
 pid_t child=fork();require(child>=0,"fork failure");
 if(child==0){
    if(type==0){int32_t ids[10]={};float in[2560]={},out[10]={};e.run(in,ids,1,10,out,9);}
    if(type==1){int32_t ids[10]={};ids[0]=512;float in[2560]={},out[10]={};e.run(in,ids,1,10,out,0);}
    if(type==2){fe::fail_dispatch=true;int32_t ids[10]={};float in[2560]={},out[10]={};e.run(in,ids,1,10,out,0);}
    if(type==3){fe::throw_dispatch=true;int32_t ids[10]={};float in[2560]={},out[10]={};e.run(in,ids,1,10,out,0);}
    std::_Exit(0);
 }
 int status=0;require(waitpid(child,&status,0)==child,"waitpid failed");
 require(WIFEXITED(status)&&WEXITSTATUS(status)==75,"fatal expert callback did not fail-stop worker");
}
}
int main(){
try{
 std::string err;
 for(size_t stage:{size_t(0),size_t(1)}){
   reset();HipFileExperts experts;auto o=opts();
   require(!experts.ready()&&!experts.poisoned(),"fresh state");
   require(experts.open(plan(),stage,geom(),o,err),"open real provider with doubles");
   require(err.empty()&&experts.ready(),"prepared status");
   require(fake_cuda::device==static_cast<int>(stage),"correct HIP stage ordinal");
   require(experts.filled_slots()==2&&fc::admitted==2&&fc::verifies==2,"two profiled verified slots");
   require(fc::opens==1&&fc::live&&fe::source_opens==1,"native source/cache opened");
   require(fp::created==1&&fp::destroyed==0,"actual worker pool constructed");
   require(fl::loaded==1&&!fl::native,"canonical Q2_0 expert layout");
   require(fake_cuda::allocations==1&&fake_cuda::copies==1&&fake_cuda::syncs==1&&fake_cuda::infos==1,"device table uploaded and fenced");
   auto h=experts.hits();
   require(h.d_res!=nullptr&&h.h_res!=nullptr&&h.cache_base!=nullptr,"verifier cache bindings");
   require(h.blob==64&&h.n_slots==2,"verifier slot geometry");
   require(h.h_res[stage*4*512+1*(stage==0)+2*(stage==1)]==0,"top ranked slot index");
   require(h.h_res[(1-stage)*4*512+1]==-1,"foreign-stage rows never resident");
   auto* d=static_cast<const int32_t*>(h.d_res);
   require(d[stage*4*512+1*(stage==0)+2*(stage==1)]==0,"device residency matches host");
   float x[2560]{};int32_t ids[20]{};float y[20];
   for(int i=0;i<20;++i)ids[i]=static_cast<int32_t>(i%10);
   if(stage==0)ids[0]=1;else ids[0]=2;
   ids[1]=0;
   experts.run(x,ids,2,10,y,static_cast<int64_t>(stage*4));
   require(y[0]==0.0f&&y[1]==1.0f,"resident rows zero and CPU misses computed");
   require(fe::dispatches==1,"first native dispatch");
   experts.run(x,ids,2,10,y,static_cast<int64_t>(stage*4+1));
   require(fe::dispatches==2,"next layer native dispatch");
   experts.run(x,ids,2,10,y,static_cast<int64_t>(stage*4));
   require(fe::dispatches==3,"next window resets layer");
   require(!experts.open(plan(),stage,geom(),o,err),"one-shot opening");
   require(experts.ready(),"failed repeat open leaves live provider");
   if(stage==0){for(int failtype=0;failtype<4;++failtype)reject_exit(experts,failtype);}
   experts.shutdown();
   require(!experts.ready()&&experts.poisoned(),"closed provider poisoned");
   require(fake_cuda::allocations==0&&fc::closes==1&&!fc::live,"cache and table released");
   require(fe::source_closes==1&&fp::destroyed==1,"file source and worker pool released");
   require(fake_cuda::syncs>=2,"GPU synchronized before teardown");
   experts.shutdown();require(fc::closes==1&&fp::destroyed==1,"shutdown idempotent");
 }
 // STRP/v1 path: a loaded profile, filtered to each HIP stage.
 reset();{
   auto o=opts();o.ranked_pairs.clear();o.profile_path="trusted.strp";
   HipFileExperts e;require(e.open(plan(),1,geom(),o,err),"read STRP/v1 profile");
   require(fe::profile_reads==1&&e.filled_slots()==2,"profile filtered to stage");
 }
 require(fake_cuda::allocations==0,"profile test cleanup");
 // Full explicit stage capacity exceeds the previous 1024-slot cap.
 reset();{
   auto o=opts();o.ranked_pairs.clear();o.profile_path="trusted.strp";
   o.max_cache_slots=4*512;
   fe::profile_pairs.clear();
   for(int l=0;l<8;++l)
     for(int e=0;e<512;++e)fe::profile_pairs.emplace_back(l,e);
   fe::profile_slots=8*512;
   HipFileExperts e;require(e.open(plan(),1,geom(),o,err),"complete stage profile admitted");
   require(e.filled_slots()==2048&&fc::verifies==2048,"every stage slot verified");
   auto h=e.hits();require(h.n_slots==2048,"full stage cache length");
   auto* d=static_cast<const int32_t*>(h.d_res);
   for(int i=0;i<8*512;++i){
     const int32_t expected=i<4*512?-1:i-4*512;
     require(h.h_res[i]==expected&&d[i]==expected,"stage-local host/device map");
   }
   float x[2560]{};int32_t ids[10]={0,1,2,3,4,5,6,7,8,511};float y[10]{};
   e.run(x,ids,1,10,y,7);
   for(float v:y)require(v==0.0f,"all routes resident in host-served control");
 }
 require(fake_cuda::allocations==0,"full residency cleanup");
 reset();{
   auto o=opts();o.max_cache_slots=4*512;
   HipFileExperts e;require(!e.open(plan(),0,geom(),o,err),"incomplete full-stage profile refused");
   require(fc::opens==0&&fake_cuda::sets==0,"incomplete profile refused before backend allocation");
 }
 reset();{
   auto o=opts();o.max_cache_slots=4*512+1;
   HipFileExperts e;require(!e.open(plan(),0,geom(),o,err),"foreign-stage capacity refused");
   require(fc::opens==0&&fake_cuda::sets==0,"over-cap request refused before backend allocation");
 }
 // Invalid input is refused before HIP initialization.
 for(int kind=0;kind<18;++kind){
   reset();auto p=plan();auto g=geom();auto o=opts();size_t stage=0;
   if(kind==0)stage=2;
   if(kind==1)p.stages[0].backend=Backend::cuda;
   if(kind==2)g.n_layers=9;
   if(kind==3)g.n_embd=1024;
   if(kind==4)g.n_expert=8;
   if(kind==5)o.pack_dir="";
   if(kind==6)o.cpu_workers=0;
   if(kind==7)o.cpu_workers=29;
   if(kind==8)o.max_cache_slots=-1;
   if(kind==9)o.profile_path="also-a-profile";
   if(kind==10)o.ranked_pairs.clear();
   if(kind==11)o.ranked_pairs={{8,2}};
   if(kind==12)o.ranked_pairs={{0,512}};
   if(kind==13)o.ranked_pairs={{0,3},{0,3}};
   if(kind==14)o.ranked_pairs={{4,5}};
   if(kind==15){o.profile_path="profile";o.ranked_pairs.clear();fe::fail_profile=true;}
   if(kind==16){fl::fail=true;}
   if(kind==17){fl::native=true;}
   HipFileExperts e;require(!e.open(p,stage,g,o,err),"preflight rejection");
   require(!e.ready(),"invalid input not ready");
   require(fake_cuda::mallocs==0&&fc::opens==0,"preflight failure before cache HIP");
 }
 // Runtime, file, pool, partial cache/slot/transfer failures, all fail closed.
 for(int kind=0;kind<13;++kind){
   reset();auto o=opts();
   if(kind==0)fe::fail_open=true;
   if(kind==1)fake_cuda::fail_set=1;
   if(kind==2)fc::fail_open=true;
   if(kind==3)fc::fail_admit=true;
   if(kind==4)fe::missing_blob=true;
   if(kind==5)fc::fail_fill=true;
   if(kind==6)fc::fail_verify=true;
   if(kind==7)fake_cuda::fail_malloc=1;
   if(kind==8)fake_cuda::fail_copy=1;
   if(kind==9)fake_cuda::fail_sync=1;
   if(kind==10)fp::fail_create=true;
   if(kind==11)fake_cuda::fail_info=true;
   if(kind==12)fake_cuda::free_bytes=1;
   HipFileExperts e;require(!e.open(plan(),0,geom(),o,err),"fault must fail");
   require(!e.ready(),"fault not ready");
   e.shutdown();
   require(fake_cuda::allocations==0&&!fc::live,"failure reaps cache and HIP table");
   require(fp::created==fp::destroyed,"failure reaps pool workers");
   require(fe::source_opens==fe::source_closes ||
           (kind==0 && fe::source_opens==1 && fe::source_closes==0),"file mapping closed");
 }
 std::cout<<"PASS: M1.15 native expert provider CPU contract ("<<assertions<<" assertions, no GPU)\n";
 return 0;
}catch(const std::exception& ex){std::cerr<<"FAIL after "<<assertions<<": "<<ex.what()<<'\n';return 1;}
}
