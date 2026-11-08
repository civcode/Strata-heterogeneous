#include "strata/core/primary_serving_resources.hpp"
#include "strata/core/layer_split_stage_owner.hpp"
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
namespace {
int current=0, syncs=0, frees=0, events=0,fail=0;
bool fenced=false;
void require(bool ok, const char* s){if(!ok){std::fprintf(stderr,"ASSERTION_FAILED %s\n",s);std::_Exit(88);}}
struct Dependency {~Dependency(){require(frees==0&&events==0,"dependent destruction after resources");}};
struct Owned : strata::core::PrimaryServingResources {Dependency dep;};
}
extern "C" cudaError_t cudaGetDevice(int* p){if(fail==1)return cudaErrorUnknown;*p=current;return cudaSuccess;}
extern "C" cudaError_t cudaSetDevice(int d){if((fail==2&&d==1)||(fail==6&&d==0))return cudaErrorUnknown;current=d;return cudaSuccess;}
extern "C" cudaError_t cudaDeviceSynchronize(void){if(fail==3)return cudaErrorUnknown;require(current==1,"wrong sync owner");++syncs;fenced=true;return cudaSuccess;}
extern "C" cudaError_t cudaFree(void* p){require(p&&current==1&&fenced,"bad free owner or fence");if(fail==4)return cudaErrorUnknown;++frees;std::printf("FREE %p\n",p);return cudaSuccess;}
extern "C" cudaError_t cudaEventDestroy(cudaEvent_t p){require(p&&current==1&&fenced,"bad event owner or fence");if(fail==5)return cudaErrorUnknown;++events;std::printf("EVENT %p\n",p);return cudaSuccess;}
extern "C" const char* cudaGetErrorString(cudaError_t e){return e==cudaSuccess?"success":"injected failure";}
int main(int argc,char** argv){
if (argc != 2) return 2;
const std::string mode = argv[1];
if(mode=="fail-get")fail=1;
if(mode=="fail-select")fail=2;
if(mode=="fail-sync")fail=3;
if(mode=="fail-free")fail=4;
if(mode=="fail-event")fail=5;
if(mode=="fail-restore")fail=6;
if(mode=="same")current=1;
using Ptr=std::unique_ptr<Owned,strata::core::LayerSplitStageDelete<Owned>>;
Ptr p(new Owned);p->dev=1;
if(mode!="empty"){
p->cleanup_armed=true;
void* base[]={reinterpret_cast<void*>(0x1001),reinterpret_cast<void*>(0x1002),reinterpret_cast<void*>(0x1003),reinterpret_cast<void*>(0x1004),reinterpret_cast<void*>(0x1005),reinterpret_cast<void*>(0x1006),reinterpret_cast<void*>(0x1007),reinterpret_cast<void*>(0x1008),reinterpret_cast<void*>(0x1009),reinterpret_cast<void*>(0x1010),reinterpret_cast<void*>(0x1011),reinterpret_cast<void*>(0x1012)};
auto& r=*p;
r.session_arena=base[0];
if(mode!="partial"){
r.logits=base[1];r.hit_scratch=base[2];r.hit_slot=base[3];r.hit_destination=base[4];r.hit_q8=base[5];r.hit_q8_scale=base[6];r.hit_output=base[7];r.embedding=base[8];r.next_token=base[9];r.residency=base[10];r.hit_count=base[11];r.hit_done=reinterpret_cast<void*>(0x2020);
}}
p.reset();
if(mode=="empty")require(syncs==0&&frees==0&&events==0,"empty must not use GPU");
else if(mode=="partial")require(syncs==1&&frees==1&&events==0,"partial must free one");
else require(syncs==1&&frees==12&&events==1,"release all raw resources");
require(current==(mode=="same"?1:0),"restore prior device");
std::puts("PASSED");return 0;
}
