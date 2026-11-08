#include "strata_heterogeneous/cuda_worker_config.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <cstdlib>
#include <unistd.h>
using namespace strata::heterogeneous;
namespace {
int assertions=0;
void need(bool ok, const char* m){++assertions;if(!ok)throw std::runtime_error(m);}
struct Fixture {
 std::string dir, pack, profile, config, index;
 Fixture(){char p[]="/tmp/strata-m116-XXXXXX";if(!::mkdtemp(p))throw std::runtime_error("mkdtemp");dir=p;pack=dir+"/pack";std::filesystem::create_directory(pack);
 profile=dir+"/profile.strp";config=dir+"/worker.cfg";index=pack+"/index.txt";
 {std::ofstream f(index);f<<"# align 256 pool 512 tensors 1\n";f<<"output.weight 0 0 0 10 0 512 2560 248320 0 0 0 0 0 0 0 0 0 0\n";}
 {std::ofstream f(profile,std::ios::binary);char hdr[24]={};hdr[0]='S';hdr[1]='T';hdr[2]='R';hdr[3]='P';hdr[4]=1;f.write(hdr,sizeof hdr);}
 }
 ~Fixture(){std::error_code ec;std::filesystem::remove_all(dir,ec);}
 std::string text(const std::string& sha) const {
  return "version=1\npack_dir="+pack+"\nprofile_path="+profile+"\nindex_sha256="+sha+"\nmodel_fingerprint=12345\nstage_index=0\ndevice_ordinal=0\nfirst_layer=0\nend_layer=12\nn_layers=48\nhc=4\nn_embd=2560\nn_vocab=248320\nmax_cells=2048\nmax_t=8\ncpu_workers=4\nmax_cache_slots=16\nio_timeout_ms=30000\nfree_reserve_bytes=536870912\narena_budget_bytes=0\n";
 }
 void write(const std::string& contents)const {std::ofstream out(config);out<<contents;}
};
Plan plan(){Plan p;p.total_layers=48;p.model_fingerprint=12345;p.geometry={4,2560};p.stages={{Backend::cuda,0,0,12},{Backend::hip,0,12,30},{Backend::hip,1,30,48}};return p;}
}
int main(int,char**){try{
 Fixture f;
 // SHA-256 digest of the exact fixture bytes, confirmed independently via sha256sum.
 const std::string sha="PLACEHOLDER";
 f.write(f.text(sha));CudaWorkerConfig c;std::string err;
 need(!read_cuda_worker_config(f.config,c,err),"bad digest syntax must fail");
 need(c.pack_dir.empty(),"out unchanged on failed parse");
 std::string text=f.text("a"+std::string(63,'0'));
 f.write(text);
 need(read_cuda_worker_config(f.config,c,err),"valid schema parsed");
 need(c.pack_dir==f.pack&&c.profile_path==f.profile,"paths");
 need(c.max_cache_slots==16&&c.cpu_workers==4&&c.max_cells==2048,"bounds");
 need(c.free_reserve_bytes==(512ull<<20),"mem reserve");
 need(!validate_cuda_worker_files(c,err)&&err.find("SHA-256")!=std::string::npos,"digest rejects");
 // Valid SHA digest is injected at configure from the known fixture.
 const std::string expected="7b7894ba64859946e9f6febcfbc1e2470696c076e9585a252d040c6ce96c9861";
 text=f.text(expected);f.write(text);
 need(read_cuda_worker_config(f.config,c,err),"load correct digest");
 need(validate_cuda_worker_files(c,err),"SHA-256 and STRP/v1 check");
 need(validate_cuda_worker_config(c,plan(),0,err),"match plan");
 auto other=plan();other.stages[0].ordinal=1;
 need(!validate_cuda_worker_config(c,other,0,err),"ordinal mismatch");
 other=plan();other.model_fingerprint=9;
 need(!validate_cuda_worker_config(c,other,0,err),"fingerprint mismatch");
 other=plan();other.geometry.hc=5;
 need(!validate_cuda_worker_config(c,other,0,err),"shape mismatch");
 need(!validate_cuda_worker_config(c,plan(),1,err),"index mismatch");
 {std::ofstream out(f.index,std::ios::app);out<<"tampered";}
 need(!validate_cuda_worker_files(c,err),"index mutation rejected");
 {std::ofstream out(f.profile,std::ios::binary);out<<"XXXX";}
 need(!validate_cuda_worker_files(c,err),"mutation still rejected");
 const auto good=f.text(expected);
 for(const std::string& bad:{
    good+"extra=123\n",good+"max_t=3\n",good.substr(0,good.find("version=1\n"))+good.substr(10),
    std::string("version=2\n")+good.substr(10),
    good.substr(0,good.find("max_cells=2048"))+"max_cells=-2"+good.substr(good.find("max_cells=2048")+14),
    good.substr(0,good.find("pack_dir="))+"pack_dir=relative"+good.substr(good.find("pack_dir=")+9+f.pack.size()),
    good.substr(0,good.find("cpu_workers=4"))+"cpu_workers=29"+good.substr(good.find("cpu_workers=4")+13),
    good.substr(0,good.find("free_reserve_bytes=536870912"))+"free_reserve_bytes=7"+good.substr(good.find("free_reserve_bytes=536870912")+27),
    good.substr(0,good.find("max_cache_slots=16"))+"max_cache_slots=0"+good.substr(good.find("max_cache_slots=16")+18)
 }){
    f.write(bad);CudaWorkerConfig unchanged=c;
    need(!read_cuda_worker_config(f.config,unchanged,err),"malformed config rejected");
    need(unchanged.model_fingerprint==c.model_fingerprint,"failed config leaves old object unchanged");
 }
 need(!read_cuda_worker_config("relative-config",c,err),"config must be absolute");
 std::cout<<"M1.16 CUDA worker configuration: PASS ("<<assertions<<" assertions; no GPU)\n";
 return 0;
 }catch(const std::exception& ex){std::cerr<<"M1.16 config FAIL after "<<assertions<<": "<<ex.what()<<'\n';return 1;}}
