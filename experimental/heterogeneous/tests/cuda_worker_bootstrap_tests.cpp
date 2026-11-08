#include "strata_heterogeneous/cuda_worker_bootstrap.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
using namespace strata::heterogeneous;
namespace {
int assertions=0;
void need(bool ok,const char* msg){++assertions;if(!ok)throw std::runtime_error(msg);}
}
int main(){try{
 const char* argv[]={"worker","--worker","cuda:0:0-12,hip:0:12-30,hip:1:30-48","48","4","2560","12345","0","--config","/tmp/test.cfg",nullptr};
 CudaWorkerInvocation parsed;std::string error;
 need(parse_cuda_worker_invocation(10,argv,parsed,error),"valid launcher syntax");
 need(parsed.plan.total_layers==48&&parsed.plan.stages.size()==3,"plan parsed");
 need(parsed.stage_index==0&&!parsed.shared_memory,"basic flags");
 need(parsed.plan.model_fingerprint==12345&&parsed.config_path=="/tmp/test.cfg","exact identity");
 const char* shared[]={"worker","--worker-shared","cuda:0:0-12,hip:0:12-30,hip:1:30-48","48","4","2560","12345","0","--config","/tmp/test.cfg",nullptr};
 need(parse_cuda_worker_invocation(10,shared,parsed,error)&&parsed.shared_memory,"shared mode");
 need(!parse_cuda_worker_invocation(8,argv,parsed,error),"unconfigured worker refused");
 const char* wrong_backend[]={"worker","--worker","cuda:0:0-12,hip:0:12-30,hip:1:30-48","48","4","2560","12345","1","--config","/tmp/test.cfg",nullptr};
 need(!parse_cuda_worker_invocation(10,wrong_backend,parsed,error),"HIP stage cannot use CUDA worker");
 const char* bad_index[]={"worker","--worker","cuda:0:0-12,hip:0:12-30,hip:1:30-48","48","4","2560","12345","3","--config","/tmp/test.cfg",nullptr};
 need(!parse_cuda_worker_invocation(10,bad_index,parsed,error),"out-of-range index");
 const char* relative[]={"worker","--worker","cuda:0:0-12,hip:0:12-30,hip:1:30-48","48","4","2560","12345","0","--config","relative.cfg",nullptr};
 need(!parse_cuda_worker_invocation(10,relative,parsed,error),"relative config refused");
 const char* extra[]={"worker","--worker","cuda:0:0-12,hip:0:12-30,hip:1:30-48","48","4","2560","12345","0","--else","/tmp/test.cfg",nullptr};
 need(!parse_cuda_worker_invocation(10,extra,parsed,error),"wrong option refused");
 const char* negative[]={"worker","--worker","cuda:0:0-12,hip:0:12-30,hip:1:30-48","48","4","2560","-1","0","--config","/tmp/test.cfg",nullptr};
 need(!parse_cuda_worker_invocation(10,negative,parsed,error),"negative fingerprint refused");
 const char* overflow[]={"worker","--worker","cuda:0:0-12,hip:0:12-30,hip:1:30-48","4294967296","4","2560","12345","0","--config","/tmp/test.cfg",nullptr};
 need(!parse_cuda_worker_invocation(10,overflow,parsed,error),"u32 overflow refused");
 const char* trailing[]={"worker","--worker","cuda:0:0-12,hip:0:12-30,hip:1:30-48","48x","4","2560","12345","0","--config","/tmp/test.cfg",nullptr};
 need(!parse_cuda_worker_invocation(10,trailing,parsed,error),"bad numeric suffix refused");
 std::cout<<"M1.16 CUDA worker process argv: PASS ("<<assertions<<" assertions; no GPU)\n";return 0;
}catch(const std::exception& ex){std::cerr<<"FAIL after "<<assertions<<": "<<ex.what()<<'\n';return 1;}}
