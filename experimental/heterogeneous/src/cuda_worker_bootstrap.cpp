#include "strata_heterogeneous/cuda_worker_bootstrap.hpp"
#include <charconv>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace strata::heterogeneous {
namespace {
uint64_t number(const char* str) {
    if(!str)throw std::invalid_argument("missing CUDA worker CLI number");
    const std::string s(str);
    if(s.empty()||s.front()=='-'||s.front()=='+')throw std::invalid_argument("invalid CUDA worker CLI number");
    uint64_t n=0;const auto end=std::from_chars(s.data(),s.data()+s.size(),n);
    if(end.ec!=std::errc{}||end.ptr!=s.data()+s.size())throw std::invalid_argument("invalid CUDA worker CLI number");
    return n;
}
uint32_t as_u32(const char* s) {
    const uint64_t n=number(s);
    if(n>std::numeric_limits<uint32_t>::max())throw std::invalid_argument("CUDA worker CLI number exceeds u32");
    return static_cast<uint32_t>(n);
}
}
bool parse_cuda_worker_invocation(int argc,const char* const* argv,
                                  CudaWorkerInvocation& out,std::string& error) {
    try {
        if(argc!=10||!argv||!argv[1]||!argv[8]||!argv[9]||
           (std::string(argv[1])!="--worker"&&std::string(argv[1])!="--worker-shared")||
           std::string(argv[8])!="--config")
            throw std::invalid_argument("expected --worker[/-shared] plan layers hc n_embd fingerprint stage_index --config /absolute/path");
        CudaWorkerInvocation w;
        w.shared_memory=std::string(argv[1])=="--worker-shared";
        w.plan=parse_plan(argv[2],as_u32(argv[3]),{as_u32(argv[4]),as_u32(argv[5])},number(argv[6]));
        w.stage_index=as_u32(argv[7]);
        if(w.stage_index>=w.plan.stages.size()||w.plan.stages[w.stage_index].backend!=Backend::cuda)
            throw std::invalid_argument("CUDA worker stage index/backend mismatch");
        w.config_path=argv[9];
        if(w.config_path.empty()||w.config_path[0]!='/')throw std::invalid_argument("CUDA worker config path must be absolute");
        out=std::move(w);error.clear();return true;
    }catch(const std::exception& ex){error=ex.what();return false;}
}
} // namespace strata::heterogeneous
