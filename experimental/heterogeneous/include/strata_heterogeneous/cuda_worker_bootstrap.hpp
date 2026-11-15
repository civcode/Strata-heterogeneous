#pragma once
// CPU-only parser for the process supervisor's CUDA worker argv convention.
// No CUDA include; a supervisor and CPU tests may use this without GPUs.
#include "strata_heterogeneous/host.hpp"
#include "strata_heterogeneous/kv_mode.hpp"
#include <cstddef>
#include <string>

namespace strata::heterogeneous {
struct CudaWorkerInvocation {
    Plan plan;
    size_t stage_index = 0;
    bool shared_memory = false;
    // Legacy internal argv without --kv retains its historical FP16 contract.
    KvMode kv = KvMode::f16;
    std::string config_path;
};
// Accepts ONLY the WorkerProcessGroup convention with explicit --config:
//   --worker[/-shared] <plan> <layers> <hc> <n_embd> <fp> <index> --config <abs_path>
// No GPU use and no file I/O. Output unchanged on error.
bool parse_cuda_worker_invocation(int argc,const char* const* argv,
                                  CudaWorkerInvocation& out,std::string& error);
} // namespace strata::heterogeneous
