#pragma once
// Runtime-free parser for the separate HIP subprocess invocation.
#include "strata_heterogeneous/host.hpp"
#include <cstddef>
#include <string>
namespace strata::heterogeneous {
struct HipWorkerInvocation {
    Plan plan;
    size_t stage_index = 0;
    bool shared_memory = false;
    std::string config_path;
};
// Exact WorkerProcessGroup argv; require a HIP stage and safe absolute config.
// Never loads ROCm, the model or config bytes. Leaves output unchanged on error.
bool parse_hip_worker_invocation(int argc, const char* const* argv,
                                 HipWorkerInvocation& out, std::string& error);
} // namespace strata::heterogeneous
