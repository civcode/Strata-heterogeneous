#pragma once
// CPU-only preflight for an independent HIP process. The v2 file schema is
// shared with CUDA, but HIP plan identity is checked separately. No ROCm API.
#include "strata_heterogeneous/cuda_worker_config.hpp"
#include <cstddef>
#include <string>
namespace strata::heterogeneous {
struct HipWorkerConfig {
    CudaWorkerConfig fields; // reused rigorously bounded v2 GGUF/pack schema
};
// Version 1 is deliberately not accepted for new HIP workers.
// Both parsers preserve the previously parsed object on a failure.
bool read_hip_worker_config(const std::string& filename,
                            HipWorkerConfig& out, std::string& error);
bool validate_hip_worker_config(const HipWorkerConfig& config,
                                const Plan& plan, size_t stage_index,
                                std::string& error);
} // namespace strata::heterogeneous
