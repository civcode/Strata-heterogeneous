#pragma once
// Native runtime only. All callers select the physical owner; these are
// synchronized allocation-boundary samples, NOT a measurement of transient peak.
#include "strata_heterogeneous/long_context.hpp"
#include "strata/prefill/prefill.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <string>

namespace strata::heterogeneous {
inline bool sample_native_capacity(int device, size_t stage, const char* phase,
                                   uint64_t reserve, uint64_t extra, std::string& error) {
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaSetDevice(device) != cudaSuccess ||
        cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess || free_bytes > total_bytes) {
        error = "long-context capacity: cannot query owning device";
        return false;
    }
    const bool fit = capacity_fits(free_bytes, reserve, extra);
    std::fprintf(stderr, "HETERO_CAPACITY stage=%zu phase=%s free_bytes=%llu total_bytes=%llu reserve_bytes=%llu required_extra_bytes=%llu fits=%d\n",
                 stage, phase, static_cast<unsigned long long>(free_bytes),
                 static_cast<unsigned long long>(total_bytes), static_cast<unsigned long long>(reserve),
                 static_cast<unsigned long long>(extra), fit ? 1 : 0);
    if (!fit) error = "long-context capacity: native owner cannot preserve configured VRAM reserve";
    return fit;
}

inline bool check_native_prefill_capacity(int device, size_t stage, const char* phase,
                                          const core::ModelGeometry& g, const core::SessionState& s,
                                          uint32_t chunk, uint64_t reserve, std::string& error) {
    if (!chunk || s.max_cells < 2 || cudaSetDevice(device) != cudaSuccess) {
        error = "long-context capacity: invalid owner or prefill chunk";
        return false;
    }
    const int64_t bounded_chunk = std::min<int64_t>(chunk, s.max_cells - 1);
    const uint64_t extra = prefill::Prefill::bytes_needed_owned(g, s, bounded_chunk, true);
    if (!extra) { error = "long-context capacity: native prefill sizing failed"; return false; }
    return sample_native_capacity(device, stage, phase, reserve, extra, error);
}
} // namespace strata::heterogeneous
