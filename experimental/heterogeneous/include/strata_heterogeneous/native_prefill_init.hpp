#pragma once

// Include only in a native CUDA or HIP worker, never in the neutral coordinator.
#include "strata_heterogeneous/native_prefill_resources.hpp"
#include "strata/prefill/prefill.hpp"
#include <cuda_runtime.h>
#include <limits>

namespace strata::heterogeneous {

// Prepare vanilla Prefill with an already-loaded stage's native resources.
// Caller configures set_stage/set_external_handoff first, supplies a local
// owned stream and owns lifetime; no concurrent decode or cache mutation.
// No expert-slot loans: refuse a chunk that cannot fit the existing reserve.
// A false return after init was attempted requires the caller to drain and
// reset/destroy Prefill on its owning device before retry or owner teardown.
inline bool init_owned_native_prefill(prefill::Prefill& pf, const core::WeightTable& weights,
                                      const core::ModelGeometry& g, core::SessionState& session,
                                      NativePrefillExpertView experts, int64_t chunk, void* stream,
                                      uint64_t reserve_bytes, uint64_t allocation_budget,
                                      std::string& error) {
    error.clear();
    if (!experts || !stream || chunk <= 0 || chunk > session.max_cells ||
        g.n_layers <= 0 || g.n_expert <= 0 ||
        static_cast<uint64_t>(g.n_layers) > std::numeric_limits<size_t>::max() / static_cast<uint64_t>(g.n_expert) ||
        experts.residency_entries != static_cast<size_t>(g.n_layers) * static_cast<size_t>(g.n_expert) ||
        experts.first_layer != session.layer_lo || experts.end_layer != session.layer_hi ||
        experts.first_layer < 0 || experts.end_layer <= experts.first_layer || experts.end_layer > g.n_layers) {
        error = "native prefill: invalid resource identity or capacity";
        return false;
    }
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess || device != experts.device) {
        error = "native prefill: stage's owning device must be current";
        return false;
    }
    size_t free_before = 0, total = 0;
    if (cudaMemGetInfo(&free_before, &total) != cudaSuccess) {
        error = "native prefill: free-memory query failed";
        return false;
    }
    const uint64_t bytes = prefill::Prefill::bytes_needed_owned(g, session, chunk, true);
    if (!bytes || bytes > free_before || reserve_bytes > free_before - bytes ||
        (allocation_budget && bytes > allocation_budget)) {
        error = "native prefill: owned prompt buffers do not fit the existing budget/reserve";
        return false;
    }
    if (!pf.init(weights, g, session, experts.source, experts.cache, experts.host_res,
                 chunk, stream, error)) return false;
    size_t free_after = 0;
    if (cudaMemGetInfo(&free_after, &total) != cudaSuccess || free_after < reserve_bytes ||
        (allocation_budget && free_before > free_after && free_before - free_after > allocation_budget)) {
        error = "native prefill: actual allocation crossed budget/reserve; reset before retry";
        return false;
    }
    return true;
}

} // namespace strata::heterogeneous
