#pragma once

// CPU-only stage selection for Strata's native pack/index.txt format.
// The resulting skip set is passed unmodified to core::WeightTable::pool_bytes
// and core::WeightTable::load. No CUDA/HIP runtime is initialized here.
#include "strata_heterogeneous/host.hpp"
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>

namespace strata::heterogeneous {

struct StageWeightPlan {
    std::set<std::string> skip;  // exact index names outside this stage's layer range
    uint64_t full_pool_bytes = 0;      // native pack header (not a GPU-memory budget)
    uint64_t selected_pool_bytes = 0;  // compact arena size, same algorithm as WeightTable::pool_bytes(skip)
    size_t total_tensors = 0;
    size_t selected_tensors = 0;
    size_t stage_layer_tensors = 0;
    // Global/non-blk tensors are kept deliberately: deciding whether an
    // embedding, final head or native representation can be omitted needs a
    // real loader and verifier check, not a guess based on its name.
};

// Parses the real 19-column tools/pack_index.py format and the '# align ...
// pool ... tensors ...' header; refuses malformed/duplicate rows, bad layer
// names, inconsistent offsets/counts and integer overflows. Requires that
// every owned layer has at least one index row. On failure out is unchanged.
// Does NOT load weights, construct model state, authenticate a pack, or infer
// a fingerprint. All unrecognized/non-layer names remain selected.
bool plan_stage_weights(const Plan& plan, size_t stage_index,
                        const std::string& index_path,
                        StageWeightPlan& out, std::string& error);

} // namespace strata::heterogeneous
