#pragma once
#include "strata_heterogeneous/host.hpp"
#include "strata/core/verify.hpp"
#include <cstdint>
#include <cstddef>
namespace strata::heterogeneous {
struct HipVerifierBindings {
    const strata::core::WeightTable& weights;
    const strata::core::ModelGeometry& geometry;
    strata::core::SessionState& session;
    strata::core::VerifyHits hits;
    void(*pool)(void*,const float*,const int32_t*,int64_t,int64_t,float*,int64_t);
    void* pool_user;
    uint64_t model_fingerprint;
    int max_t, timeout_ms;
    int64_t initial_position;
};
}
