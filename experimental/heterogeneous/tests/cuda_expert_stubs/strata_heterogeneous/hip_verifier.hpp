#pragma once
#include "strata/core/verify.hpp"
#include <cstdint>
namespace strata::heterogeneous {
struct HipVerifierBindings {
 const strata::core::WeightTable& weights;
 const strata::core::ModelGeometry& geometry;
 strata::core::SessionState& session;
 strata::core::VerifyHits hits;
 strata::core::PoolMultiFn pool;
 void* pool_user;
 uint64_t model_fingerprint;
 int max_t=8, timeout_ms=30000;int64_t initial_position=0;
 void (*zero_session)(strata::core::SessionState&,
                      const strata::core::ModelGeometry&, void*) = nullptr;
};
}
namespace strata::kernels { constexpr int kVerifyMaxT=8; }
