#pragma once
#include "strata/core/verify.hpp"
namespace strata::kernels {
inline void rope_table_release(const float*) {++strata::core::fake_resource::rope_releases;}
}
