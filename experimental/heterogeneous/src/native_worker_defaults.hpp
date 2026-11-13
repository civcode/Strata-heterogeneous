#pragma once

#include "strata/core/layer.hpp"
#include "strata/kernels/gr.hpp"
#include "strata/kernels/ple.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_rope.hpp"

// Match generate.cpp's canonical --native kernel configuration before any
// session or verifier captures. These settings are process-local to workers.
inline void strata_heterogeneous_native_worker_defaults() {
    strata::kernels::gr_set_native_mmvf(true);
    strata::core::layer_set_fast_attn(true);
    strata::core::layer_set_fused_gdn(true);
    strata::core::layer_set_fast_select(true);
    strata::core::layer_set_fused_gr(true);
    strata::core::layer_set_native_bf16(true);
    strata::core::layer_set_native_flash_attn_short(false);
    strata::kernels::ple_set_native_bf16(true);
    strata::kernels::shared_expert_set_native_bf16(true);
    strata::kernels::native_moe_combine_set_enabled(true);
    strata::kernels::native_gdn_set_enabled(true);
    strata::kernels::native_router_set_enabled(true);
    strata::kernels::native_qsa_set_enabled(true);
    strata::kernels::native_qsa_indexer_set_enabled(true);
    strata::kernels::native_rope_set_enabled(true);
    strata::kernels::ple_set_native_postops(true);
}
