#pragma once

#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"
#include "strata_heterogeneous/kv_mode.hpp"
#include <cstdio>
#include <stdexcept>
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
inline void strata_heterogeneous_native_worker_defaults(strata::heterogeneous::KvMode kv) {
    // Set BEFORE native session_bytes/session_init, not after allocation.
    // Use vanilla's native quantized cache; no alternative cache kernels.
    strata::core::qsa_set_kv_int8(kv == strata::heterogeneous::KvMode::int8);
    strata::core::qsa_set_kv_q4(false);
    strata::core::qsa_set_kv_hybrid(false);
    strata::core::qsa_set_kv_int8_rotate(false);
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

// Verify every allocated QSA state rather than assuming the configuration
// selected its requested physical format. No streaming or ring admission.
inline void strata_heterogeneous_verify_kv(const strata::core::SessionState& s,
        strata::heterogeneous::KvMode kv, size_t stage) {
    const bool q8 = kv == strata::heterogeneous::KvMode::int8;
    if (!s.qsa_states || s.qsa_alloc <= 0)
        throw std::runtime_error("native KV allocation missing");
    for (int64_t j=0; j<s.qsa_alloc; ++j) {
        const auto& st = s.qsa_states[s.qsa_ord0+j];
        const bool correct = q8
            ? st.k_q && st.v_q && st.k_scale && st.v_scale && !st.k_pool && !st.v_pool
            : st.k_pool && st.v_pool && !st.k_q && !st.v_q && !st.k_scale && !st.v_scale;
        if (!correct || st.kv_int8 != q8 || st.kv_q4 || st.kv_hybrid ||
            st.kv_rot || st.kv_mode != 0 || st.n_slots != st.n_pages ||
            st.kv_elastic != -1 || st.host.present() || st.max_cells != s.max_cells)
            throw std::runtime_error("native KV allocation format/residency mismatch");
    }
    std::fprintf(stderr,"HETERO_KV stage=%zu format=%s cells=%lld qsa_layers=%lld resident=1\n",
        stage, strata::heterogeneous::kv_mode_name(kv),
        static_cast<long long>(s.max_cells), static_cast<long long>(s.qsa_alloc));
}
