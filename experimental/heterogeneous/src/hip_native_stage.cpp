#include "strata_heterogeneous/hip_native_stage.hpp"
#if !defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906) || defined(STRATA_GPU_METAL) || defined(STRATA_ENABLE_SYCL)
#error "HIP stage resources are for an isolated wave32 HIP-only build"
#endif
#include "strata/core/weights.hpp"
#include "strata/kernels/ple.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <limits>
#include <set>
#include <stdexcept>

namespace strata::heterogeneous {
namespace {
bool ck(cudaError_t code, const char* action, std::string& err) {
    if (code == cudaSuccess) return true;
    err = std::string("HIP native/PLE: ") + action + ": " + cudaGetErrorString(code);
    return false;
}
bool add(uint64_t a, uint64_t b, uint64_t& result) {
    if (a > std::numeric_limits<uint64_t>::max() - b) return false;
    result = a + b;
    return true;
}
bool belongs(const std::string& name, int64_t lo, int64_t hi) {
    if (name.rfind("blk.", 0) != 0) return false;
    const size_t dot = name.find('.', 4);
    if (dot == std::string::npos) return false;
    const std::string digits = name.substr(4, dot - 4);
    if (digits.empty() || digits[0] < '0' || digits[0] > '9') return false;
    for (char c : digits) if (c < '0' || c > '9') return false;
    // All model layer indices fit in an int; avoid silently accepting a
    // malformed or overflowing GGUF layer name.
    try {
        const long long l = std::stoll(digits);
        return l >= lo && l < hi;
    } catch (...) { return false; }
}
const core::WeightRef* required(const core::WeightTable& table, const char* name,
                                std::string& err) {
    const auto* ref = table.find(name);
    if (!ref || (!ref->data && !ref->native_data)) {
        err = std::string("PLE missing model weight: ") + name;
        return nullptr;
    }
    return ref;
}
} // namespace

bool HipNativeStage::open(HipStageResources& owner, const Plan& plan,
                            size_t index, const HipNativeStageOptions& options,
                            std::string& error) {
    if (attempted_) { error = "native stage is already initialized or poisoned"; return false; }
    attempted_ = true;
    try {
        plan.validate();
        if (!owner.loaded() || owner.native_attached_ || owner.native_ready_ ||
            index >= plan.stages.size() ||
            plan.model_fingerprint != owner.model_fingerprint_ ||
            owner.geometry_.n_layers != plan.total_layers ||
            owner.geometry_.n_embd != plan.geometry.n_embd ||
            owner.geometry_.hc != plan.geometry.hc) {
            error = "native stage requires a matching loaded stage resource owner";
            return false;
        }
        const auto& stage = plan.stages[index];
        needs_ple_ = stage.first_layer <= 1 && stage.end_layer > 1;
        if (stage.backend != Backend::hip || stage.ordinal > INT_MAX ||
            owner.device_ != static_cast<int>(stage.ordinal) ||
            owner.session_.layer_lo != stage.first_layer ||
            owner.session_.layer_hi != stage.end_layer ||
            options.gguf_shards.empty() || options.gguf_shards.size() > 64 ||
            (needs_ple_ && options.ple_gguf_path.empty()) ||
            (!needs_ple_ && !options.ple_gguf_path.empty()) ||
            (needs_ple_ && std::find(options.gguf_shards.begin(), options.gguf_shards.end(),
                                    options.ple_gguf_path) == options.gguf_shards.end())) {
            error = "native stage missing authorized GGUF, PLE table or correct stage identity";
            return false;
        }
        // Reject unreadable/unserved matrix names before allocating native
        // device memory. The core GGUF reader validates tensor metadata.
        std::set<std::string> served;
        if (!core::NativeDense::served_names(options.gguf_shards, true, served, error)) return false;
        if (served.empty()) { error = "native GGUF contains no supported dense projections"; return false; }
        // For the reported first-stage failure, require the exact native QKV
        // projection. A packed/canonical fallback did not serve this tensor.
        if (stage.first_layer == 0 && !served.count("blk.0.attn_qkv.weight")) {
            error = "blk.0.attn_qkv.weight is not served by the authorized native GGUF";
            return false;
        }
        for (const auto& name : served) {
            if (belongs(name, stage.first_layer, stage.end_layer) &&
                owner.weights_.find(name) == nullptr) {
                error = "native dense tensor missing from indexed pack: " + name;
                return false;
            }
        }
        if (!ck(cudaSetDevice(static_cast<int>(stage.ordinal)), "cudaSetDevice", error)) return false;
        device_ = static_cast<int>(stage.ordinal);
        native_ = std::make_unique<core::NativeDense>();
        owner_ = &owner;
        owner.native_attached_ = true;
        if (!core::NativeDense::weight_bytes_for(options.gguf_shards, owner.weights_,
                true, stage.first_layer, stage.end_layer, native_bytes_, error)) {
            shutdown(); return false;
        }
        const uint64_t ple_dev = needs_ple_ ?
            (static_cast<uint64_t>(owner.geometry_.n_embd) * sizeof(float) +
             strata::core::ple_run_scratch_bytes()) : 0;
        uint64_t extra = 0, total = 0;
        if (!add(native_bytes_, ple_dev, extra) ||
            !add(owner.weight_bytes_, owner.session_bytes_, total) ||
            !add(total, extra, total) ||
            (options.total_arena_budget_bytes && total > options.total_arena_budget_bytes)) {
            error = "native dense and PLE allocation exceeds configured arena budget";
            shutdown(); return false;
        }
        size_t free_bytes = 0, total_bytes = 0;
        if (!ck(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo", error) ||
            extra > free_bytes || options.free_reserve_bytes > free_bytes - extra) {
            if (error.empty()) error = "native projections and PLE exceed free VRAM reserve";
            shutdown(); return false;
        }
        if (!native_->load(options.gguf_shards, owner.weights_, error,
                           true, stage.first_layer, stage.end_layer)) {
            shutdown(); return false;
        }
        // Check the GGUF reader's advertised native set against the actual
        // attached WeightRefs. An advert alone is not evidence of an upload.
        for (const auto& name : served) {
            if (!belongs(name, stage.first_layer, stage.end_layer)) continue;
            const auto* ref = owner.weights_.find(name);
            const bool unquantized_ple_key = name == "blk.1.ple_key.weight" &&
                ref != nullptr && !ref->quantized() && ref->data != nullptr;
            if (!ref || (!ref->native_data && !unquantized_ple_key)) {
                error = "native GGUF projection was not attached: " + name;
                shutdown(); return false;
            }
        }
        if (stage.first_layer == 0) {
            const auto* qkv = owner.weights_.find("blk.0.attn_qkv.weight");
            if (!qkv || !qkv->native_data) {
                error = "blk.0.attn_qkv.weight was not bound to native HIP projection";
                shutdown(); return false;
            }
        }
        for (const auto& kv : owner.weights_.all()) {
            const auto& ref = kv.second;
            if (belongs(kv.first, stage.first_layer, stage.end_layer) &&
                !ref.data && !ref.native_data && !ref.hc_q8) {
                error = "native stage has an unserved layer tensor: " + kv.first;
                shutdown(); return false;
            }
        }
        if (needs_ple_ && !setup_ple(options, error)) { shutdown(); return false; }
        if (!ck(cudaDeviceSynchronize(), "native/PLE initialization sync", error)) {
            shutdown(); return false;
        }
        owner.native_ready_ = true;
        ready_ = true;
        error.clear();
        return true;
    } catch (const std::exception& ex) {
        error = ex.what();
        shutdown();
        return false;
    }
}

bool HipNativeStage::setup_ple(const HipNativeStageOptions& options,
                                 std::string& error) {
    auto& session = owner_->session_;
    if (!session.ple_hist) { error = "PLE session history was not carved"; return false; }
    auto& table = owner_->weights_;
    const auto* key = required(table, "blk.1.ple_key.weight", error);
    const auto* value = required(table, "blk.1.ple_value.weight", error);
    const auto* nk = required(table, "blk.1.ple_norm_key.weight", error);
    const auto* nq = required(table, "blk.1.ple_norm_query.weight", error);
    const auto* nc = required(table, "blk.1.ple_norm_conv.weight", error);
    const auto* conv = required(table, "blk.1.ple_conv1d.weight", error);
    if (!key || !value || !nk || !nq || !nc || !conv) return false;
    // This model's PLE activation/norm kernels have exact element types;
    // never reinterpret a quantized or packed alternative as a BF16/F32.
    const bool conv_f16 = conv->kind == core::WeightKind::F16InF32 ||
        (conv->kind == core::WeightKind::Verbatim && conv->code_bits == 0);
    if (value->quantized() || !value->data || value->elements <= 0 ||
        value->bytes != static_cast<uint64_t>(value->elements) * 2 ||
        nk->kind != core::WeightKind::F32 || nq->kind != core::WeightKind::F32 ||
        nc->kind != core::WeightKind::F32 || !nk->data || !nq->data || !nc->data ||
        !conv_f16 || !conv->data || conv->elements <= 0 ||
        conv->bytes != static_cast<uint64_t>(conv->elements) * 2 ||
        (!key->native_data && !key->data)) {
        error = "PLE weight format is unsupported by the canonical HIP PLE path";
        return false;
    }
    ple_table_ = std::make_unique<strata::kernels::PleTable>();
    strata::kernels::PleIoOptions io;
    io.mode = strata::kernels::PleIo::Direct;
    io.cache_rows = 0; // no large implicit RAM allocation in M1.17
    io.keepalive_ms = 0;
    if (!ple_table_->open(options.ple_gguf_path, error, io)) return false;
    const uint64_t emb_bytes = static_cast<uint64_t>(owner_->geometry_.n_embd) * sizeof(float);
    const uint64_t scratch_bytes = core::ple_run_scratch_bytes();
    if (!ck(cudaMalloc(reinterpret_cast<void**>(&ple_emb_dev_), static_cast<size_t>(emb_bytes)),
            "PLE device embedding", error) ||
        !ck(cudaMallocHost(reinterpret_cast<void**>(&ple_emb_host_), static_cast<size_t>(emb_bytes)),
            "PLE pinned embedding", error) ||
        !ck(cudaMalloc(reinterpret_cast<void**>(&ple_scratch_), static_cast<size_t>(scratch_bytes)),
            "PLE graph scratch", error)) return false;
    auto& p = session.ple;
    p.table = ple_table_.get();
    p.token = &session.ple_token;
    p.prev = session.ple_prev;
    p.hist = session.ple_hist;
    p.emb_dev = ple_emb_dev_;
    p.emb_host = ple_emb_host_;
    p.scratch = ple_scratch_;
    p.consts = strata::kernels::ple_artifact_consts();
    if (key->native_data && key->quantized()) {
        if (!key->native_q8_1 ||
            (key->native_type != 42 && key->native_type != 18 &&
             key->native_type != 23 && key->native_type != 8)) {
            error = "native PLE key has an incompatible type or missing Q8 scratch";
            return false;
        }
        p.w.key_native_data = key->native_data;
        p.w.key_native_type = key->native_type;
        p.w.key_native_q8_1 = key->native_q8_1;
    }
    if (!key->quantized() && key->data) p.w.key_bf16 = static_cast<const uint16_t*>(key->data);
    if (key->quantized() && key->data && key->code_bits == 2 && key->codes_bytes > 0 &&
        key->scales_bytes > 0) {
        p.w.key_codes = static_cast<const uint8_t*>(key->data);
        p.w.key_scales = reinterpret_cast<const float*>(
            static_cast<const uint8_t*>(key->data) + key->codes_bytes);
    }
    p.w.value_bf16 = static_cast<const uint16_t*>(value->data);
    p.w.norm_key = static_cast<const float*>(nk->data);
    p.w.norm_query = static_cast<const float*>(nq->data);
    p.w.norm_conv = static_cast<const float*>(nc->data);
    p.w.conv1d_f16 = static_cast<const uint16_t*>(conv->data);
    if (key->quantized() && !key->native_data &&
        (key->code_bits != 2 || !key->data || !key->codes_bytes || !key->scales_bytes)) {
        error = "PLE key has neither a valid S2 pack representation nor native GGUF projection";
        return false;
    }
    if (!p.ready()) { error = "canonical PLE was not fully configured"; return false; }
    return true;
}

void HipNativeStage::shutdown() noexcept {
    if (!owner_) return;
    ready_ = false;
    owner_->native_ready_ = false;
    // The verifier must be destroyed before this call. When the driver
    // cannot prove completion, do not free anything a graph may still use.
    if (device_ >= 0 && (cudaSetDevice(device_) != cudaSuccess ||
                          cudaDeviceSynchronize() != cudaSuccess)) std::_Exit(70);
    owner_->session_.ple = {};
    if (ple_scratch_ && cudaFree(ple_scratch_) != cudaSuccess) std::_Exit(70);
    ple_scratch_ = nullptr;
    if (ple_emb_dev_ && cudaFree(ple_emb_dev_) != cudaSuccess) std::_Exit(70);
    ple_emb_dev_ = nullptr;
    if (ple_emb_host_ && cudaFreeHost(ple_emb_host_) != cudaSuccess) std::_Exit(70);
    ple_emb_host_ = nullptr;
    ple_table_.reset();
    native_.reset();
    owner_->native_attached_ = false;
    owner_ = nullptr;
    device_ = -1;
}
} // namespace strata::heterogeneous
