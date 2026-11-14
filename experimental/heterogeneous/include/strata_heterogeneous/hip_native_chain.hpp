#pragma once
// Native same-runtime chain assembly only; no new scheduler or kernels.
#include "strata_heterogeneous/backend_domains.hpp"
#include "strata_heterogeneous/hip_verifier.hpp"
#include <cuda_runtime.h>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace strata::heterogeneous {
struct HipNativeChainStage {
    HipVerifierBindings bindings;
    uint64_t reserve_bytes;
};
class HipNativeVerifierChain final {
public:
    HipNativeVerifierChain() = default;
    ~HipNativeVerifierChain() noexcept { shutdown(); }
    HipNativeVerifierChain(const HipNativeVerifierChain&) = delete;
    HipNativeVerifierChain& operator=(const HipNativeVerifierChain&) = delete;
    bool prepare(const Plan& plan, const std::vector<HipNativeChainStage>& stages,
                 const float* input, const core::NativeHead* head, std::string& error) {
        if (attempted_) { error = "HIP chain already attempted"; return false; }
        attempted_ = true;
        try {
            const auto domains = native_backend_domains(plan);
            const auto& domain = domains.back();
            if (domain.backend != Backend::hip || domain.size() != stages.size() || stages.empty() ||
                (plan.stages[domain.first_stage].first_layer != 0) != (input != nullptr)) {
                error = "HIP chain requires complete terminal native domain and matching input";
                return false;
            }
            const auto& first = stages.front().bindings;
            if (first.max_t < 1 || first.max_t > strata::kernels::kVerifyMaxT || !first.pool ||
                first.timeout_ms <= 0 || first.initial_position < 0) {
                error = "HIP chain invalid native window limits"; return false;
            }
            const uint64_t stride = plan.geometry.floats_per_row();
            if (stride > std::numeric_limits<size_t>::max() / sizeof(float) / static_cast<size_t>(first.max_t)) {
                error = "HIP chain activation size overflow"; return false;
            }
            const size_t bytes = static_cast<size_t>(stride) * sizeof(float) * static_cast<size_t>(first.max_t);
            // All physical identities and full resident maps validated before any allocation.
            for (size_t j = 0; j < stages.size(); ++j) {
                const auto& p = plan.stages[domain.first_stage + j];
                const auto& b = stages[j].bindings;
                const auto& g = b.geometry;
                const auto& s = b.session;
                if (p.ordinal > INT_MAX || b.model_fingerprint != plan.model_fingerprint ||
                    g.n_layers != plan.total_layers || g.hc != plan.geometry.hc || g.n_embd != plan.geometry.n_embd ||
                    g.n_vocab != first.geometry.n_vocab || g.n_vocab <= 0 ||
                    g.n_expert != first.geometry.n_expert || g.n_expert <= 0 ||
                    s.layer_lo != p.first_layer || s.layer_hi != p.end_layer || s.max_cells <= 0 ||
                    s.max_cells != first.session.max_cells || b.initial_position != first.initial_position ||
                    b.initial_position > s.max_cells || b.max_t != first.max_t || b.pool != first.pool ||
                    !b.pool_user || b.timeout_ms != first.timeout_ms || !b.hits.h_res || !b.hits.d_res ||
                    !b.hits.cache_base || b.hits.n_slots <= 0) {
                    error = "HIP chain stage identity, geometry, pool or limits mismatch"; return false;
                }
                for (int64_t l = p.first_layer; l < p.end_layer; ++l)
                    for (int64_t e = 0; e < g.n_expert; ++e) {
                        const int32_t slot = b.hits.h_res[l * g.n_expert + e];
                        if (slot < 0 || slot >= b.hits.n_slots) {
                            error = "HIP chain needs full verified resident stage"; return false;
                        }
                    }
            }
            nodes_.reserve(stages.size());
            hands_.resize(stages.size() - 1);
            // Vanilla mapped portable pinned handoff, with per-device aliases.
            for (size_t j = 0; j < hands_.size(); ++j) {
                auto& h = hands_[j];
                h.owner = static_cast<int>(plan.stages[domain.first_stage + j].ordinal);
                const int receiver = static_cast<int>(plan.stages[domain.first_stage + j + 1].ordinal);
                if (!ck(cudaSetDevice(h.owner), error) ||
                    !ck(cudaHostAlloc(&h.host, bytes, cudaHostAllocMapped | cudaHostAllocPortable), error) ||
                    !ck(cudaHostGetDevicePointer(reinterpret_cast<void**>(&h.writer), h.host, 0), error) ||
                    !ck(cudaSetDevice(receiver), error) ||
                    !ck(cudaHostGetDevicePointer(reinterpret_cast<void**>(&h.reader), h.host, 0), error)) return false;
                std::memset(h.host, 0, bytes);
            }
            for (size_t j = 0; j < stages.size(); ++j) {
                const auto& p = plan.stages[domain.first_stage + j];
                const auto& b = stages[j].bindings;
                const int dev = static_cast<int>(p.ordinal);
                if (!ck(cudaSetDevice(dev), error)) return false;
                nodes_.push_back({dev, std::make_unique<core::Verifier>()});
                auto& v = *nodes_.back().verifier;
                v.set_stage(p.first_layer, p.end_layer, j ? hands_[j-1].reader : input,
                            j < hands_.size() ? hands_[j].writer : nullptr);
                v.set_handoff_rows(b.max_t);
                // Legacy serial transactional control, not speculative scheduling.
                v.set_always_publish(true);
                v.set_head_sampling(j + 1 == stages.size());
                if (!v.init(b.weights, b.geometry, b.session, b.hits,
                            j + 1 == stages.size() ? head : nullptr, b.max_t, error)) return false;
                size_t free_bytes = 0, total_bytes = 0;
                if (!ck(cudaMemGetInfo(&free_bytes, &total_bytes), error) || free_bytes < stages[j].reserve_bytes) {
                    if (error.empty()) error = "HIP chain verifier exceeds existing device reserve";
                    return false;
                }
            }
            // Vanilla run()/commit()/wait_commit() recurse through set_next().
            for (size_t j = 0; j + 1 < nodes_.size(); ++j)
                nodes_[j].verifier->set_next(nodes_[j+1].verifier.get(), stages[j+1].bindings.pool_user);
            ready_ = true;
            return true;
        } catch (const std::exception& ex) { error = ex.what(); return false; }
    }
    core::Verifier& at(size_t local_stage) {
        if (!ready_) throw std::logic_error("HIP chain not ready");
        return *nodes_.at(local_stage).verifier;
    }
    core::Verifier& first() { return at(0); }
    core::Verifier& last() { return at(nodes_.empty() ? 0 : nodes_.size() - 1); }
    bool ready() const noexcept { return ready_; }
    // No producer/consumer spin-flag overlap is enabled for this control.
    // On a failed window, release every native wait before freeing anything.
    void abort(int timeout_ms) noexcept {
        for (const auto& n : nodes_)
            if (cudaSetDevice(n.device) != cudaSuccess || !n.verifier->release_gpu_waits(timeout_ms) ||
                cudaStreamQuery(n.verifier->stream()) != cudaSuccess) std::_Exit(70);
    }
    void shutdown() noexcept {
        ready_ = false;
        for (const auto& n : nodes_)
            if (cudaSetDevice(n.device) != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) std::_Exit(70);
        for (auto& n : nodes_) {
            if (cudaSetDevice(n.device) != cudaSuccess) std::_Exit(70);
            n.verifier.reset();
        }
        nodes_.clear();
        for (auto& h : hands_)
            if (h.host && (cudaSetDevice(h.owner) != cudaSuccess || cudaFreeHost(h.host) != cudaSuccess)) std::_Exit(70);
        hands_.clear();
    }
private:
    struct Node { int device; std::unique_ptr<core::Verifier> verifier; };
    struct Handoff { int owner = -1; void* host = nullptr; float* writer = nullptr; float* reader = nullptr; };
    std::vector<Node> nodes_;
    std::vector<Handoff> hands_;
    bool attempted_ = false, ready_ = false;
    static bool ck(cudaError_t status, std::string& error) {
        if (status == cudaSuccess) return true;
        error = std::string("HIP chain: ") + cudaGetErrorString(status); return false;
    }
};
} // namespace strata::heterogeneous
