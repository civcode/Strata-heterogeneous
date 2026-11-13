#include "strata_heterogeneous/hip_file_experts.hpp"
#if !defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906) || defined(STRATA_GPU_METAL) || defined(STRATA_ENABLE_SYCL)
#error "HIP stage resources are for an isolated wave32 HIP-only build"
#endif

#include <cuda_runtime.h>

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

namespace strata::heterogeneous {
namespace {

bool cuda_ok(cudaError_t result, const char* op, std::string& error) {
    if (result == cudaSuccess) return true;
    error = std::string("HIP expert provider: ") + op + ": " + cudaGetErrorString(result);
    return false;
}

// The CPU callback is part of the verifier's GPU/host doorbell loop. Once
// the source/pool cannot produce a real expert, returning would release the
// GPU to consume stale/partial outputs. Terminate this worker and let its
// supervisor reap/rebuild it. This intentionally skips unsafe teardown.
[[noreturn]] void fatal_pool(const char* what, int64_t layer, int64_t expert) noexcept {
    std::fprintf(stderr, "strata HIP worker: fatal expert pool at layer %lld, expert %lld: %s\n",
                 static_cast<long long>(layer), static_cast<long long>(expert),
                 what != nullptr ? what : "unknown expert failure");
    std::fflush(stderr);
    std::_Exit(75);
}

} // namespace

bool HipFileExperts::open(const Plan& plan, size_t stage_index,
                           const core::ModelGeometry& geometry,
                           const HipFileExpertOptions& options, std::string& error) {
    if (state_ != State::fresh) {
        error = "HIP expert provider is already initialized or poisoned";
        return false;
    }
    try {
        plan.validate();
        if (stage_index >= plan.stages.size()) {
            error = "HIP expert provider stage index is out of range";
            return false;
        }
        const auto& stage = plan.stages[stage_index];
        if (stage.backend != Backend::hip || stage.ordinal > INT_MAX ||
            stage.first_layer >= stage.end_layer ||
            geometry.n_layers != plan.total_layers || geometry.hc != plan.geometry.hc ||
            geometry.n_embd != plan.geometry.n_embd ||
            geometry.n_embd != strata::kernels::cpu::H ||
            geometry.n_expert != strata::kernels::cpu::NE ||
            geometry.n_layers <= 0 || geometry.n_layers > INT_MAX ||
            options.pack_dir.empty() || options.cpu_workers < 1 || options.cpu_workers > 28 ||
            options.max_cache_slots < 0 ||
            options.profile_path.empty() == options.ranked_pairs.empty()) {
            error = "HIP expert provider requires one canonical pack, one profile source and bounded workers";
            return false;
        }
        std::vector<std::pair<int32_t, int32_t>> ranked;
        if (!options.profile_path.empty()) {
            int64_t profile_slots = 0;
            if (!core::read_expert_profile(options.profile_path, geometry.n_layers,
                                           geometry.n_expert, ranked, profile_slots, error)) return false;
            if (profile_slots < 1 || static_cast<int64_t>(ranked.size()) > profile_slots) {
                error = "expert profile has no usable cache capacity";
                return false;
            }
        } else {
            ranked = options.ranked_pairs;
        }
        std::set<std::pair<int32_t, int32_t>> seen;
        std::vector<std::pair<int32_t, int32_t>> selected;
        for (const auto& pair : ranked) {
            const int64_t l = pair.first, e = pair.second;
            if (l < 0 || l >= geometry.n_layers || e < 0 || e >= geometry.n_expert ||
                !seen.insert(pair).second) {
                error = "expert profile contains an invalid or duplicate (layer, expert) pair";
                return false;
            }
            if (l >= stage.first_layer && l < stage.end_layer &&
                (!options.max_cache_slots ||
                 selected.size() < static_cast<size_t>(options.max_cache_slots)))
                selected.push_back(pair);
        }
        if (selected.empty() || selected.size() > static_cast<size_t>(INT_MAX) ||
            static_cast<uint64_t>(geometry.n_layers) >
                std::numeric_limits<size_t>::max() / static_cast<uint64_t>(geometry.n_expert)) {
            error = "expert profile has no bounded resident expert for this HIP stage";
            return false;
        }

        // The layout is process-wide: never reuse this worker for another
        // model after initialization. Native/IQ expert blobs require their
        // own per-layer formats and are explicitly outside M1.15.
        if (!strata::kernels::cpu::expert_layout_load(options.pack_dir,
                geometry.n_layers, geometry.n_expert, error)) return false;
        const auto& layout = strata::kernels::cpu::expert_layout();
        if (layout.native || layout.n_expert != geometry.n_expert ||
            layout.blob_bytes(stage.first_layer) == 0 ||
            layout.blob_bytes(stage.first_layer) > static_cast<uint64_t>(INT64_MAX)) {
            error = "HIP expert provider only supports canonical Q2_0 blobs";
            return false;
        }
        blob_bytes_ = static_cast<int64_t>(layout.blob_bytes(stage.first_layer));
        n_expert_ = geometry.n_expert;
        first_ = stage.first_layer;
        end_ = stage.end_layer;

        // FileExpertSource checks that experts.bin holds every model layer
        // and every expert, not just the few profiled cache pairs.
        if (!source_.open(options.pack_dir, geometry.n_layers,
                          geometry.n_expert, error)) return false;

        state_ = State::initializing;
        if (!cuda_ok(cudaSetDevice(static_cast<int>(stage.ordinal)), "cudaSetDevice", error)) {
            shutdown(); return false;
        }
        device_ = static_cast<int>(stage.ordinal);
        const uint64_t slot_count = static_cast<uint64_t>(selected.size());
        const uint64_t width = static_cast<uint64_t>(blob_bytes_);
        const uint64_t table_entries = static_cast<uint64_t>(geometry.n_layers) *
                                       static_cast<uint64_t>(geometry.n_expert);
        if (slot_count > std::numeric_limits<uint64_t>::max() / width ||
            table_entries > std::numeric_limits<uint64_t>::max() / sizeof(int32_t) ||
            slot_count * width > std::numeric_limits<uint64_t>::max() -
                                 table_entries * sizeof(int32_t)) {
            error = "expert cache and residency size overflow";
            shutdown(); return false;
        }
        const uint64_t need = slot_count * width + table_entries * sizeof(int32_t);
        size_t free_bytes = 0, total_bytes = 0;
        if (!cuda_ok(cudaMemGetInfo(&free_bytes, &total_bytes),
                     "cache free-memory preflight", error) ||
            need > free_bytes || options.free_reserve_bytes > free_bytes - need) {
            if (error.empty()) error = "expert cache exceeds free HIP memory or reserve";
            shutdown(); return false;
        }
        if (!cache_.open(static_cast<int64_t>(selected.size()), geometry.n_layers,
                          geometry.n_expert, blob_bytes_, error)) {
            shutdown(); return false;
        }
        host_res_.assign(static_cast<size_t>(geometry.n_layers) *
                             static_cast<size_t>(geometry.n_expert), -1);
        for (const auto& pair : selected) {
            const int32_t slot = cache_.admit(pair.first, pair.second);
            if (slot < 0 || slot >= cache_.slots()) {
                error = "expert cache refused a validated profile slot";
                shutdown(); return false;
            }
            // A real file-mapped blob; never fill the cache with a sentinel.
            const uint8_t* blob = source_.blob(pair.first, pair.second);
            if (blob == nullptr ||
                !cache_.fill_slot_blocking(slot, blob, error, blob_bytes_) ||
                !cache_.verify_slot(slot, blob, error, blob_bytes_)) {
                if (error.empty()) error = "expert file blob missing or cache verification failed";
                shutdown(); return false;
            }
            host_res_[static_cast<size_t>(pair.first) *
                      static_cast<size_t>(n_expert_) + static_cast<size_t>(pair.second)] = slot;
            ++filled_slots_;
        }
        const size_t table_bytes = host_res_.size() * sizeof(int32_t);
        if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&device_res_), table_bytes),
                     "residency table allocation", error) ||
            !cuda_ok(cudaMemcpy(device_res_, host_res_.data(), table_bytes, cudaMemcpyHostToDevice),
                     "residency table upload", error) ||
            !cuda_ok(cudaDeviceSynchronize(), "cache/residency completion", error)) {
            shutdown(); return false;
        }

        // All non-resident experts are computed by the actual CPU pool. A
        // resident row is zeroed by the native dispatch; the verifier's GPU
        // planner computes it from the matching verified slot.
        pool_ = std::make_unique<strata::kernels::cpu::ExpertPool>(options.cpu_workers, false, true);
        dispatch_.src = &source_;
        dispatch_.pool = pool_.get();
        dispatch_.n_expert = geometry.n_expert;
        dispatch_.host_res = host_res_.data();
        dispatch_.cache_base = cache_.device_slot(0);
        dispatch_.cache_blob = blob_bytes_;
        dispatch_.cache_slot_off = nullptr;
        dispatch_.plan = nullptr; // no optional PCIe share or adaptive writes
        dispatch_.layers = stage.first_layer;
        core::set_tail_skip_rank(0); // exact expert arithmetic, never drop tail routes
        state_ = State::ready;
        error.clear();
        return true;
    } catch (const std::exception& ex) {
        error = ex.what();
        if (state_ == State::initializing) shutdown();
        return false;
    }
}

core::VerifyHits HipFileExperts::hits() const {
    if (!ready() || device_res_ == nullptr || host_res_.empty() ||
        filled_slots_ == 0 || cache_.device_slot(0) == nullptr)
        throw std::logic_error("HIP expert residency is not prepared");
    core::VerifyHits result;
    result.d_res = device_res_;
    result.h_res = host_res_.data();
    result.cache_base = cache_.device_slot(0);
    result.n_slots = cache_.slots();
    result.blob = blob_bytes_;
    return result;
}

void HipFileExperts::run(const float* x_f, const int32_t* ids,
                          int64_t n_tok, int64_t k, float* out, int64_t layer) {
    if (!ready() || x_f == nullptr || ids == nullptr || out == nullptr ||
        layer < first_ || layer >= end_ || n_tok < 1 ||
        n_tok > strata::kernels::kVerifyMaxT || k != 10 ||
        n_tok > std::numeric_limits<int64_t>::max() / k ||
        dispatch_.failed)
        fatal_pool("invalid window, stage or poisoned dispatch", layer, -1);
    for (int64_t i = 0; i < n_tok * k; ++i)
        if (ids[i] < 0 || ids[i] >= n_expert_)
            fatal_pool("routed expert id out of range", layer, ids[i]);
    // The core dispatcher increments 'layers' internally; align it to the
    // verifier's explicit layer argument for every token group/window.
    dispatch_.layers = layer;
    try {
        core::expert_pool_dispatch_multi(dispatch_, x_f, ids, n_tok, k, out);
    } catch (const std::exception& ex) {
        fatal_pool(ex.what(), layer, -1);
    } catch (...) {
        fatal_pool("unhandled exception in native expert pool", layer, -1);
    }
    if (dispatch_.failed)
        fatal_pool(dispatch_.fail, dispatch_.fail_layer, dispatch_.fail_expert);
}

void HipFileExperts::shutdown() noexcept {
    if (state_ == State::dead) return;
    state_ = State::dead;
    // The verifier must be destroyed BEFORE this provider. If the GPU is
    // unreachable or still has uncompleted work, never free its resident
    // slots or the host mmap while captured kernels may still reference them.
    if (device_ >= 0 &&
        (cudaSetDevice(device_) != cudaSuccess ||
         cudaDeviceSynchronize() != cudaSuccess)) std::_Exit(75);
    pool_.reset();
    dispatch_.src = nullptr;
    dispatch_.pool = nullptr;
    dispatch_.host_res = nullptr;
    dispatch_.cache_base = nullptr;
    if (device_res_ != nullptr) {
        if (cudaFree(device_res_) != cudaSuccess) std::_Exit(75);
        device_res_ = nullptr;
    }
    cache_.close();
    source_.close();
    host_res_.clear();
    filled_slots_ = 0;
    device_ = -1;
}

} // namespace strata::heterogeneous
