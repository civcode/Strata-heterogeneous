#pragma once

// Real HIP-only NativeDense and layer-1 PLE lifetime companion to the
// stage weight/session owner. Both must outlive ALL verifier graphs.
#include "strata_heterogeneous/hip_stage_resources.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/kernels/ngram.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::heterogeneous {

struct HipNativeStageOptions {
    std::vector<std::string> gguf_shards; // explicit, ordered, matching original GGUF files
    std::string ple_gguf_path;           // required only if this stage owns layer 1
    uint64_t free_reserve_bytes = 512ull << 20;
    // Optional budget across compact weights, session, native matrices, PLE
    // staging/scratch. 0 leaves native allocations bounded by current free VRAM.
    uint64_t total_arena_budget_bytes = 0;
};

// ONE-SHOT: load stage resources -> open native stage -> attach experts ->
// construct Verifier -> destroy Verifier -> shutdown native -> shutdown owner.
// The HIP native owner does not attest the GGUF binaries; it does validate
// their headers, tensor geometry, actual payload extents and association with
// the native WeightTable via the existing core::NativeDense implementation.
class HipNativeStage final {
public:
    HipNativeStage() = default;
    ~HipNativeStage() noexcept { shutdown(); }
    HipNativeStage(const HipNativeStage&) = delete;
    HipNativeStage& operator=(const HipNativeStage&) = delete;

    bool open(HipStageResources& owner, const Plan& plan, size_t stage_index,
              const HipNativeStageOptions& options, std::string& error);
    void shutdown() noexcept;
    bool ready() const noexcept { return ready_; }
    bool ple_ready() const noexcept { return !needs_ple_ || (ple_table_ && ple_table_->is_open()); }
    uint64_t native_bytes() const noexcept { return native_bytes_; }
private:
    bool setup_ple(const HipNativeStageOptions&, std::string& error);
    HipStageResources* owner_ = nullptr;
    std::unique_ptr<core::NativeDense> native_;
    std::unique_ptr<strata::kernels::PleTable> ple_table_;
    float* ple_emb_dev_ = nullptr;
    float* ple_emb_host_ = nullptr;
    float* ple_scratch_ = nullptr;
    uint64_t native_bytes_ = 0;
    int device_ = -1;
    bool attempted_ = false;
    bool ready_ = false;
    bool needs_ple_ = false;
};

} // namespace strata::heterogeneous
