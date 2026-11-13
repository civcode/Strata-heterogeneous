#include "strata_heterogeneous/hip_worker_config.hpp"
#include <stdexcept>
#include <utility>

namespace strata::heterogeneous {
bool read_hip_worker_config(const std::string& filename,
                            HipWorkerConfig& out, std::string& error) {
    CudaWorkerConfig next;
    if (!read_cuda_worker_config(filename, next, error)) return false;
    if (next.version != 2) {
        error = "HIP worker needs v2 native GGUF/PLE configuration";
        return false;
    }
    HipWorkerConfig checked;
    checked.fields = std::move(next);
    out = std::move(checked);
    error.clear();
    return true;
}

bool validate_hip_worker_config(const HipWorkerConfig& config,
                                const Plan& plan, size_t index,
                                std::string& error) {
    try {
        plan.validate();
        const auto& c = config.fields;
        if (c.version != 2 || c.gguf_shards.empty() ||
            index >= plan.stages.size() || index != c.stage_index)
            throw std::invalid_argument("HIP configuration version or stage index mismatch");
        const Stage& stage = plan.stages[index];
        if (stage.backend != Backend::hip || stage.ordinal != c.device_ordinal ||
            stage.first_layer != c.first_layer || stage.end_layer != c.end_layer ||
            plan.total_layers != c.n_layers || plan.geometry.hc != c.hc ||
            plan.geometry.n_embd != c.n_embd ||
            plan.model_fingerprint != c.model_fingerprint)
            throw std::invalid_argument("HIP config and coordinator plan identity mismatch");
        if (!validate_cuda_worker_files(c, error)) return false;
        error.clear();
        return true;
    } catch (const std::exception& ex) {
        error = ex.what();
        return false;
    }
}
} // namespace strata::heterogeneous
