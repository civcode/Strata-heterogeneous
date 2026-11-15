#pragma once
#include "strata_heterogeneous/host.hpp"

namespace strata::heterogeneous {
// Physical stages remain part of the original Plan and fingerprint. A domain
// is one native runtime execution chain, never a synthetic GPU.
struct BackendDomain {
    Backend backend;
    size_t first_stage;
    size_t end_stage;
    size_t size() const noexcept { return end_stage - first_stage; }
    size_t terminal_stage() const noexcept { return end_stage - 1; }
};

inline std::vector<BackendDomain> native_backend_domains(const Plan& plan) {
    plan.validate();
    std::vector<BackendDomain> domains;
    for (size_t i = 0; i < plan.stages.size(); ++i) {
        const auto backend = plan.stages[i].backend;
        if (!domains.empty() && domains.back().backend == backend) {
            domains.back().end_stage = i + 1;
        } else {
            for (const auto& d : domains)
                if (d.backend == backend)
                    throw std::invalid_argument("native backend domain cannot re-enter a runtime");
            domains.push_back({backend, i, i + 1});
        }
    }
    return domains;
}

// Supported grouped topology shapes: CUDA front + terminal HIP pair, or a full HIP pair.
// Grouped HIP-only is experimental; physical stage identity remains unchanged.
// Do not quietly fall back to per-GPU execution for unsupported grouping.
inline std::vector<BackendDomain> worker_domains(const Plan& plan, bool group_hip) {
    if (group_hip) {
        auto domains = native_backend_domains(plan);
        const bool hip_only = domains.size() == 1 && domains[0].backend == Backend::hip &&
                              domains[0].size() == 2;
        const bool mixed = domains.size() == 2 && domains[0].backend == Backend::cuda &&
                           domains[0].size() == 1 && domains[1].backend == Backend::hip &&
                           domains[1].size() == 2;
        if (!hip_only && !mixed)
            throw std::invalid_argument("grouped mode requires two HIP stages, optionally preceded by one CUDA stage");
        return domains;
    }
    plan.validate();
    std::vector<BackendDomain> domains;
    for (size_t i = 0; i < plan.stages.size(); ++i)
        domains.push_back({plan.stages[i].backend, i, i + 1});
    return domains;
}
} // namespace strata::heterogeneous
