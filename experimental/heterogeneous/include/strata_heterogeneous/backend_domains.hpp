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

// Initial qualified topology target: CUDA front, terminal HIP chain.
// Do not quietly fall back to per-GPU execution for unsupported grouping.
inline std::vector<BackendDomain> worker_domains(const Plan& plan, bool group_hip) {
    if (group_hip) {
        auto domains = native_backend_domains(plan);
        if (domains.size() != 2 || domains[0].backend != Backend::cuda ||
            domains[0].size() != 1 || domains[1].backend != Backend::hip ||
            domains[1].size() != 2)
            throw std::invalid_argument("grouped mode requires one CUDA stage followed by two HIP stages");
        return domains;
    }
    plan.validate();
    std::vector<BackendDomain> domains;
    for (size_t i = 0; i < plan.stages.size(); ++i)
        domains.push_back({plan.stages[i].backend, i, i + 1});
    return domains;
}
} // namespace strata::heterogeneous
