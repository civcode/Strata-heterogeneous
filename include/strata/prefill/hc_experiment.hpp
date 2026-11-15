#pragma once
// Opt-in diagnostic selection, shared by the native prompt path and CPU preflight.
#include <cstdlib>
#include <cstring>

namespace strata::prefill {
enum class HcExperimentMode { off, bf16x2, fp32, invalid };
enum class HcExperimentBackend { all, cuda, hip, invalid };
struct HcExperiment {
    HcExperimentMode mode = HcExperimentMode::off;
    HcExperimentBackend backend = HcExperimentBackend::all;
    bool valid() const {
        return mode != HcExperimentMode::invalid && backend != HcExperimentBackend::invalid &&
               (mode != HcExperimentMode::off || backend == HcExperimentBackend::all);
    }
    bool requested() const { return valid() && mode != HcExperimentMode::off; }
    HcExperimentMode effective(HcExperimentBackend local) const {
        return requested() && (local == HcExperimentBackend::cuda || local == HcExperimentBackend::hip) &&
               (backend == HcExperimentBackend::all || backend == local) ? mode : HcExperimentMode::off;
    }
};
inline HcExperiment parse_hc_experiment(const char* mode, const char* backend) {
    HcExperiment c;
    if (mode && *mode && std::strcmp(mode, "off")) {
        if (!std::strcmp(mode, "bf16x2")) c.mode = HcExperimentMode::bf16x2;
        else if (!std::strcmp(mode, "fp32")) c.mode = HcExperimentMode::fp32;
        else c.mode = HcExperimentMode::invalid;
    }
    if (backend && *backend && std::strcmp(backend, "all")) {
        if (!std::strcmp(backend, "cuda")) c.backend = HcExperimentBackend::cuda;
        else if (!std::strcmp(backend, "hip")) c.backend = HcExperimentBackend::hip;
        else c.backend = HcExperimentBackend::invalid;
    }
    return c;
}
inline HcExperiment hc_experiment_config() {
    return parse_hc_experiment(std::getenv("STRATA_PREFILL_HC_EXPERIMENT"),
                               std::getenv("STRATA_PREFILL_HC_EXPERIMENT_BACKEND"));
}
inline const char* hc_experiment_name(HcExperimentMode mode) {
    switch (mode) {
        case HcExperimentMode::off: return "off";
        case HcExperimentMode::bf16x2: return "bf16x2";
        case HcExperimentMode::fp32: return "fp32";
        default: return "invalid";
    }
}
inline const char* hc_experiment_name(HcExperimentBackend backend) {
    switch (backend) {
        case HcExperimentBackend::all: return "all";
        case HcExperimentBackend::cuda: return "cuda";
        case HcExperimentBackend::hip: return "hip";
        default: return "invalid";
    }
}
}  // namespace strata::prefill
