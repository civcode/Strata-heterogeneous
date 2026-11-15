// CPU-only exec probe: verifies the stage's own --config argument without
// initializing a runtime or opening model/config files.
#include "strata_heterogeneous/host.hpp"
#include <charconv>
#include <cstdint>
#include <string>
using namespace strata::heterogeneous;
int main(int argc, char** argv) {
    if ((argc != 10 && argc != 12) || !argv || !argv[8] || std::string(argv[8]) != "--config") return 12;
    const std::string mode(argv[1]);
    const bool grouped = mode=="--worker-hip-domain" || mode=="--worker-hip-domain-shared";
    if ((!grouped && mode!="--worker" && mode!="--worker-shared") || argc!=(grouped?12:10)) return 13;
    uint32_t index = 0;
    const std::string value(argv[7]);
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), index);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || index > 2)
        return 14;
    try {
        const Plan plan = parse_plan(argv[2], 10, {2, 6}, 999);
        const bool mixed = plan.stages.size() == 3 &&
            plan.stages[0].backend == Backend::cuda &&
            plan.stages[1].backend == Backend::hip && plan.stages[2].backend == Backend::hip;
        const bool hip_pair = plan.stages.size() == 2 &&
            plan.stages[0].backend == Backend::hip && plan.stages[1].backend == Backend::hip;
        if ((!mixed && !hip_pair) || index >= plan.stages.size()) return 16;
        if (grouped) {
            const uint32_t first = mixed ? 1u : 0u;
            if (index != first + 1 ||
                std::string(argv[9]) != "/tmp/m20-stage-" + std::to_string(first) + ".cfg" ||
                std::string(argv[10]) != "--config" ||
                std::string(argv[11]) != "/tmp/m20-stage-" + std::to_string(index) + ".cfg") return 15;
        } else if (std::string(argv[9]) != "/tmp/m20-stage-" + std::to_string(index) + ".cfg") return 15;
    } catch (...) { return 17; }
    return 0;
}
