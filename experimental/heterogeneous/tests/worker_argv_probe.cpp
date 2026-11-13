// CPU-only exec probe: verifies the stage's own --config argument without
// initializing a runtime or opening model/config files.
#include "strata_heterogeneous/host.hpp"
#include <charconv>
#include <cstdint>
#include <string>
using namespace strata::heterogeneous;
int main(int argc, char** argv) {
    if (argc != 10 || !argv || !argv[8] || std::string(argv[8]) != "--config") return 12;
    const std::string mode(argv[1]);
    if (mode != "--worker" && mode != "--worker-shared") return 13;
    uint32_t index = 0;
    const std::string value(argv[7]);
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), index);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || index > 2)
        return 14;
    if (std::string(argv[9]) != "/tmp/m20-stage-" + std::to_string(index) + ".cfg")
        return 15;
    try {
        const Plan plan = parse_plan(argv[2], 10, {2, 6}, 999);
        if (plan.stages.size() != 3 ||
            (index == 0 && plan.stages[index].backend != Backend::cuda) ||
            (index != 0 && plan.stages[index].backend != Backend::hip)) return 16;
    } catch (...) { return 17; }
    return 0;
}
