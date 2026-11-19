#pragma once

#include <string>
#include <vector>

namespace strata::mixed_split {

// The host and front stage each pin their main thread; the front runner inherits that pin. A pipelined host can
// spin on its core while the runner services GPU doorbells, so placing both on the default first core can starve the
// runner. Leave an explicit front-stage choice alone. The OS / pool still owns the available CPU set.
inline std::string front_pipeline_host_core(int windows, const std::string& host_core,
                                            const std::vector<std::string>& front_args) {
    if (windows <= 0) return {};
    for (const auto& arg : front_args)
        if (arg == "--host-core") return {};
    return host_core == "last" ? "first" : "last";
}

} // namespace strata::mixed_split
