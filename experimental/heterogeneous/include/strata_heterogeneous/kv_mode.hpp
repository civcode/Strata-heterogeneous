#pragma once

// Shared CPU-only vocabulary for the coordinator and both native workers.
// Names match vanilla Strata's --kv option. INT8 is Q8 with FP16 scales.
#include <stdexcept>
#include <string_view>

namespace strata::heterogeneous {
enum class KvMode { f16, int8 };
inline constexpr KvMode default_kv_mode = KvMode::int8;
inline constexpr const char* kv_mode_name(KvMode mode) noexcept {
    return mode == KvMode::int8 ? "int8" : "f16";
}
inline KvMode parse_kv_mode(std::string_view name) {
    if (name == "int8") return KvMode::int8;
    if (name == "f16") return KvMode::f16;
    throw std::invalid_argument("KV mode must be int8 or f16");
}
} // namespace strata::heterogeneous
