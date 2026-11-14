#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::core { class FileExpertSource; class ExpertCache; }
namespace strata::heterogeneous {

// Borrowed native Prefill resources. No ownership or inference math here.
// Only valid while the original stage provider is ready, with immutable
// residency and no concurrent refill/loan. Never serialize these pointers.
struct NativePrefillExpertView {
    core::FileExpertSource* source = nullptr;
    const core::ExpertCache* cache = nullptr;
    const int32_t* host_res = nullptr;
    size_t residency_entries = 0;
    int device = -1;
    int64_t first_layer = 0, end_layer = 0;
    explicit operator bool() const noexcept { return source && cache && host_res && residency_entries && device >= 0; }
};

} // namespace strata::heterogeneous
