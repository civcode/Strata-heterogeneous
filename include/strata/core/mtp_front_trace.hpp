#pragma once
#include <array>
#include <cstddef>
#include <cstdint>

namespace strata::core {
// Fixed Qwen4Exp one-row diagnostic ABI. Opaque exact bytes; no padding.
// Field order is execution order except rest metadata, which is staged first.
enum class MtpFrontField : size_t {
    token, step, position, rest_step, rest_position, selected_row,
    residual, embedding, embedding_norm, embedding_fc, hidden_norm,
    hidden_fc, mixed_residual, inject, mixed, q8_mixed,
    k_project, v_project, k_append, v_append, count
};
inline constexpr std::array<size_t, static_cast<size_t>(MtpFrontField::count)> mtp_front_sizes = {
    4, 16, 96, 16, 96, 8, 40960, 10240, 10240, 10240,
    40960, 40960, 40960, 16, 10240, 2880, 2048, 2048, 2048, 2048
};
constexpr size_t mtp_front_offset(MtpFrontField field) noexcept {
    size_t at = 0;
    for (size_t i=0; i<static_cast<size_t>(field); ++i) at+=mtp_front_sizes[i];
    return at;
}
inline constexpr size_t mtp_front_bytes=mtp_front_offset(MtpFrontField::count);
static_assert(mtp_front_bytes<256*1024, "bounded one-row MTP trace");
} // namespace strata::core
