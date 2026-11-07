#include "strata_heterogeneous/host.hpp"

#include <cstring>
#include <limits>

namespace strata::heterogeneous {
namespace {
size_t checked_bytes(uint64_t count) {
    if (count > max_payload_bytes / sizeof(float))
        throw std::invalid_argument("activation window exceeds the 16 MiB frame cap");
    return static_cast<size_t>(count) * sizeof(float);
}
} // namespace
float ByteRange::read_f32(size_t element) const {
    if (!data || element >= bytes / sizeof(float) || bytes % sizeof(float))
        throw std::out_of_range("activation float read out of range");
    float result;
    std::memcpy(&result, data + element * sizeof(float), sizeof(result));
    return result;
}
void ByteRange::write_f32(size_t element, float value) const {
    if (!data || element >= bytes / sizeof(float) || bytes % sizeof(float))
        throw std::out_of_range("activation float write out of range");
    std::memcpy(data + element * sizeof(float), &value, sizeof(value));
}
ActivationView::ActivationView(uint8_t* data, size_t bytes, Geometry geometry, uint32_t rows)
    : data_(data), bytes_(bytes), geometry_(geometry), rows_(rows), stride_bytes_(0) {
    if (!data || !rows) throw std::invalid_argument("empty activation view");
    const uint64_t stride = geometry.floats_per_row();
    stride_bytes_ = checked_bytes(stride);
    if (rows > max_payload_bytes / stride_bytes_ ||
        bytes != static_cast<size_t>(rows) * stride_bytes_)
        throw std::invalid_argument("activation view does not match row geometry or capacity");
}
HandoffRow ActivationView::row(uint32_t index) const {
    if (index >= rows_) throw std::out_of_range("activation row is out of range");
    uint8_t* p = data_ + static_cast<size_t>(index) * stride_bytes_;
    const size_t r_bytes = static_cast<size_t>(geometry_.hc) * geometry_.n_embd * sizeof(float);
    const size_t bo_bytes = static_cast<size_t>(geometry_.n_embd) * sizeof(float);
    const size_t inject_bytes = static_cast<size_t>(geometry_.hc) * sizeof(float);
    return {{p, r_bytes}, {p + r_bytes, bo_bytes}, {p + r_bytes + bo_bytes, inject_bytes}};
}
} // namespace strata::heterogeneous
