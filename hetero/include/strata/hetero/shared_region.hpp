#pragma once

// The memory the two processes of a mixed split share: the stage hand-off buffers and their ready words.
//
// Vanilla allocates these with cudaHostAlloc(Mapped | Portable) inside one process.  Here they are slots of one
// memfd mapping that the host creates and passes to the front worker over their socket (SCM_RIGHTS).  Each process
// registers its own mapping with its own runtime; no device pointer, stream or event ever crosses.
//
// Host-only code: nothing here touches CUDA or HIP.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::hetero {

class SharedRegion {
public:
    static constexpr size_t kPage = 4096;
    static constexpr size_t kMaxSlots = 32;
    static constexpr size_t kMaxBytes = size_t(1) << 32;

    SharedRegion() = default;
    ~SharedRegion() noexcept { release(); }
    SharedRegion(const SharedRegion&) = delete;
    SharedRegion& operator=(const SharedRegion&) = delete;
    SharedRegion(SharedRegion&& o) noexcept;
    SharedRegion& operator=(SharedRegion&& o) noexcept;

    /// A zeroed region with one slot per entry of `slot_bytes`.  Each slot starts on a page boundary, so a slot can
    /// be registered with a GPU runtime by itself.
    static bool create(const std::vector<size_t>& slot_bytes, SharedRegion& out, std::string& err);

    /// Map a region from its descriptor and slot table (the receiving process).  Takes ownership of `fd`.
    static bool adopt(int fd, const std::vector<size_t>& slot_bytes, SharedRegion& out, std::string& err);

    bool valid() const { return base_ != nullptr; }
    int descriptor() const { return fd_; }               ///< borrowed; do not close
    size_t bytes() const { return bytes_; }
    size_t slots() const { return offsets_.size(); }
    size_t slot_bytes(size_t i) const { return sizes_.at(i); }
    const std::vector<size_t>& slot_sizes() const { return sizes_; }
    void* slot(size_t i) { return base_ + offsets_.at(i); }
    const void* slot(size_t i) const { return base_ + offsets_.at(i); }
    void* base() { return base_; }

private:
    int fd_ = -1;
    uint8_t* base_ = nullptr;
    size_t bytes_ = 0;
    std::vector<size_t> sizes_, offsets_;
    static bool layout(const std::vector<size_t>& slot_bytes, std::vector<size_t>& offsets, size_t& total, std::string& err);
    void release() noexcept;
};

} // namespace strata::hetero
