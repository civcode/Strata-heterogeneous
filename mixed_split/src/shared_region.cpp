#include "strata/mixed_split/shared_region.hpp"

#include <cerrno>
#include <cstring>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace strata::mixed_split {

SharedRegion::SharedRegion(SharedRegion&& o) noexcept
    : fd_(o.fd_), base_(o.base_), bytes_(o.bytes_), sizes_(std::move(o.sizes_)), offsets_(std::move(o.offsets_)) {
    o.fd_ = -1; o.base_ = nullptr; o.bytes_ = 0; o.sizes_.clear(); o.offsets_.clear();
}

SharedRegion& SharedRegion::operator=(SharedRegion&& o) noexcept {
    if (this != &o) {
        release();
        fd_ = o.fd_; base_ = o.base_; bytes_ = o.bytes_;
        sizes_ = std::move(o.sizes_); offsets_ = std::move(o.offsets_);
        o.fd_ = -1; o.base_ = nullptr; o.bytes_ = 0; o.sizes_.clear(); o.offsets_.clear();
    }
    return *this;
}

void SharedRegion::release() noexcept {
    if (base_ != nullptr) (void) ::munmap(base_, bytes_);
    if (fd_ >= 0) (void) ::close(fd_);
    base_ = nullptr; fd_ = -1; bytes_ = 0;
    sizes_.clear(); offsets_.clear();
}

bool SharedRegion::layout(const std::vector<size_t>& slot_bytes, std::vector<size_t>& offsets, size_t& total,
                          std::string& err) {
    if (slot_bytes.empty() || slot_bytes.size() > kMaxSlots) { err = "shared region: bad slot count"; return false; }
    offsets.clear();
    total = 0;
    for (const size_t b : slot_bytes) {
        if (b == 0 || b > kMaxBytes) { err = "shared region: bad slot size"; return false; }
        offsets.push_back(total);
        total += (b + kPage - 1) / kPage * kPage;
        if (total > kMaxBytes) { err = "shared region: too large"; return false; }
    }
    return true;
}

bool SharedRegion::create(const std::vector<size_t>& slot_bytes, SharedRegion& out, std::string& err) {
    std::vector<size_t> offsets;
    size_t total = 0;
    if (!layout(slot_bytes, offsets, total, err)) return false;
    const int fd = ::memfd_create("strata-mixed-split-handoff", MFD_CLOEXEC);
    if (fd < 0) { err = std::string("shared region: memfd_create: ") + std::strerror(errno); return false; }
    if (::ftruncate(fd, (off_t) total) != 0) {
        err = std::string("shared region: ftruncate: ") + std::strerror(errno);
        (void) ::close(fd);
        return false;
    }
    return adopt(fd, slot_bytes, out, err);
}

bool SharedRegion::adopt(int fd, const std::vector<size_t>& slot_bytes, SharedRegion& out, std::string& err) {
    std::vector<size_t> offsets;
    size_t total = 0;
    struct stat st {};
    if (fd < 0 || !layout(slot_bytes, offsets, total, err)) {
        if (fd < 0) err = "shared region: no descriptor";
        if (fd >= 0) (void) ::close(fd);
        return false;
    }
    if (::fstat(fd, &st) != 0 || (size_t) st.st_size != total) {
        err = "shared region: the descriptor's size does not match the slot table";
        (void) ::close(fd);
        return false;
    }
    void* p = ::mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) {
        err = std::string("shared region: mmap: ") + std::strerror(errno);
        (void) ::close(fd);
        return false;
    }
    out.release();
    out.fd_ = fd;
    out.base_ = static_cast<uint8_t*>(p);
    out.bytes_ = total;
    out.sizes_ = slot_bytes;
    out.offsets_ = std::move(offsets);
    return true;
}

} // namespace strata::mixed_split
