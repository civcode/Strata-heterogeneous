#pragma once

// Include only in an exec-isolated native worker. Every pointer is resolved
// locally, never transported to a different runtime or process.
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "strata_heterogeneous/direct_handoff_layout.hpp"

namespace strata::heterogeneous {
inline void direct_check(cudaError_t status, const char* what) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

struct DirectHandoffView {
    uint8_t* address = nullptr;
    DirectHandoffLayout layout{};
    float* payload() const noexcept { return reinterpret_cast<float*>(address); }
    uint32_t* ready() const noexcept { return reinterpret_cast<uint32_t*>(address + layout.ready_offset); }
    uint32_t* error() const noexcept { return reinterpret_cast<uint32_t*>(address + layout.error_offset); }
};

// Deliberately requires explicit release after all captured graphs and streams
// have drained/died. On an uncertain drain, leave resources to process exit;
// never unmap pages still potentially referenced by device work.
class DirectHandoffRegistration {
public:
    DirectHandoffRegistration() = default;
    DirectHandoffRegistration(const DirectHandoffRegistration&) = delete;
    DirectHandoffRegistration& operator=(const DirectHandoffRegistration&) = delete;

    void open(int fd, int owner, DirectHandoffLayout layout) {
        if (page_ || owner < 0 || !layout.extent) throw std::logic_error("invalid direct edge initialization");
        struct stat st{};
        constexpr int seals = F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL;
        const int actual = fcntl(fd, F_GET_SEALS);
        if (fstat(fd, &st) || !S_ISREG(st.st_mode) || st.st_size != static_cast<off_t>(layout.extent) ||
            actual < 0 || (actual & seals) != seals)
            throw std::runtime_error("direct handoff requires exact size-sealed memfd");
        owner_ = owner;
        layout_ = layout;
        direct_check(cudaSetDevice(owner_), "direct registration owner");
        void* page = mmap(nullptr, layout.extent, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (page == MAP_FAILED) throw std::runtime_error("direct handoff mmap failed");
        page_ = static_cast<uint8_t*>(page);
        direct_check(cudaHostRegister(page_, layout.extent, cudaHostRegisterMapped | cudaHostRegisterPortable),
                     "direct host registration");
        registered_ = true;
    }
    DirectHandoffView host() const {
        if (!registered_) throw std::logic_error("direct host mapping absent");
        return {page_, layout_};
    }
    DirectHandoffView local_alias(int device) const {
        if (!registered_) throw std::logic_error("direct device mapping absent");
        direct_check(cudaSetDevice(device), "direct alias owner");
        void* alias = nullptr;
        direct_check(cudaHostGetDevicePointer(&alias, page_, 0), "direct device alias");
        if (!alias) throw std::runtime_error("null direct device alias");
        return {static_cast<uint8_t*>(alias), layout_};
    }
    bool release_after_drain() noexcept {
        if (registered_) {
            if (cudaSetDevice(owner_) != cudaSuccess || cudaHostUnregister(page_) != cudaSuccess) return false;
            registered_ = false;
        }
        if (page_) {
            if (munmap(page_, layout_.extent)) return false;
            page_ = nullptr;
        }
        return true;
    }
private:
    int owner_ = -1;
    uint8_t* page_ = nullptr;
    bool registered_ = false;
    DirectHandoffLayout layout_{};
};
} // namespace strata::heterogeneous
