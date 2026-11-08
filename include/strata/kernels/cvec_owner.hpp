#pragma once

#include "strata/core/transfer_terminal.hpp"
#include <cuda_runtime.h>
#include <cstddef>

namespace strata::kernels::detail {

inline void cvec_cleanup_check(cudaError_t status, const char* action) noexcept {
    if (status != cudaSuccess)
        strata::core::transfer_terminal(action, cudaGetErrorString(status));
}

template<class Tables>
void cvec_release_here(Tables& t) noexcept {
    if (t.dir == nullptr && t.s == nullptr && t.on == nullptr) return;
    // Do not release tables while a captured kernel can still read them.
    cvec_cleanup_check(cudaDeviceSynchronize(), "control-vector table completion");
    if (t.on) cvec_cleanup_check(cudaFree(t.on), "control-vector flag release");
    if (t.s) cvec_cleanup_check(cudaFree(t.s), "control-vector scale release");
    if (t.dir) cvec_cleanup_check(cudaFree(t.dir), "control-vector direction release");
    t = Tables{};
}

template<class Tables, std::size_t N>
void cvec_release_devices(Tables (&tables)[N]) noexcept {
    bool any = false;
    for (const auto& t : tables)
        if (t.dir || t.s || t.on) { any = true; break; }
    if (!any) return; // no HIP/CUDA calls on the empty path
    int previous = -1;
    cvec_cleanup_check(cudaGetDevice(&previous), "control-vector previous device");
    int current = previous;
    for (std::size_t d = 0; d < N; ++d) {
        auto& t = tables[d];
        if (!t.dir && !t.s && !t.on) continue;
        if (current != static_cast<int>(d)) {
            cvec_cleanup_check(cudaSetDevice(static_cast<int>(d)), "control-vector select device");
            current = static_cast<int>(d);
        }
        cvec_release_here(t);
    }
    if (current != previous)
        cvec_cleanup_check(cudaSetDevice(previous), "control-vector restore device");
}

} // namespace strata::kernels::detail
