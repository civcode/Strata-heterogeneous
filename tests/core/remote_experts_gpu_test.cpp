// Exercises the real RemoteExperts and RemotePrefillSource with model-independent blobs.
#include "strata/core/remote_experts.hpp"
#include "strata/core/device.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace {
bool armed = false, unknown = false, submit_error = false, alloc_error = false;
hipStream_t selected_stream = nullptr;
const void* selected_source = nullptr;
void* selected_landing = nullptr;
int copies = 0, allocs = 0, source_frees = 0, landing_frees = 0;
void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
struct Source : strata::core::ExpertSource {
    std::vector<uint8_t> bytes;
    int fallback = 0;
    Source() : bytes(strata::kernels::cpu::BLOB) {
        for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = (uint8_t) (3 + i * 17);
    }
    const uint8_t* blob(int64_t, int64_t) override { return bytes.data(); }
    bool copy_blob(int64_t, int64_t, uint8_t* dst) override {
        ++fallback; std::memcpy(dst, bytes.data(), bytes.size()); return true;
    }
};
}
extern "C" {
hipError_t __real_hipMemcpyAsync(void*, const void*, size_t, hipMemcpyKind, hipStream_t);
hipError_t __real_hipStreamSynchronize(hipStream_t);
hipError_t __real_hipHostMalloc(void**, size_t, unsigned);
hipError_t __real_hipFreeHost(void*);
hipError_t __real_hipFree(void*);
hipError_t __wrap_hipMemcpyAsync(void* d, const void* s, size_t n, hipMemcpyKind k, hipStream_t stream) {
    auto e = __real_hipMemcpyAsync(d, s, n, k, stream);
    if (armed && e == hipSuccess && k == hipMemcpyDeviceToHost && n == strata::kernels::cpu::BLOB) {
        ++copies; selected_stream = stream; selected_source = s; selected_landing = d;
        if (submit_error) return hipErrorUnknown; // actual valid work remains queued
    }
    return e;
}
hipError_t __wrap_hipStreamSynchronize(hipStream_t s) {
    auto e = __real_hipStreamSynchronize(s);
    if (armed && unknown && selected_stream == s && e == hipSuccess) return hipErrorUnknown;
    return e;
}
hipError_t __wrap_hipHostMalloc(void** p, size_t n, unsigned f) {
    if (armed && alloc_error) return hipErrorOutOfMemory;
    auto e = __real_hipHostMalloc(p, n, f); if (armed && e == hipSuccess) ++allocs; return e;
}
hipError_t __wrap_hipFreeHost(void* p) {
    if (p == selected_landing) ++landing_frees;
    return __real_hipFreeHost(p);
}
hipError_t __wrap_hipFree(void* p) {
    if (p == selected_source) ++source_frees;
    return __real_hipFree(p);
}
}
int main(int argc, char** argv) {
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count < 2) return 77;
    const std::string mode = argc > 1 ? argv[1] : "success";
    const int primary_device = argc > 2 ? 1 : 0;
    const int helper_device = 1 - primary_device;
    strata::core::set_primary_device(primary_device);
    try {
        require(hipSetDevice(primary_device) == hipSuccess, "primary device");
        Source source;
        strata::core::ExpertCache primary;
        std::vector<uint8_t> claimed(2, 0), out(source.bytes.size(), 99), missed(out.size(), 88);
        std::string err;
        {
            strata::core::RemoteExperts remote;
            require(remote.open(helper_device, 1, 1, 2, {{0, 0}}, primary, source, claimed, err), err.c_str());
            strata::core::RemotePrefillSource view(source, {&remote});
            const int32_t layers[] = {0, 0}, experts[] = {0, 1};
            uint8_t* destinations[] = {out.data(), missed.data()};
            unknown = mode == "unknown" || mode == "submit-unknown";
            submit_error = mode == "submit" || mode == "submit-unknown";
            alloc_error = mode == "allocation";
            armed = true;
            const bool ok = view.copy_blobs(layers, experts, destinations, 2);
            if (mode == "success" || mode == "allocation") {
                require(ok && out == source.bytes && missed == source.bytes, "hit/miss fallback bytes");
                require(source.fallback == (alloc_error ? 2 : 1), "clean fallback selection");
                require(view.helper_blobs() == (alloc_error ? 0 : 1), "helper hit accounting");
            } else {
                require(!ok && source.fallback == 0 && out[0] == 99 && missed[0] == 88, "failed batch reached fallback/publication");
                require(selected_landing != out.data(), "GPU received borrowed caller destination");
                require(view.helper_blobs() == 0, "failed hit published");
                const int a = allocs, c = copies;
                for (int i = 0; i < 100; ++i) require(!view.copy_blob(0, 0, out.data()), "terminal helper retried");
                require(allocs == a && copies == c && source.fallback == 0, "terminal allocation/copy/fallback growth");
                require(!remote.open(helper_device, 1, 1, 2, {{0, 0}}, primary, source, claimed, err), "terminal helper reopened");
            }
        }
        if (unknown) require(source_frees == 0 && landing_frees == 0, "unknown-completion source or destination freed by real owner");
        else if (!alloc_error) require(source_frees == 1 && landing_frees == 1, "completed real owners did not release storage");
        armed = false;
        std::cout << "RemoteExperts " << mode << ": PASS copies=" << copies << " new_host_allocs=" << allocs
                  << " source_frees=" << source_frees << " landing_frees=" << landing_frees << '\n';
        return 0;
    } catch (const std::exception& e) { armed = false; std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
