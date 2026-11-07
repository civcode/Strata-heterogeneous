// Test-only HIP link wrappers: real, valid work executes before a reported error.
#include "strata/core/remote_readback.hpp"
#include "strata/core/event_completion.hpp"
#include "strata/prefill/expert_stager.hpp"
#include <hip/hip_runtime_api.h>
#include <array>
#include <atomic>
#include <iostream>
#include <map>
#include <string>

namespace {
std::atomic<bool> armed{false};
std::atomic<int> injected{0};
std::string mode;
std::mutex allocation_mu;
std::map<void*, size_t> live_host;
int host_frees = 0;
void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
struct Source { bool copy_blobs(const int32_t*, const int32_t*, uint8_t* const*, size_t) { return false; } };
using Worker = strata::prefill::detail::ExpertStager<Source>;
}
extern "C" {
hipError_t __real_hipMemcpyAsync(void*, const void*, size_t, hipMemcpyKind, hipStream_t);
hipError_t __real_hipEventRecord(hipEvent_t, hipStream_t);
hipError_t __real_hipEventSynchronize(hipEvent_t);
hipError_t __real_hipEventQuery(hipEvent_t);
hipError_t __real_hipStreamSynchronize(hipStream_t);
hipError_t __real_hipHostMalloc(void**, size_t, unsigned);
hipError_t __real_hipFreeHost(void*);
hipError_t __wrap_hipHostMalloc(void** p, size_t n, unsigned flags) {
    auto e = __real_hipHostMalloc(p, n, flags);
    if (e == hipSuccess) { std::lock_guard<std::mutex> lk(allocation_mu); live_host[*p] = n; }
    return e;
}
hipError_t __wrap_hipFreeHost(void* p) {
    auto e = __real_hipFreeHost(p);
    if (e == hipSuccess) { std::lock_guard<std::mutex> lk(allocation_mu); live_host.erase(p); ++host_frees; }
    return e;
}
hipError_t __wrap_hipMemcpyAsync(void* dst, const void* src, size_t n, hipMemcpyKind kind, hipStream_t s) {
    auto e = __real_hipMemcpyAsync(dst, src, n, kind, s);
    if (e == hipSuccess && armed && mode == "remote-submit" && injected.fetch_add(1) == 0) return hipErrorUnknown;
    return e;
}
hipError_t __wrap_hipEventRecord(hipEvent_t e, hipStream_t s) {
    auto r = __real_hipEventRecord(e, s);
    if (r == hipSuccess && armed && mode == "stager-record" && injected.fetch_add(1) == 0) return hipErrorUnknown;
    return r;
}
hipError_t __wrap_hipEventSynchronize(hipEvent_t e) {
    auto r = __real_hipEventSynchronize(e);
    if (r == hipSuccess && armed && mode == "stager-wait") { ++injected; return hipErrorUnknown; }
    return r;
}
hipError_t __wrap_hipEventQuery(hipEvent_t e) {
    if (armed && mode == "query-error") {
        const auto r = __real_hipEventSynchronize(e);
        if (r != hipSuccess) return r;
        ++injected; return hipErrorUnknown;
    }
    return __real_hipEventQuery(e);
}
hipError_t __wrap_hipStreamSynchronize(hipStream_t s) {
    auto r = __real_hipStreamSynchronize(s);
    if (r == hipSuccess && armed && (mode == "remote-sync" || mode == "stager-record" || mode == "stager-wait")) {
        ++injected; return hipErrorUnknown;
    }
    return r;
}
}
int main(int argc, char** argv) {
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count == 0) return 77;
    mode = argc > 1 ? argv[1] : "remote-ok";
    const int device = argc > 2 ? std::atoi(argv[2]) : 0;
    if (hipSetDevice(device) != hipSuccess) return 77;
    void* dev = nullptr;
    hipStream_t stream = nullptr;
    hipEvent_t event = nullptr;
    int result = 0;
    try {
        require(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking) == hipSuccess, "stream create");
        require(hipMalloc(&dev, 4096) == hipSuccess, "device allocation");
        std::array<uint8_t, 4096> input{}, out{};
        for (size_t i = 0; i < input.size(); ++i) input[i] = (uint8_t) (i * 17 + 3);
        require(hipMemcpy(dev, input.data(), input.size(), hipMemcpyHostToDevice) == hipSuccess, "initial upload");
        if (mode.find("remote-") == 0) {
            const uint8_t* src[] = {static_cast<const uint8_t*>(dev)};
            uint8_t* dst[] = {out.data()}; size_t bytes[] = {out.size()};
            {
                strata::core::detail::RemoteReadback owner;
                armed = true;
                auto e = owner.copy(src, dst, bytes, 1, stream);
                if (mode == "remote-ok") {
                    require(e == cudaSuccess && out == input, "remote exact bytes");
                    owner.release_completed();
                } else {
                    require(e != cudaSuccess && out[0] == 0 && injected > 0, "remote error published bytes / injection missed");
                    const size_t cap = owner.capacity();
                    for (int i = 0; i < 100; ++i) require(owner.copy(src, dst, bytes, 1, stream) != cudaSuccess, "remote poisoned retry");
                    require(cap == owner.capacity(), "remote resource growth");
                    if (!owner.unknown_completion()) owner.release_completed();
                    else require(host_frees == 0, "remote uncertain stage freed");
                }
            }
        } else if (mode.find("stager-") == 0) {
            bool failed = false;
            {
                Worker worker; worker.kRing = 2;
                require(worker.init(input.size(), 2, true), "worker init");
                // Wait injection is armed only after preparation, so it hits reuse, not an initial empty event.
                worker.start({{input.data(), input.size()}, {input.data(), input.size()}, {input.data(), input.size()}});
                (void) worker.wait(0); (void) worker.wait(1);
                armed = true;
                try {
                    worker.upload(0, dev, input.size(), stream);
                    worker.upload(1, dev, input.size(), stream);
                    worker.upload(2, dev, input.size(), stream);
                } catch (const strata::prefill::detail::StagerFailure&) { failed = true; }
                worker.finish();
                if (mode == "stager-ok") {
                    require(!failed, "healthy worker failed");
                    require(__real_hipStreamSynchronize(stream) == hipSuccess, "worker drain");
                    require(hipMemcpy(out.data(), dev, out.size(), hipMemcpyDeviceToHost) == hipSuccess && out == input,
                            "worker exact bytes");
                } else require(failed && worker.failed && injected > 0, "worker failure not propagated / injection missed");
            }
            if (mode != "stager-ok") require(host_frees == 0 && live_host.size() == 2, "worker sources not retained");
        } else {
            require(hipEventCreateWithFlags(&event, hipEventDisableTiming) == hipSuccess, "event create");
            require(hipMemcpyAsync(dev, input.data(), input.size(), hipMemcpyHostToDevice, stream) == hipSuccess, "query upload");
            require(hipEventRecord(event, stream) == hipSuccess, "query record");
            armed = true;
            const auto e = strata::core::wait_event_completion(event);
            require((e == cudaSuccess) == (mode == "query-ok"), "query error accepted as completion");
            if (mode == "query-error") require(injected > 0, "query injection missed");
        }
        armed = false;
        require(__real_hipStreamSynchronize(stream) == hipSuccess, "final real completion");
        std::cout << mode << " device=" << device << " injected=" << injected << " retained_host=" << live_host.size() << " PASS\n";
    } catch (const std::exception& e) { armed = false; std::cerr << "FAIL: " << e.what() << '\n'; result = 1; }
    // Fixture-only reclamation after a real successful fence, never production timeout/error cleanup.
    const bool idle = !stream || __real_hipStreamSynchronize(stream) == hipSuccess;
    if (idle) {
        for (const auto& [p, n] : live_host) (void) __real_hipFreeHost(p);
        if (event) (void) hipEventDestroy(event);
        if (dev) (void) hipFree(dev);
        if (stream) (void) hipStreamDestroy(stream);
    }
    return result;
}
