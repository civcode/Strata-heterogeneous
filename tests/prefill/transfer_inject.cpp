// Explicit LD_PRELOAD integration fixture only. Not linked/installed with Strata.
// Filters the Q2_0 expert blob size and owned allocation flags; every injected
// synchronization error follows a real successful fence. Never causes a GPU fault.
#include <hip/hip_runtime_api.h>
#include <dlfcn.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
namespace {
template<class F> F next(const char* name) { return reinterpret_cast<F>(dlsym(RTLD_NEXT, name)); }
struct Allocation { size_t n; unsigned flags; };
std::mutex mu;
std::map<void*, Allocation> hosts;
std::atomic<bool> have_stager{false}, selected{false}, injected{false};
std::atomic<hipStream_t> stream{nullptr};
std::atomic<hipEvent_t> event{nullptr};
std::atomic<void*> target{nullptr};
constexpr size_t blob = 1382400;
const char* mode() { const char* p = std::getenv("STRATA_TRANSFER_INJECT"); return p ? p : "observe"; }
bool is(const char* p) { return std::strcmp(mode(), p) == 0; }
bool remote() { return std::strncmp(mode(), "remote-", 7) == 0; }
}
extern "C" hipError_t hipHostMalloc(void** p, size_t n, unsigned flags) {
    auto e = next<hipError_t(*)(void**, size_t, unsigned)>("hipHostMalloc")(p, n, flags);
    if (e == hipSuccess) {
        std::lock_guard<std::mutex> lk(mu); hosts[*p] = {n, flags};
        if (n == blob && flags == hipHostMallocDefault) have_stager = true;
    }
    return e;
}
extern "C" hipError_t hipMemcpyAsync(void* dst, const void* src, size_t n, hipMemcpyKind kind, hipStream_t s) {
    auto e = next<hipError_t(*)(void*, const void*, size_t, hipMemcpyKind, hipStream_t)>("hipMemcpyAsync")(dst, src, n, kind, s);
    if (e != hipSuccess || n != blob || !have_stager || selected) return e;
    bool pick = false;
    {
        std::lock_guard<std::mutex> lk(mu);
        auto it = hosts.find(remote() ? dst : const_cast<void*>(src));
        pick = it != hosts.end() &&
               (remote() ? kind == hipMemcpyDeviceToHost && (it->second.flags & hipHostMallocPortable)
                         : kind == hipMemcpyHostToDevice && it->second.flags == hipHostMallocDefault);
    }
    bool expected = false;
    if (!pick || !selected.compare_exchange_strong(expected, true)) return e;
    stream = s; target = remote() ? dst : const_cast<void*>(src);
    std::fprintf(stderr, "TRANSFER_INJECT selected %s bytes=%zu owned_host=%p stream=%p\n", mode(), n, target.load(), (void*) s);
    if (is("remote-submit")) { injected = true; std::fprintf(stderr, "TRANSFER_INJECT queued copy reports synthetic error\n"); return hipErrorUnknown; }
    return e;
}
extern "C" hipError_t hipEventRecord(hipEvent_t ev, hipStream_t s) {
    auto e = next<hipError_t(*)(hipEvent_t, hipStream_t)>("hipEventRecord")(ev, s);
    if (e == hipSuccess && selected && !remote() && s == stream.load() && event.load() == nullptr) {
        event = ev;
        std::fprintf(stderr, "TRANSFER_INJECT staging completion record event=%p\n", (void*) ev);
        if (is("stager-record")) { injected = true; return hipErrorUnknown; }
    }
    return e;
}
extern "C" hipError_t hipEventSynchronize(hipEvent_t ev) {
    auto e = next<hipError_t(*)(hipEvent_t)>("hipEventSynchronize")(ev);
    if (e == hipSuccess && ev == event.load() && is("stager-wait")) {
        injected = true; std::fprintf(stderr, "TRANSFER_INJECT completed staging wait reports synthetic error\n"); return hipErrorUnknown;
    }
    return e;
}
extern "C" hipError_t hipStreamSynchronize(hipStream_t s) {
    auto e = next<hipError_t(*)(hipStream_t)>("hipStreamSynchronize")(s);
    if (e == hipSuccess && selected && s == stream.load() &&
        (is("remote-sync") || ((is("stager-record") || is("stager-wait")) && injected))) {
        injected = true; std::fprintf(stderr, "TRANSFER_INJECT completed stream reports synthetic error\n"); return hipErrorUnknown;
    }
    return e;
}
extern "C" hipError_t hipFreeHost(void* p) {
    if (selected && p == target.load()) std::fprintf(stderr, "TRANSFER_INJECT freeing selected host allocation\n");
    { std::lock_guard<std::mutex> lk(mu); hosts.erase(p); }
    return next<hipError_t(*)(void*)>("hipFreeHost")(p);
}
