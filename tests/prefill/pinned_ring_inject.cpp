// Explicitly loaded test-only interposer for the real engine. Build outside the normal engine;
// no production failure switches. Report an error only AFTER real HIP work has completed safely.
#include <hip/hip_runtime_api.h>
#include <dlfcn.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_set>

namespace {
std::mutex mu;
std::unordered_set<void*> slots;
std::atomic<bool> fired{false}, reported_error{false};
std::atomic<void*> selected_slot{nullptr};
thread_local hipStream_t selected_stream=nullptr;
thread_local bool selected=false;
template<class F> F next(const char* name) {
    auto f=reinterpret_cast<F>(dlsym(RTLD_NEXT,name));
    if(!f)std::abort();
    return f;
}
}
extern "C" hipError_t hipHostMalloc(void** p,size_t n,unsigned flags) {
    const auto e=next<hipError_t(*)(void**,size_t,unsigned)>("hipHostMalloc")(p,n,flags);
    if(e==hipSuccess && n==65536) { std::lock_guard<std::mutex> lock(mu); slots.insert(*p); }
    return e;
}
extern "C" hipError_t hipMemcpyAsync(void* dst,const void* src,size_t n,hipMemcpyKind kind,hipStream_t s) {
    const auto e=next<hipError_t(*)(void*,const void*,size_t,hipMemcpyKind,hipStream_t)>("hipMemcpyAsync")(dst,src,n,kind,s);
    bool target=false;
    { std::lock_guard<std::mutex> lock(mu); target=slots.count(dst)!=0; }
    // --prefill 1280, top-k 10 => the grouping IDs are 1280*10*4 bytes.
    if(e==hipSuccess && target && kind==hipMemcpyDeviceToHost && n==51200 && !fired.exchange(true)) {
        selected_slot=dst;
        selected_stream=s; selected=true;
        std::fprintf(stderr,"PINNED_RING_INJECT: routed-ID D2H selected bytes=%zu\n",n);
        if(const char* observe=std::getenv("PINNED_RING_OBSERVE_ONLY"); observe && observe[0]=='1') {
            selected=false;
            return e;
        }
        if(const char* mode=std::getenv("PINNED_RING_INJECT_SUBMIT"); mode && mode[0]=='1') {
            // A reported submission error with real work queued; the production ring must drain it.
            std::fprintf(stderr,"PINNED_RING_INJECT: reporting submission error after real enqueue\n");
            selected=false;
            return hipErrorUnknown;
        }
    }
    return e;
}
extern "C" hipError_t hipFreeHost(void* p) {
    if(p==selected_slot.load() && reported_error.load())
        std::fprintf(stderr,"PINNED_RING_INJECT: releasing selected staging slot after reported error\n");
    return next<hipError_t(*)(void*)>("hipFreeHost")(p);
}
extern "C" hipError_t hipStreamSynchronize(hipStream_t s) {
    const auto e=next<hipError_t(*)(hipStream_t)>("hipStreamSynchronize")(s);
    if(e==hipSuccess && selected && s==selected_stream) {
        selected=false;
        reported_error=true;
        std::fprintf(stderr,"PINNED_RING_INJECT: reporting sync error after successful real drain\n");
        return hipErrorUnknown;
    }
    if(e==hipSuccess && reported_error.load()) {
        if(const char* sticky=std::getenv("PINNED_RING_INJECT_STICKY"); sticky && sticky[0]=='1') {
            std::fprintf(stderr,"PINNED_RING_INJECT: reporting cleanup error after successful real drain\n");
            return hipErrorUnknown;
        }
    }
    return e;
}
