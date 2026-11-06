#include "conversation_copy.hpp"
#include "strata/core/conversation_snapshot.hpp"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <string>
#include <thread>
#include <vector>

using namespace strata::core;
using namespace strata::core::conversation_detail;
namespace {
void check(bool ok, const char* why) { if (!ok) { std::fprintf(stderr, "FAIL: %s\n", why); std::exit(1); } }
void ck(cudaError_t e) { check(e == cudaSuccess, cudaGetErrorString(e)); }
#if defined(STRATA_PREPARE_WRAP)
enum class Fault { none, stream, event, select, query };
thread_local Fault fault = Fault::none;
thread_local int fault_device = -1;
std::atomic<int> streams{0}, events{0}, queries{0};
#endif
}
#if defined(STRATA_PREPARE_WRAP)
#if defined(STRATA_USE_HIP)
#define REAL(name) __real_hip##name
#define WRAP(name) __wrap_hip##name
#else
#define REAL(name) __real_cuda##name
#define WRAP(name) __wrap_cuda##name
#endif
extern "C" cudaError_t REAL(StreamCreateWithFlags)(cudaStream_t*, unsigned int);
extern "C" cudaError_t WRAP(StreamCreateWithFlags)(cudaStream_t* s, unsigned int flags) {
    int dev = 0; ck(cudaGetDevice(&dev));
    if (fault == Fault::stream && dev == fault_device) return cudaErrorMemoryAllocation;
    const auto e = REAL(StreamCreateWithFlags)(s, flags); if (e == cudaSuccess) ++streams; return e;
}
extern "C" cudaError_t REAL(EventCreateWithFlags)(cudaEvent_t*, unsigned int);
extern "C" cudaError_t WRAP(EventCreateWithFlags)(cudaEvent_t* event, unsigned int flags) {
    int dev = 0; ck(cudaGetDevice(&dev));
    if (fault == Fault::event && dev == fault_device) return cudaErrorMemoryAllocation;
    const auto e = REAL(EventCreateWithFlags)(event, flags); if (e == cudaSuccess) ++events; return e;
}
extern "C" cudaError_t REAL(SetDevice)(int);
extern "C" cudaError_t WRAP(SetDevice)(int dev) {
    if (fault == Fault::select && dev == fault_device) return cudaErrorMemoryAllocation;
    return REAL(SetDevice)(dev);
}
extern "C" cudaError_t REAL(MemGetInfo)(size_t*, size_t*);
extern "C" cudaError_t WRAP(MemGetInfo)(size_t* f, size_t* t) {
    ++queries;
    if (fault == Fault::query) return cudaErrorMemoryAllocation;
    return REAL(MemGetInfo)(f, t);
}
#endif
int main(int argc, char** argv) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return 77;
    ck(cudaSetDevice(0));
    const std::string mode = argc > 1 ? argv[1] : "reuse";
    const int zero[] = {0};
    std::string error;
    if (mode == "reuse") {
        cudaStream_t first = nullptr; cudaEvent_t first_event = nullptr;
        std::thread a([&] {
            ck(cudaSetDevice(0));
            check(conversation_copy_prepare(zero, 1, error) == ConversationCopyPreparation::ready, "cold prepare");
            first = thread_copy_handles().streams[0]; first_event = thread_copy_handles().events[0];
            for (int i = 0; i < 20; ++i)
                check(conversation_copy_prepare(zero, 1, error) == ConversationCopyPreparation::ready, "owned prepare");
            check(thread_copy_handles().streams[0] == first && thread_copy_handles().events[0] == first_event,
                  "repeated preparation retains exact handles");
        }); a.join();
        std::thread b([&] {
            ck(cudaSetDevice(0));
            check(conversation_copy_prepare(zero, 1, error) == ConversationCopyPreparation::ready, "pooled prepare");
            check(thread_copy_handles().streams[0] == first && thread_copy_handles().events[0] == first_event,
                  "thread exit returns the same stream and event");
        }); b.join();
        check(copy_streams_created() == 1 && copy_events_created() == 1, "no growth on sequential reuse");
    } else if (mode == "contention") {
        std::promise<cudaStream_t> held; std::promise<void> release;
        auto until = release.get_future();
        std::thread holder([&] {
            ck(cudaSetDevice(0)); std::string why;
            check(conversation_copy_prepare(zero, 1, why) == ConversationCopyPreparation::ready, "holder owns resources");
            held.set_value(thread_copy_handles().streams[0]); until.wait();
        });
        const auto other = held.get_future().get();
        check(!copy_stream_ready(0), "held resource is not pooled");
        check(conversation_copy_prepare(zero, 1, error) == ConversationCopyPreparation::ready, "contender prepares");
        check(thread_copy_handles().streams[0] != other, "contender cannot steal held stream");
        const auto own = thread_copy_handles().streams[0];
        release.set_value(); holder.join();
        check(conversation_copy_prepare(zero, 1, error) == ConversationCopyPreparation::ready &&
              thread_copy_handles().streams[0] == own, "ownership retained when another thread exits");
        check(copy_streams_created() == 2 && copy_events_created() == 2, "bounded by simultaneous owners");
#if defined(STRATA_PREPARE_WRAP)
    } else if (mode == "stream" || mode == "event" || mode == "second" || mode == "device" || mode == "query") {
        if ((mode == "second" || mode == "device") && n < 2) return 77;
        const int both[] = {0, 1};
        fault_device = mode == "second" ? 1 : 0;
        fault = mode == "event" ? Fault::event : mode == "query" ? Fault::query : Fault::stream;
        if (mode == "device") { fault = Fault::select; fault_device = 0; }
        const int* devs = mode == "device" ? both + 1 : mode == "second" ? both : zero;
        const size_t count = mode == "second" ? 2 : 1;
        const auto result = conversation_copy_prepare(devs, count, error);
        if (mode == "query") {
            check(result == ConversationCopyPreparation::ready && queries == 0, "memory query failure irrelevant: actual acquisition used");
        } else if (mode == "device") {
            check(result == ConversationCopyPreparation::fatal, "failed device restoration prohibits fallback");
        } else {
            check(result == ConversationCopyPreparation::unavailable && !error.empty(), "allocation failure explicit");
            int cur = -1; ck(cudaGetDevice(&cur)); check(cur == 0, "failure restores original device");
            if (mode == "second") check(thread_copy_handles().streams[0] && thread_copy_handles().events[0],
                                        "first GPU resources remain owned on second GPU failure");
        }
        fault = Fault::none; ck(cudaSetDevice(0));
        check(conversation_copy_prepare(devs, count, error) == ConversationCopyPreparation::ready, "retry after injected failure");
        check(copy_streams_created() == count && copy_events_created() == count, "earlier handles reused, no lost resources");
#endif
    } else if (mode == "wrong-device") {
        if (n < 2) return 77;
        const int one[] = {1};
        check(conversation_copy_prepare(one, 1, error) == ConversationCopyPreparation::ready, "prepare noncurrent GPU");
        int cur = -1; ck(cudaGetDevice(&cur)); check(cur == 0, "successful prepare restores current device");
        const auto stream = thread_copy_handles().streams[1];
        check(!thread_copy_handles().streams[0], "GPU 1 stream not assigned to GPU 0");
        ck(cudaSetDevice(1)); cudaStream_t again = nullptr; ck(private_stream(again));
        check(again == stream, "selected GPU uses its own prepared stream"); ck(cudaSetDevice(0));
    } else { check(false, "unknown mode"); }
    std::printf("conversation_prepare_test: %s PASS\n", mode.c_str());
}
