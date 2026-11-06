// The conversation copy streams and events are pooled across threads (#3350, #3470): short-lived threads (the layer
// split's later stages run each prompt chunk on a fresh std::async thread) must not leave a stream and an event behind
// each.  Synthetic 4 KiB copies, no model; needs a CUDA device (skips, exit 77, without one).
#include "conversation_copy.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

using namespace strata::core::conversation_detail;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

// one thread's work on device `dev`: a queued copy, the ordering event, a synchronous copy
bool one_thread(int dev, void* d, void* h, size_t bytes,
                bool trace = false, size_t baseline_free = 0) {
    if (cudaSetDevice(dev) != cudaSuccess) return false;

    auto mark = [&](const char* stage) {
        if (!trace) return;

        size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
            std::fprintf(stderr, "[VRAM] %s: query failed\n", stage);
            return;
        }

        const long long used =
            (long long)baseline_free - (long long)free_b;

        std::fprintf(stderr,
            "[VRAM] %-24s delta=%lld KiB\n",
            stage, used / 1024);
    };

    mark("after cudaSetDevice");

    if (trace) {
        cudaStream_t stream = nullptr;
        if (private_stream(stream) != cudaSuccess) return false;
        mark("after stream creation");
    }

    if (copy_nonblocking(d, h, bytes, true) != cudaSuccess)
        return false;
    mark("after first copy");

    if (order_after_queued() != cudaSuccess)
        return false;
    mark("after event/wait");

    if (copy_nonblocking(h, d, bytes, false) != cudaSuccess)
        return false;
    mark("after synchronous copy");

    const bool ok = cudaDeviceSynchronize() == cudaSuccess;
    mark("after device sync");

    return ok;
}
}  // namespace

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev < 1) {
        (void) cudaGetLastError();
        std::printf("conversation_copy_streams_test: no CUDA device, skipped\n");
        return 77;
    }
    const bool trace = std::getenv("STRATA_COPY_VRAM_TRACE") != nullptr;
    constexpr size_t bytes = 4096;
    void* d = nullptr;
    check(cudaMalloc(&d, bytes) == cudaSuccess, "device buffer");
    std::vector<char> h(bytes, 7);

    // The first resource acquisition is checked by its results and handle counts, not by device-wide free memory.
    // HIP may allocate substantial runtime resources here; other processes can also change cudaMemGetInfo.
    {
        bool fresh_ready = true;
        std::thread probe([&] { fresh_ready = copy_stream_ready(0); });
        probe.join();
        check(!fresh_ready, "before any copy no thread has a stream for the card (copy_stream_ready is false)");
        check(cudaDeviceSynchronize() == cudaSuccess, "synchronize before first copy");
        size_t free0 = 0, free1 = 0, tot = 0;
        if (trace) check(cudaMemGetInfo(&free0, &tot) == cudaSuccess, "free memory before the first stream");
        const long first_s = copy_streams_created(), first_e = copy_events_created();
        std::vector<char> result(bytes, 0);
        bool ok = false, ready_after = false, ordered = false;
        std::thread t([&] {
            ok = one_thread(0, d, h.data(), bytes, trace, free0);
            ready_after = copy_stream_ready(0);
            ordered = cudaSetDevice(0) == cudaSuccess &&
                cudaMemset(d, 0, bytes) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess &&
                copy_nonblocking(d, h.data(), bytes, true) == cudaSuccess &&
                order_after_queued() == cudaSuccess &&
                cudaMemcpyAsync(result.data(), d, bytes, cudaMemcpyDefault, nullptr) == cudaSuccess &&
                cudaDeviceSynchronize() == cudaSuccess;
        });
        t.join();
        check(ok && ready_after, "the first copy creates the stream, and the thread then has it");
        check(ordered && result == h, "legacy stream observes queued copy after the ordering event");
        check(copy_streams_created() - first_s == 1 && copy_events_created() - first_e == 1,
              "first copy creates exactly one stream and one event");
        if (trace) {
            check(cudaMemGetInfo(&free1, &tot) == cudaSuccess, "free memory after the first stream");
            std::fprintf(stderr, "[VRAM] device-wide first-use delta=%lld KiB (includes other processes)\n",
                         ((long long) free0 - (long long) free1) / 1024);
        }
        bool pooled = false;
        std::thread t2([&] { pooled = copy_stream_ready(0); });
        t2.join();
        check(pooled, "a finished thread's stream is in the pool: the next thread needs no new one");

        // Readiness is not a reservation: another thread can acquire the pooled stream immediately afterward.
        std::atomic<bool> acquired{false}, release{false};
        bool holder_ok = false;
        std::thread holder([&] {
            cudaStream_t stream = nullptr;
            holder_ok = cudaSetDevice(0) == cudaSuccess && private_stream(stream) == cudaSuccess;
            acquired.store(true, std::memory_order_release);
            while (!release.load(std::memory_order_acquire)) std::this_thread::yield();
        });
        while (!acquired.load(std::memory_order_acquire)) std::this_thread::yield();
        check(holder_ok && !copy_stream_ready(0), "a stream held by another thread is not pooled");
        release.store(true, std::memory_order_release);
        holder.join();
        check(copy_stream_ready(0), "the holder returns its stream at thread exit");
    }

    // 1. 200 threads, one after the other (a chunk per std::async thread): the handles are reused, not created per thread
    size_t reuse_free0 = 0, reuse_total = 0;
    if (trace) check(cudaMemGetInfo(&reuse_free0, &reuse_total) == cudaSuccess, "free memory before reuse");
    const long s0 = copy_streams_created(), e0 = copy_events_created();
    for (int i = 0; i < 200; ++i) {
        bool ok = false;
        std::thread t([&] { ok = one_thread(0, d, h.data(), bytes); });
        t.join();
        check(ok, "a short-lived thread's copies succeed");
    }
    check(copy_streams_created() == s0, "200 sequential threads reuse the pooled stream");
    check(copy_events_created() == e0, "200 sequential threads reuse the pooled event");
    if (trace) {
        size_t f = 0, total = 0;
        check(cudaMemGetInfo(&f, &total) == cudaSuccess, "free memory after reuse");
        std::fprintf(stderr, "[VRAM] after 200 reused threads delta=%lld KiB streams=%ld events=%ld\n",
                     ((long long) reuse_free0 - (long long) f) / 1024, copy_streams_created(), copy_events_created());
    }

    // 2. threads at the same time each have their own (the number is bounded by the concurrency)
    const long s1 = copy_streams_created(), e1 = copy_events_created();
    std::atomic<int> fails{0};
    void* concurrent_d[6] = {};
    std::vector<std::vector<char>> concurrent_h(6, std::vector<char>(bytes, 7));
    for (auto& p : concurrent_d) check(cudaMalloc(&p, bytes) == cudaSuccess, "concurrent device buffer");
    for (int round = 0; round < 10; ++round) {
        std::vector<std::thread> ts;
        for (int k = 0; k < 6; ++k)
            ts.emplace_back([&, k] { for (int j = 0; j < 5; ++j)
                if (!one_thread(0, concurrent_d[k], concurrent_h[k].data(), bytes)) ++fails; });
        for (auto& t : ts) t.join();
    }
    for (auto p : concurrent_d) check(cudaFree(p) == cudaSuccess, "free concurrent device buffer");
    check(fails.load() == 0, "concurrent threads' copies succeed");
    check(copy_streams_created() - s1 <= 5, "six concurrent threads need at most five additional streams");
    check(copy_events_created() - e1 <= 5, "six concurrent threads need at most five additional events");
    if (trace) {
        size_t f = 0, total = 0;
        check(cudaMemGetInfo(&f, &total) == cudaSuccess, "free memory after concurrent copies");
        std::fprintf(stderr, "[VRAM] after concurrent threads delta=%lld KiB streams=%ld events=%ld\n",
                     ((long long) reuse_free0 - (long long) f) / 1024, copy_streams_created(), copy_events_created());
    }

    // 3. a second device has its own
    if (n_dev > 1) {
        const long s2 = copy_streams_created(), e2 = copy_events_created();
        bool device1_ready = true;
        std::thread probe1([&] { device1_ready = copy_stream_ready(1); });
        probe1.join();
        check(!device1_ready, "device 0 resources do not make device 1 ready");
        for (int i = 0; i < 50; ++i) {
            bool ok = false;
            std::thread t([&] {
                ok = cudaSetDevice(1) == cudaSuccess;
                void* d1 = nullptr;
                ok = ok && cudaMalloc(&d1, bytes) == cudaSuccess && one_thread(1, d1, h.data(), bytes);
                if (d1) ok = (cudaFree(d1) == cudaSuccess) && ok;
            });
            t.join();
            check(ok, "device 1: a short-lived thread's copies succeed");
        }
        check(copy_streams_created() - s2 == 1, "device 1: exactly one stream for 50 sequential threads");
        check(copy_events_created() - e2 == 1, "device 1: exactly one event for 50 sequential threads");
    }
    check(cudaFree(d) == cudaSuccess, "free device buffer");
    std::printf("conversation_copy_streams_test: %d checks passed (%ld streams, %ld events created in all)\n", checks,
                copy_streams_created(), copy_events_created());
    return 0;
}
