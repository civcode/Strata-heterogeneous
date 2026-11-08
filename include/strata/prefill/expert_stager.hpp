#pragma once
#include <cuda_runtime.h>
#include "strata/platform/aux_cpus.hpp"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace strata::prefill::detail {
struct StagerFailure : std::runtime_error {
    explicit StagerFailure(const char* message = "prefill: expert staging failed; restart the engine")
        : std::runtime_error(message) {}
};

// Source is ExpertSource in production. Keeping the source interface parametric
// permits deterministic tests of this exact worker without a model or GPU.
template<class Source> struct ExpertStager {
    int kRing = 16;
    static constexpr int kMaxBatch = 32;
    int batch = 1;
    struct Job { const uint8_t* src; size_t bytes; Source* from = nullptr; int32_t l = 0, e = 0; };
    std::vector<uint8_t*> buf;
    std::vector<char> pinned;
    std::vector<std::unique_ptr<uint8_t[]>> pageable;
    std::vector<cudaEvent_t> dma_done;
    std::vector<Job> jobs;
    std::unique_ptr<std::atomic<int>[]> ready;
    std::unique_ptr<std::atomic<char>[]> skipped;
    size_t ready_cap = 0;
    std::atomic<uint64_t> head{0};
    std::atomic<int> issued{0}, active{0};
    std::atomic<bool> failed{false}, cancelled{false};
    uint32_t gen = 0;
    bool quit = false, sleep = false;
    std::mutex mu;
    std::condition_variable cv;
    std::vector<std::thread> threads;
    int device = -1;
    // Issuer-only fields, read by destruction only after the issuer and workers join.
    cudaStream_t copy_stream = nullptr;
    bool submitted = false;
    size_t slot_bytes = 0;

    void fail() noexcept {
        failed.store(true, std::memory_order_release);
        issued.store(1 << 30, std::memory_order_release); // wake, NOT a completion publication
#if defined(__cpp_lib_atomic_wait)
        issued.notify_all();
#endif
    }
    void check() const { if (failed.load(std::memory_order_acquire)) throw StagerFailure(); }
    void guard(cudaError_t status) {
        if (status != cudaSuccess) { fail(); throw StagerFailure(); }
    }
    bool init(size_t blob_bytes, int nthreads, bool use_sleep = false, bool force_pageable = false) {
        check();
        if (!buf.empty() || quit || nthreads <= 0 || blob_bytes == 0) return false;
        slot_bytes = blob_bytes;
        sleep = use_sleep;
        if (cudaGetDevice(&device) != cudaSuccess) { fail(); return false; }
        if (const char* v = std::getenv("STRATA_STAGER_RING")) kRing = std::clamp(std::atoi(v), 2, 256);
        buf.assign((size_t) kRing, nullptr);
        pinned.assign((size_t) kRing, 0);
        dma_done.assign((size_t) kRing, nullptr);
        pageable.resize(kRing);
        for (int i = 0; i < kRing; ++i) {
            pinned[i] = !force_pageable && cudaHostAlloc((void**) &buf[i], blob_bytes, cudaHostAllocDefault) == cudaSuccess;
            if (!pinned[i]) {
                (void) cudaGetLastError();
                pageable[i] = std::make_unique<uint8_t[]>(blob_bytes);
                buf[i] = pageable[i].get();
            }
            if (cudaEventCreateWithFlags(&dma_done[i], sleep ? (cudaEventDisableTiming | cudaEventBlockingSync)
                                                           : cudaEventDisableTiming) != cudaSuccess) {
                fail(); return false;
            }
        }
        for (int t = 0; t < nthreads; ++t) threads.emplace_back([this] {
            strata::platform::name_this_thread("strata-stage");
            const char* keep = std::getenv("STRATA_AUX_STAGER");
            if (keep && std::atoi(keep) == 0) strata::aux_cpus::note_owned_thread();
            else strata::aux_cpus::pin_current_thread();
            work();
        });
        return true;
    }
    ~ExpertStager() {
        stop();
        if (buf.empty()) return;
        int previous = -1;
        const bool selected = cudaGetDevice(&previous) == cudaSuccess &&
                              (previous == device || cudaSetDevice(device) == cudaSuccess);
        const bool idle = !submitted || (selected && cudaStreamSynchronize(copy_stream) == cudaSuccess);
        if (!idle || !selected) {
            // Detach pageable owners too: their vector destructors must not free active H2D sources.
            for (auto& p : pageable) (void) p.release();
            std::fprintf(stderr, "strata prefill: retaining %zu expert staging slots (%zu bytes); device/completion is unknown\n",
                         buf.size(), buf.size() * slot_bytes);
        } else if (selected) {
            for (size_t i = 0; i < buf.size(); ++i) {
                if (dma_done[i]) (void) cudaEventDestroy(dma_done[i]);
                if (buf[i] && pinned[i]) (void) cudaFreeHost(buf[i]);
            }
        }
        if (selected && previous != device) (void) cudaSetDevice(previous);
    }
    void stop() noexcept {
        finish();
        { std::lock_guard<std::mutex> lk(mu); quit = true; }
        cv.notify_all();
        for (auto& t : threads) if (t.joinable()) t.join();
    }
    bool stopping() const { return cancelled.load(std::memory_order_acquire) || failed.load(std::memory_order_acquire); }
    void work() noexcept {
        if (cudaSetDevice(device) != cudaSuccess) { fail(); return; }
        uint32_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return quit || gen != seen; });
                if (quit) return;
                seen = gen;
            }
            for (;;) {
                active.fetch_add(1, std::memory_order_acq_rel);
                struct Done { std::atomic<int>& n; ~Done() { n.fetch_sub(1, std::memory_order_acq_rel); } } done{active};
                if (stopping()) break;
                int k = 1;
                const int j = claim(seen, batch, k);
                if (j < 0) break;
                struct Cancelled {};
                try {
                    for (int i = j; i < j + k; ++i) {
                        if (i >= kRing) {
                            for (;;) {
                                if (stopping()) throw Cancelled{};
                                const int x = issued.load(std::memory_order_acquire);
                                if (x > i - kRing) break;
#if defined(__cpp_lib_atomic_wait)
                                if (sleep) issued.wait(x);
                                else
#endif
                                    std::this_thread::yield();
                            }
                            while (!ready[(size_t) (i - kRing)].load(std::memory_order_acquire)) {
                                if (stopping()) throw Cancelled{};
                                std::this_thread::yield();
                            }
                        }
                        if (stopping()) throw Cancelled{};
                        if (cudaEventSynchronize(dma_done[i % kRing]) != cudaSuccess) throw StagerFailure();
                    }
                    if (stopping()) throw Cancelled{};
                    int32_t rl[kMaxBatch], re[kMaxBatch];
                    uint8_t* rd[kMaxBatch];
                    size_t nr = 0;
                    Source* from = nullptr;
                    for (int i = j; i < j + k; ++i) {
                        if (skipped[(size_t) i].load(std::memory_order_acquire)) continue;
                        const Job& jb = jobs[(size_t) i];
                        if (!jb.from) { std::memcpy(buf[i % kRing], jb.src, jb.bytes); continue; }
                        from = jb.from;
                        rl[nr] = jb.l; re[nr] = jb.e; rd[nr++] = buf[i % kRing];
                    }
                    if (nr && !from->copy_blobs(rl, re, rd, nr)) throw StagerFailure();
                    for (int i = j; i < j + k; ++i) ready[(size_t) i].store(1, std::memory_order_release);
                } catch (const Cancelled&) { break; }
                  catch (...) { fail(); return; }
            }
        }
    }
    int claim(uint32_t g, int want, int& k) {
        uint64_t cur = head.load(std::memory_order_acquire);
        for (;;) {
            if ((uint32_t) (cur >> 32) != g) return -1;
            const int n = (int) ((cur >> 16) & 0xffff), j = (int) (cur & 0xffff);
            if (j >= n) return -1;
            k = std::min(want, n - j);
            if (head.compare_exchange_weak(cur, cur + (uint64_t) k, std::memory_order_acq_rel, std::memory_order_acquire))
                return j;
        }
    }
    // Caller must finish the previous generation first (including on cancellation).
    void start(std::vector<Job>&& js) {
        check();
        if (js.empty()) return;
        std::lock_guard<std::mutex> lk(mu);
        jobs = std::move(js);
        if (ready_cap < jobs.size()) {
            ready_cap = jobs.size() * 2;
            ready.reset(new std::atomic<int>[ready_cap]);
            skipped.reset(new std::atomic<char>[ready_cap]);
        }
        for (size_t i = 0; i < jobs.size(); ++i) {
            ready[i].store(0, std::memory_order_relaxed);
            skipped[i].store(0, std::memory_order_relaxed);
        }
        cancelled.store(false, std::memory_order_release);
        issued.store(0);
        ++gen;
        head.store((uint64_t) gen << 32 | (uint64_t) jobs.size() << 16, std::memory_order_release);
        cv.notify_all();
    }
    void skip(int j) { check(); skipped[(size_t) j].store(1, std::memory_order_release); }
    void publish_issued(int watermark) {
        // Never overwrite fail()/finish()'s wake-up sentinel with a smaller watermark.
        // A worker error may race the issuer after its pre-submission check.
        int previous = issued.load(std::memory_order_acquire);
        while (previous < watermark &&
               !issued.compare_exchange_weak(previous, watermark, std::memory_order_acq_rel,
                                             std::memory_order_acquire)) {}
#if defined(__cpp_lib_atomic_wait)
        issued.notify_all();
#endif
        check();
    }
    void pass(int j) { check(); publish_issued(j); }
    const uint8_t* wait(int j) {
        while (!ready[(size_t) j].load(std::memory_order_acquire)) { check(); std::this_thread::yield(); }
        check();
        return buf[j % kRing];
    }
    // A watermark means a valid event was recorded, never that the DMA completed.
    void issued_one(int j, cudaStream_t copy) {
        if (cudaEventRecord(dma_done[j % kRing], copy) != cudaSuccess) { fail(); throw StagerFailure(); }
        check();
        publish_issued(j + 1);
    }
    void upload(int j, void* dst, size_t bytes, cudaStream_t copy) {
        if (bytes > slot_bytes) { fail(); throw StagerFailure(); }
        const uint8_t* src = wait(j);
        check();
        // All uploads of one Stager use one copy stream (the prefill instance's m.copy).
        if (submitted && copy_stream != copy) { fail(); throw StagerFailure(); }
        copy_stream = copy;
        submitted = true;
        if (cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, copy) != cudaSuccess) {
            fail(); throw StagerFailure();
        }
        issued_one(j, copy);
    }
    void finish() noexcept {
        cancelled.store(true, std::memory_order_release);
        // Acquire the claims' release sequence: their active increments must be visible
        // before deciding the generation is quiescent. A release-only store is insufficient.
        (void) head.exchange((uint64_t) gen << 32, std::memory_order_acq_rel);
        issued.store(1 << 30, std::memory_order_release);
#if defined(__cpp_lib_atomic_wait)
        issued.notify_all();
#endif
        while (active.load(std::memory_order_acquire) != 0) std::this_thread::yield();
    }
};
} // namespace strata::prefill::detail
