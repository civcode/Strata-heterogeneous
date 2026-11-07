// #1996: pageable uploads go through four event-guarded pinned bounce slots. One ring belongs to one
// device and is used by one thread at a time. A caller may change its source after h2d() returns.
//
// Completion failure is NOT completion. On any transfer/wait error the ring fails closed. Slots whose
// completion is unknown are deliberately retained until process exit, including on destruction. No new
// allocation/submission is allowed after poisoning; retention is bounded by the four existing capacities.
// Clean allocation failure returns an error: a pageable fallback cannot protect caller-owned memory if
// its drain fails. STRATA_TEST_PAGEABLE therefore refuses acquisition rather than submitting such a copy.
#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::core {

class PinnedRing {
public:
    static constexpr int kSlots = 4;
    PinnedRing() = default;
    PinnedRing(const PinnedRing&) = delete;
    PinnedRing& operator=(const PinnedRing&) = delete;
    ~PinnedRing() { release(); }

    bool poisoned() const { return poisoned_; }
    cudaError_t status() const { return error_; }
    // Upper bound retained by this ring; after destruction the runtime owns reclamation at process exit.
    size_t quarantined_bytes() const {
        size_t n = 0;
        for (const Slot& sl : slots_) if (sl.state == State::unknown) n += sl.cap;
        return n;
    }

    cudaError_t h2d(void* dst, const void* src, size_t bytes, cudaStream_t s) {
        if (poisoned_) return error_;
        if (bytes == 0) return error_ = cudaSuccess;
        void* stage = acquire(bytes);
        if (!stage) return error_;
        std::memcpy(stage, src, bytes);
        return submit(dst, stage, bytes, s);
    }

    // Acquire/submit is a single-threaded pair. A second acquisition must not steal an acquired slot.
    void* acquire(size_t bytes) {
        if (poisoned_) return nullptr;
        error_ = cudaSuccess;
        if (acquired_) { fail(cudaErrorInvalidValue); return nullptr; }
        if (bytes == 0) return nullptr;
        if (pageable_forced()) { error_ = cudaErrorMemoryAllocation; return nullptr; }
        int cur = -1;
        cudaError_t e = cudaGetDevice(&cur);
        if (e != cudaSuccess) { fail(e); return nullptr; }
        if (dev_ < 0) dev_ = cur;
        if (cur != dev_) { error_ = cudaErrorInvalidDevice; return nullptr; }
        Slot& sl = slots_[next_];
        next_ = (next_ + 1) % kSlots;
        if (!wait(sl) || !grow(sl, bytes)) return nullptr;
        sl.state = State::acquired;
        acquired_ = &sl;
        return sl.p;
    }

    cudaError_t submit(void* dst, const void* stage, size_t bytes, cudaStream_t s) {
        if (poisoned_) return error_;
        Slot* sl = acquired_;
        if (!sl || stage != sl->p || bytes > sl->cap) return fail(cudaErrorInvalidValue);
        acquired_ = nullptr;
        sl->state = State::idle;
        if (bytes == 0) return cudaSuccess;
        // A failed submit can report an earlier asynchronous error; drain even on failure.
        sl->state = State::unknown;
        cudaError_t e = cudaMemcpyAsync(dst, stage, bytes, cudaMemcpyHostToDevice, s);
        if (e == cudaSuccess) e = cudaEventRecord(sl->ev, s);
        if (e != cudaSuccess) return drain(*sl, s, e);
        sl->state = State::pending;
        return cudaSuccess;
    }

    // Kept for callers needing a blocking upload. The per-thread owner also fails closed across calls;
    // no caller-owned host memory is ever submitted directly. Healthy release permits device changes.
    static cudaError_t direct(void* dst, const void* src, size_t bytes, cudaStream_t s) {
        thread_local PinnedRing ring;
        const cudaError_t e = ring.h2d(dst, src, bytes, s);
        ring.release();
        return e != cudaSuccess ? e : ring.status();
    }

    // CPU reads only after checked stream completion (also a HIP host-visibility fence).
    cudaError_t d2h_sync(void* dst, const void* src, size_t bytes, cudaStream_t s) {
        if (poisoned_) return error_;
        if (bytes == 0) return error_ = cudaSuccess;
        void* stage = acquire(bytes);
        if (!stage) return error_;
        Slot& sl = *acquired_;
        acquired_ = nullptr;
        sl.state = State::unknown;
        const cudaError_t submitted = cudaMemcpyAsync(stage, src, bytes, cudaMemcpyDeviceToHost, s);
        const cudaError_t e = drain(sl, s, submitted);
        if (e == cudaSuccess) std::memcpy(dst, stage, bytes);
        return e;
    }

    // Grow only after a successful wait. No allocations, even in other slots, after a poisoned wait.
    bool reserve(size_t bytes) {
        if (poisoned_) return false;
        error_ = cudaSuccess;
        if (acquired_) { fail(cudaErrorInvalidValue); return false; }
        if (bytes == 0) return true;
        if (pageable_forced()) { error_ = cudaErrorMemoryAllocation; return false; }
        int cur = -1;
        cudaError_t e = cudaGetDevice(&cur);
        if (e != cudaSuccess) { fail(e); return false; }
        if (dev_ < 0) dev_ = cur;
        if (cur != dev_) { error_ = cudaErrorInvalidDevice; return false; }
        for (Slot& sl : slots_) {
            if (sl.cap >= bytes && sl.ev) continue;
            if (!wait(sl) || !grow(sl, bytes)) return false;
        }
        return true;
    }

    // Best effort on the owning device, restoring the caller's device. Never reset the poison latch.
    // Independently drain other pending slots; never retry a slot whose completion became unknown.
    void release() {
        acquired_ = nullptr;
        if (dev_ < 0) return;
        int cur = -1;
        cudaError_t e = cudaGetDevice(&cur);
        bool selected = e == cudaSuccess;
        if (selected && cur != dev_) { e = cudaSetDevice(dev_); selected = e == cudaSuccess; }
        if (!selected) {
            fail(e);
            for (Slot& sl : slots_) if (sl.p) sl.state = State::unknown;
            return;
        }
        for (Slot& sl : slots_) {
            if (sl.state == State::unknown || !wait(sl)) continue;
            if (sl.ev) {
                e = cudaEventDestroy(sl.ev);
                if (e != cudaSuccess) { sl.state = State::unknown; fail(e); continue; }
                sl.ev = nullptr;
            }
            if (sl.p) {
                e = cudaFreeHost(sl.p);
                if (e != cudaSuccess) { sl.state = State::unknown; fail(e); continue; }
            }
            sl = Slot{};
        }
        if (cur != dev_) {
            e = cudaSetDevice(cur);
            if (e != cudaSuccess) fail(e);
        }
        const size_t retained = quarantined_bytes();
        if (retained && !reported_) {
            std::fprintf(stderr, "strata: pinned ring: retaining %zu bytes until process exit after an unconfirmed completion/cleanup\n", retained);
            reported_ = true;
        }
        bool empty = true;
        for (const Slot& sl : slots_) if (sl.p || sl.ev) empty = false;
        if (empty) { dev_ = -1; next_ = 0; }
    }

private:
    enum class State { idle, acquired, pending, unknown };
    struct Slot {
        void* p = nullptr;
        size_t cap = 0;
        cudaEvent_t ev = nullptr;
        State state = State::idle;
    };
    cudaError_t fail(cudaError_t e) {
        if (!poisoned_) error_ = e;
        poisoned_ = true;
        return error_;
    }
    bool wait(Slot& sl) {
        if (sl.state == State::unknown) return false;
        if (sl.state != State::pending) return true;
        const cudaError_t e = cudaEventSynchronize(sl.ev);
        if (e != cudaSuccess) { sl.state = State::unknown; fail(e); return false; }
        sl.state = State::idle;
        return true;
    }
    bool grow(Slot& sl, size_t bytes) {
        if (sl.cap < bytes) {
            if (sl.p) {
                const cudaError_t e = cudaFreeHost(sl.p);
                if (e != cudaSuccess) { sl.state = State::unknown; fail(e); return false; }
                sl.p = nullptr; sl.cap = 0;
            }
            const size_t cap = bytes < kMinCap ? kMinCap : bytes;
            const cudaError_t e = cudaHostAlloc(&sl.p, cap, cudaHostAllocPortable);
            if (e != cudaSuccess) {
                sl.p = nullptr;
                if (e == cudaErrorMemoryAllocation) error_ = e;
                else fail(e);
                return false;
            }
            sl.cap = cap;
        }
        if (!sl.ev) {
            const cudaError_t e = cudaEventCreateWithFlags(&sl.ev, cudaEventDisableTiming);
            if (e != cudaSuccess) {
                sl.ev = nullptr;
                if (e == cudaErrorMemoryAllocation) error_ = e;
                else fail(e);
                return false;
            }
        }
        return true;
    }
    cudaError_t drain(Slot& sl, cudaStream_t s, cudaError_t submitted) {
        const cudaError_t completed = cudaStreamSynchronize(s);
        if (completed == cudaSuccess) sl.state = State::idle;
        else { sl.state = State::unknown; return fail(completed); }
        return submitted == cudaSuccess ? cudaSuccess : fail(submitted);
    }
    static constexpr size_t kMinCap = 64 * 1024;
    static bool pageable_forced() {
        static const bool v = [] { const char* e = std::getenv("STRATA_TEST_PAGEABLE"); return e && e[0] == '1'; }();
        return v;
    }
    Slot slots_[kSlots];
    Slot* acquired_ = nullptr;
    int next_ = 0, dev_ = -1;
    cudaError_t error_ = cudaSuccess;
    bool poisoned_ = false, reported_ = false;
};

} // namespace strata::core
