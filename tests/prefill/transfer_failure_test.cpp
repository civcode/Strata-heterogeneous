// The exact production landing arena, worker and query gate, with delayed fake GPU work.
#include "strata/core/remote_readback.hpp"
#include "strata/core/event_completion.hpp"
#include "strata/prefill/expert_stager.hpp"
#include <array>
#include <chrono>
#include <iostream>
#include <map>
#include <string>

struct FakeEvent { size_t end = 0; };
namespace {
std::mutex runtime_mu;
struct Work { void* dst; const void* src; size_t n; std::vector<uint8_t> expected; bool done = false; };
std::vector<Work> pending;
std::map<void*, size_t> allocations;
std::vector<FakeEvent*> events;
int allocs = 0, copies = 0, frees = 0, unsafe = 0;
std::array<void*, 2> watched_pageable{};
int pageable_frees = 0;
int fail_copy = 0, fail_record = 0, fail_wait = 0, waits = 0;
bool fail_sync = false, fail_create = false, fail_alloc = false, source_error = false;
int not_ready = 0;
bool query_error = false;
void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
void complete(size_t end) {
    for (size_t i = 0; i < end; ++i) if (!pending[i].done) {
        auto& w = pending[i];
        if (std::memcmp(w.src, w.expected.data(), w.n) != 0) ++unsafe;
        std::memcpy(w.dst, w.src, w.n); w.done = true;
    }
}
struct Source {
    bool copy_blobs(const int32_t*, const int32_t*, uint8_t* const*, size_t) { return !source_error; }
};
#ifdef STRATA_STAGER_ORIGINAL_HEADER
bool force_pageable() { return false; }
bool stager_sleep() { return false; }
namespace core { using ExpertSource = Source; }
#include STRATA_STAGER_ORIGINAL_HEADER
using Worker = Stager;
void upload(Worker& s, int j, void* dst, size_t n) {
    auto* p = s.wait(j);
    cudaMemcpyAsync(dst, p, n, cudaMemcpyHostToDevice, nullptr);
    s.issued_one(j, nullptr);
}
#else
using Worker = strata::prefill::detail::ExpertStager<Source>;
void upload(Worker& s, int j, void* dst, size_t n) { s.upload(j, dst, n, nullptr); }
#endif
using Bytes = std::array<uint8_t, 16>;
void remote(const std::string& mode) {
    Bytes a{}, b{}, out{}, out2{}; a.fill(17); b.fill(29); out.fill(77); out2.fill(88);
    const uint8_t* src[] = {a.data(), nullptr, b.data()};
    uint8_t* dst[] = {out.data(), nullptr, out2.data()};
    size_t sizes[] = {a.size(), 0, b.size()};
    if (mode == "remote-submit") fail_copy = 2;
    if (mode == "remote-submit-drain") fail_copy = 2;
    if (mode == "remote-sync" || mode == "remote-submit") fail_sync = true;
    if (mode == "remote-alloc") fail_alloc = true;
    {
        strata::core::detail::RemoteReadback r;
        const auto e = r.copy(src, dst, sizes, 3, nullptr);
        if (mode == "remote-ok") {
            require(e == cudaSuccess && out == a && out2 == b, "remote exact publication");
            const int n = allocs;
            require(r.copy(src, dst, sizes, 3, nullptr) == cudaSuccess && allocs == n, "remote steady allocation");
            r.release_completed();
        } else {
            require(e != cudaSuccess, "remote rejection");
            require(out[0] == 77 && out2[0] == 88, "remote must not publish partial data");
            for (const auto& w : pending) require(w.dst != out.data() && w.dst != out2.data(), "borrowed dst submitted to GPU");
            if (mode != "remote-alloc") {
                const int n = allocs, c = copies;
                for (int i = 0; i < 100; ++i) require(r.copy(src, dst, sizes, 3, nullptr) != cudaSuccess, "remote terminal retry");
                require(allocs == n && copies == c, "remote terminal allocation/submission growth");
            }
            if (!r.unknown_completion()) r.release_completed();
            else require(frees == 0 && r.capacity() == 32, "remote unknown storage retained");
        }
    }
    require(unsafe == 0, "remote unsafe release");
    // Caller may now overwrite/destroy its destinations; queued GPU writes target only the owned arena.
    out.fill(99); out2.fill(99);
    { std::lock_guard<std::mutex> lk(runtime_mu); complete(pending.size()); }
    require(out[0] == 99 && out2[0] == 99, "remote late write into caller storage");
}
void stager(const std::string& mode) {
    Bytes a{}, b{}, c{}, out{}, out2{}, out3{}; a.fill(1); b.fill(2); c.fill(3);
    bool rejected = false;
    Source source;
    {
        Worker s; s.kRing = 2;
        if (mode == "stager-batch") s.batch = 2;
        if (mode == "stager-create") fail_create = true;
#ifdef STRATA_STAGER_ORIGINAL_HEADER
        const bool initialized = s.init(a.size(), 1);
#else
        const bool initialized = s.init(a.size(), mode == "stager-concurrent" ? 4 : 1, true, mode == "stager-pageable");
        if (mode == "stager-pageable") watched_pageable = {s.pageable[0].get(), s.pageable[1].get()};
#endif
        if (mode == "stager-create") { require(!initialized, "event creation failure accepted"); return; }
        require(initialized, "stager init");
        if (mode == "stager-wait" || mode == "stager-wait-drain") fail_wait = 3; // initial two unrecorded events, then slot 0 reuse
        if (mode == "stager-record" || mode == "stager-record-drain" || mode == "stager-pageable") fail_record = 1;
        if (mode == "stager-submit") fail_copy = 1;
        if (mode == "stager-record" || mode == "stager-submit" || mode == "stager-wait" || mode == "stager-pageable") fail_sync = true;
        source_error = mode == "stager-source";
        if (source_error) s.start({{nullptr, a.size(), &source}});
        else s.start({{a.data(), a.size()}, {b.data(), b.size()}, {c.data(), c.size()}});
        try {
            upload(s, 0, out.data(), out.size() + (mode == "stager-oversize" ? 1 : 0));
            if (mode == "stager-early-cancel") {
                s.finish();
                require(cudaStreamSynchronize(nullptr) == cudaSuccess && out == a, "cancel lost active source");
                return;
            }
            upload(s, 1, out2.data(), out2.size());
            upload(s, 2, out3.data(), out3.size());
        } catch (const strata::prefill::detail::StagerFailure&) { rejected = true; }
        if (mode == "stager-cancel") s.finish();
        if (mode == "stager-ok" || mode == "stager-concurrent" || mode == "stager-batch" || mode == "stager-cancel") {
            require(!rejected, "healthy worker rejected");
#ifndef STRATA_STAGER_ORIGINAL_HEADER
            const int warm_allocs = allocs;
            require(!s.init(a.size(), 1) && allocs == warm_allocs, "live worker reinitialized its slots");
#endif
            require(cudaStreamSynchronize(nullptr) == cudaSuccess, "stager completion");
            require(out == a && out2 == b && out3 == c, "stager exact bytes / no slot overwrite");
            s.finish();
            s.start({{b.data(), b.size()}}); // worker must survive healthy finish
            upload(s, 0, out.data(), out.size());
            require(cudaStreamSynchronize(nullptr) == cudaSuccess && out == b, "second generation");
        } else {
            s.finish();
            require(rejected, "worker API error accepted as reusable/completed");
#ifndef STRATA_STAGER_ORIGINAL_HEADER
            require(s.failed.load(), "worker error not published");
            const int sentinel = s.issued.load();
            try { s.publish_issued(1); } catch (const strata::prefill::detail::StagerFailure&) {}
            require(s.issued.load() == sentinel, "issuer overwrote the terminal wake-up sentinel");
            const int n = allocs, count = copies;
            for (int i = 0; i < 100; ++i) {
                bool failed = false;
                try { s.start({{a.data(), a.size()}}); } catch (const strata::prefill::detail::StagerFailure&) { failed = true; }
                require(failed, "terminal worker restarted");
                failed = false;
                try { (void) s.init(a.size(), 1); } catch (const strata::prefill::detail::StagerFailure&) { failed = true; }
                require(failed, "terminal worker reinitialized");
            }
            require(allocs == n && copies == count, "worker terminal resource growth");
#endif
        }
    }
    require(unsafe == 0, "stager freed/reused active GPU source");
    if (mode == "stager-pageable") require(pageable_frees == 0, "pageable owner freed an uncertain H2D source");
    if (rejected && fail_sync) require(frees == 0, "unknown staging source freed at shutdown");
    if (rejected && !fail_sync) require(frees == 2, "successful cleanup fence did not release idle slots");
}
void query(const std::string& mode) {
    FakeEvent e;
    not_ready = mode == "query-delayed" || mode == "query-late-error" ? 4 : 0;
    query_error = mode == "query-error" || mode == "query-late-error";
#ifdef STRATA_TEST_OLD_QUERY
    while (cudaEventQuery(&e) == cudaErrorNotReady) {}
    const auto result = cudaSuccess;
#else
    const auto result = strata::core::wait_event_completion(&e);
#endif
    bool cpu_read = result == cudaSuccess;
    require(cpu_read == !query_error, "query error authorized CPU read");
}
}
// Track actual pageable unique_ptr deletion without releasing its backing while fake work
// is outstanding. This makes a regression an assertion, not an intentional CPU/GPU fault.
void* operator new[](std::size_t n) { return ::operator new(n); }
void operator delete[](void* p) noexcept {
    if (p && (p == watched_pageable[0] || p == watched_pageable[1])) {
        ++pageable_frees; return;
    }
    ::operator delete(p);
}
void operator delete[](void* p, std::size_t) noexcept { ::operator delete[](p); }

cudaError_t cudaGetDevice(int* p) { *p = 0; return cudaSuccess; }
cudaError_t cudaSetDevice(int) { return cudaSuccess; }
cudaError_t cudaGetLastError() { return cudaSuccess; }
const char* cudaGetErrorString(cudaError_t) { return "injected"; }
cudaError_t cudaHostAlloc(void** p, size_t n, unsigned) {
    std::lock_guard<std::mutex> lk(runtime_mu);
    if (fail_alloc) return cudaErrorMemoryAllocation;
    *p = new uint8_t[n]; allocations[*p] = n; ++allocs; return cudaSuccess;
}
cudaError_t cudaFreeHost(void* p) {
    std::lock_guard<std::mutex> lk(runtime_mu);
    const auto base = reinterpret_cast<uintptr_t>(p);
    const auto bytes = allocations.at(p);
    for (const auto& w : pending) if (!w.done) {
        const auto src = reinterpret_cast<uintptr_t>(w.src), dst = reinterpret_cast<uintptr_t>(w.dst);
        if ((src >= base && src < base + bytes) || (dst >= base && dst < base + bytes)) ++unsafe;
    }
    ++frees; // retain fixture backing until after assertions, even when original code unsafely frees
    return cudaSuccess;
}
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e, unsigned) {
    std::lock_guard<std::mutex> lk(runtime_mu);
    if (fail_create) return cudaErrorUnknown;
    *e = new FakeEvent; events.push_back(*e); return cudaSuccess;
}
cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t) {
    std::lock_guard<std::mutex> lk(runtime_mu);
    if (fail_record && --fail_record == 0) return cudaErrorUnknown;
    e->end = pending.size(); return cudaSuccess;
}
cudaError_t cudaEventSynchronize(cudaEvent_t e) {
    std::lock_guard<std::mutex> lk(runtime_mu);
    if (++waits == fail_wait) return cudaErrorUnknown;
    complete(e->end); return cudaSuccess;
}
cudaError_t cudaEventQuery(cudaEvent_t) {
    if (not_ready > 0) { --not_ready; return cudaErrorNotReady; }
    return query_error ? cudaErrorUnknown : cudaSuccess;
}
cudaError_t cudaEventDestroy(cudaEvent_t) { return cudaSuccess; }
cudaError_t cudaStreamSynchronize(cudaStream_t) {
    std::lock_guard<std::mutex> lk(runtime_mu);
    if (fail_sync) return cudaErrorUnknown;
    complete(pending.size()); return cudaSuccess;
}
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t n, cudaMemcpyKind, cudaStream_t) {
    std::lock_guard<std::mutex> lk(runtime_mu);
    pending.push_back({dst, src, n, std::vector<uint8_t>((const uint8_t*) src, (const uint8_t*) src + n)});
    ++copies;
    return copies == fail_copy ? cudaErrorUnknown : cudaSuccess; // queued work despite API error
}
int main(int argc, char** argv) {
    int result = 0;
    try {
        require(argc == 2, "one case required");
        const std::string mode = argv[1];
        if (mode.find("remote-") == 0) remote(mode);
        else if (mode.find("stager-") == 0) stager(mode);
        else query(mode);
        std::cout << mode << ": PASS\n";
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << " (unsafe_access_or_free=" << unsafe
                  << ", copies=" << copies << ", pinned_frees=" << frees
                  << ", pageable_frees=" << pageable_frees << ")\n";
        result = 1;
    }
    for (const auto& a : allocations) delete[] static_cast<uint8_t*>(a.first);
    // Fixture reclamation only, after all owners/threads have gone and assertions finished.
    for (void* p : watched_pageable) if (p) ::operator delete(p);
    for (auto* e : events) delete e;
    return result;
}
