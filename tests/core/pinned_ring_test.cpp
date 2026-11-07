// Deterministic ownership tests of the real header against a test-only CUDA-shaped runtime.
// Failed calls may leave work queued. No device faults are induced.
#include "strata/core/pinned_ring.hpp"
#ifndef PINNED_RING_BASELINE
#include "strata/prefill/host_group_readback.hpp"
#endif
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

struct FakeEvent { void* stream = nullptr; size_t seq = 0; };
namespace {
struct Allocation { std::unique_ptr<unsigned char[]> data; size_t n; bool freed = false; };
struct Copy { void* dst; const void* src; size_t n; void* stream; size_t seq; bool done = false; };
std::vector<Allocation> allocations;
std::vector<std::unique_ptr<FakeEvent>> events;
std::vector<Copy> copies;
int current = 0, alloc_calls = 0, submit_calls = 0, stream_waits = 0, event_waits = 0;
int premature_frees = 0, premature_reuses = 0;
bool alloc_fail = false, submit_fail = false, enqueue_on_error = true, sync_fail = false;
bool record_fail = false, event_fail = false, create_fail = false, select_fail = false;
size_t seq = 0;
void* stream = reinterpret_cast<void*>(1);
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); std::exit(1); } } while (0)
bool contains(const Allocation& a, const void* p) {
    auto b = reinterpret_cast<size_t>(a.data.get()), q = reinterpret_cast<size_t>(p);
    return q >= b && q < b + a.n;
}
bool busy(const Allocation& a) {
    for (const auto& c : copies) if (!c.done && (contains(a,c.src) || contains(a,c.dst))) return true;
    return false;
}
void complete(void* s, size_t until = size_t(-1)) {
    for (auto& c : copies) if (!c.done && c.stream == s && c.seq <= until) {
        std::memcpy(c.dst,c.src,c.n); c.done = true;
    }
}
void reset() {
    // The fake alone reclaims its allocations after each case, never the production ring.
    allocations.clear(); events.clear(); copies.clear(); seq = 0; current = 0;
    alloc_calls = submit_calls = stream_waits = event_waits = premature_frees = premature_reuses = 0;
    alloc_fail = submit_fail = sync_fail = record_fail = event_fail = create_fail = select_fail = false;
    enqueue_on_error = true;
}
void clean() { CHECK(premature_frees == 0); CHECK(premature_reuses == 0); }
void baseline_case() {
    reset(); unsigned char src[8] = {1}, dst[8] = {};
    {
        strata::core::PinnedRing r;
        sync_fail = true;
        CHECK(r.d2h_sync(dst,src,sizeof src,stream) != cudaSuccess);
        CHECK(dst[0] == 0); // failed completion must not publish the staged bytes
    }
    CHECK(premature_frees == 0); // fails on the original destructor/release path
}
void baseline_extra(int mode) {
    reset(); unsigned char src[8]={1}, dst[8]={};
    {
        strata::core::PinnedRing r;
        if(mode==1) {
            submit_fail=true; sync_fail=true;
            CHECK(r.d2h_sync(dst,src,8,stream)!=cudaSuccess);
        } else if(mode==2) {
            submit_fail=true; sync_fail=true;
            CHECK(r.h2d(dst,src,8,stream)!=cudaSuccess);
        } else if(mode==3 || mode==5 || mode==6) {
            for(int i=0;i<4;++i)CHECK(r.h2d(dst,src,8,stream)==cudaSuccess);
            event_fail=true;
            if(mode==3)CHECK(!r.acquire(70000));
            if(mode==5)CHECK(!r.reserve(70000));
            if(mode==6)r.release();
        } else if(mode==4) {
            record_fail=true; sync_fail=true;
            CHECK(r.h2d(dst,src,8,stream)!=cudaSuccess);
        }
    }
    clean();
}
#ifndef PINNED_RING_BASELINE
void tests() {
    using strata::core::PinnedRing;
    baseline_case();
    for(int mode=1;mode<=6;++mode)baseline_extra(mode);
    reset(); unsigned char src[70000], dst[70000];
    std::memset(src,0x5a,sizeof src); std::memset(dst,0,sizeof dst);
    { PinnedRing r;
      CHECK(r.d2h_sync(dst,src,0,stream)==cudaSuccess); CHECK(alloc_calls==0);
      for (size_t n : {size_t(1),size_t(65536),sizeof src}) {
          CHECK(r.d2h_sync(dst,src,n,stream)==cudaSuccess); CHECK(std::memcmp(src,dst,n)==0);
      }
      for(int i=0;i<4;++i) CHECK(r.d2h_sync(dst,src,sizeof src,stream)==cudaSuccess);
      int before = alloc_calls;
      for(int i=0;i<12;++i) CHECK(r.d2h_sync(dst,src,sizeof src,stream)==cudaSuccess);
      CHECK(alloc_calls==before); CHECK(!r.poisoned());
    } clean();
    // Submission error: successful drain reclaims safely, but still returns the submission error.
    for(bool enqueued : {false,true}) {
      reset(); enqueue_on_error=enqueued;
      { PinnedRing r; submit_fail=true;
        std::memset(dst,0xa5,8);
        CHECK(r.d2h_sync(dst,src,8,stream)!=cudaSuccess); CHECK(stream_waits==1);
        CHECK(dst[0]==0xa5); // even a successful drain must not publish a failed submission's data
        CHECK(r.poisoned()); CHECK(r.quarantined_bytes()==0);
        r.release(); clean();
      }
    }
    // Unknown completion, explicit release and destruction, all entry points fail closed.
    reset();
    { PinnedRing r; submit_fail=true; sync_fail=true;
      CHECK(r.d2h_sync(dst,src,8,stream)!=cudaSuccess);
      CHECK(r.quarantined_bytes()==65536);
      const int a=alloc_calls, n=submit_calls;
      for(int i=0;i<100;++i) {
        CHECK(r.acquire(100000)==nullptr); CHECK(!r.reserve(100000));
        CHECK(r.h2d(dst,src,8,stream)!=cudaSuccess);
        CHECK(r.d2h_sync(dst,src,8,stream)!=cudaSuccess);
        CHECK(r.submit(dst,src,8,stream)!=cudaSuccess);
        r.release();
      }
      CHECK(alloc_calls==a && submit_calls==n); CHECK(r.poisoned()); clean();
    } clean();
    // H2D ring rotation waits for the previous access before returning its slot, including growth.
    reset();
    { PinnedRing r;
      for(int i=0;i<4;++i) CHECK(r.h2d(dst,src,8,stream)==cudaSuccess);
      CHECK(event_waits==0); CHECK(alloc_calls==4);
      CHECK(r.h2d(dst,src,sizeof src,stream)==cudaSuccess); CHECK(event_waits==1);
      CHECK(alloc_calls==5); clean();
      CHECK(r.reserve(sizeof src)); clean();
    } clean();
    // The caller can overwrite its original source as soon as a staged H2D returns.
    reset();
    { PinnedRing r; unsigned char input[8]={7}, output[8]={};
      CHECK(r.h2d(output,input,8,stream)==cudaSuccess); input[0]=9;
      r.release(); CHECK(output[0]==7); clean(); }
    // Failed event completion on rotation: protect that slot and all other outstanding slots independently.
    reset();
    { PinnedRing r;
      for(int i=0;i<4;++i) CHECK(r.h2d(dst,src,8,stream)==cudaSuccess);
      event_fail=true;
      CHECK(r.acquire(sizeof src)==nullptr); CHECK(r.poisoned());
      r.release(); CHECK(r.quarantined_bytes()==4*65536); clean();
    } clean();
    // One unknown slot does not stop release draining and reclaiming other independent slots.
    reset();
    { PinnedRing r;
      for(int i=0;i<4;++i) CHECK(r.h2d(dst,src,8,stream)==cudaSuccess);
      event_fail=true; CHECK(r.acquire(8)==nullptr);
      event_fail=false; r.release(); CHECK(r.quarantined_bytes()==65536); clean();
    } clean();
    // Event record failure: drain, propagate the original failure, and protect an unknown drain.
    for(bool fail : {false,true}) {
      reset();
      { PinnedRing r; record_fail=true; sync_fail=fail;
        CHECK(r.h2d(dst,src,8,stream)!=cudaSuccess); CHECK(stream_waits==1);
        CHECK(r.quarantined_bytes()==(fail ? 65536u : 0u)); CHECK(r.poisoned());
      } clean();
    }
    // H2D submission error is also drained, never treated as evidence of no work.
    reset();
    { PinnedRing r; submit_fail=true; sync_fail=true;
      CHECK(r.h2d(dst,src,8,stream)!=cudaSuccess); CHECK(r.quarantined_bytes()==65536);
    } clean();
    // Clean allocation failure must not submit a transfer using unowned caller memory.
    reset();
    { PinnedRing r; alloc_fail=true;
      CHECK(r.h2d(dst,src,8,stream)==cudaErrorMemoryAllocation);
      CHECK(r.d2h_sync(dst,src,8,stream)==cudaErrorMemoryAllocation);
      CHECK(submit_calls==0); CHECK(!r.poisoned());
      alloc_fail=false; CHECK(r.h2d(dst,src,8,stream)==cudaSuccess);
    } clean();
    reset();
    { PinnedRing r; create_fail=true;
      CHECK(r.acquire(8)==nullptr); CHECK(submit_calls==0);
      create_fail=false; CHECK(r.d2h_sync(dst,src,8,stream)==cudaSuccess);
    } clean();
    // Reserve cannot grow a slot whose event failed.
    reset();
    { PinnedRing r; CHECK(r.h2d(dst,src,8,stream)==cudaSuccess); event_fail=true;
      CHECK(!r.reserve(sizeof src)); CHECK(r.poisoned());
    } clean();
    // Release on another current device must select the owner and restore the caller's device.
    reset();
    { PinnedRing r; CHECK(r.h2d(dst,src,8,stream)==cudaSuccess); current=1;
      r.release(); CHECK(current==1); CHECK(r.quarantined_bytes()==0);
    } clean();
    reset();
    { PinnedRing r; CHECK(r.h2d(dst,src,8,stream)==cudaSuccess); current=1; select_fail=true;
      r.release(); CHECK(r.poisoned()); CHECK(r.quarantined_bytes()==65536);
    } clean();
    // Acquired (not submitted) storage can be released; a second acquire cannot steal it.
    reset();
    { PinnedRing r; CHECK(r.acquire(8)); CHECK(!r.acquire(8)); CHECK(submit_calls==0); }
    clean();
    // Exact host-grouping gate used before production on_routes/CPU grouping. Neither mapped
    // producer failure nor optional download failure can invoke a consumer, even on retry.
    for(bool failed_wait : {false,true}) {
      reset(); int consumed=0;
      strata::prefill::detail::HostGroupReadback gate;
      sync_fail=failed_wait;
      const auto submitted=failed_wait ? cudaSuccess : cudaErrorUnknown;
      if(gate.finish(stream,submitted)) ++consumed;
      CHECK(gate.failed()); CHECK(consumed==0); CHECK(stream_waits==1);
      sync_fail=false;
      if(gate.finish(stream)) ++consumed;
      CHECK(consumed==0 && stream_waits==1);
    }
    // The production grouping cleanup retains mapped output/activation storage after a failed
    // fence; repeated cleanup neither frees it nor adds to the retention count.
    reset();
    { strata::prefill::detail::HostGroupReadback gate; void* mapped=nullptr;
      CHECK(cudaHostAlloc(&mapped,65536,0)==cudaSuccess);
      CHECK(cudaMemcpyAsync(mapped,src,8,cudaMemcpyDeviceToHost,stream)==cudaSuccess);
      sync_fail=true; CHECK(!gate.finish(stream));
      gate.release_host(mapped,65536); CHECK(mapped==nullptr && gate.retained_bytes==65536);
      gate.release_host(mapped,65536); CHECK(gate.retained_bytes==65536); clean(); }
    reset();
    { strata::prefill::detail::HostGroupReadback gate; void* mapped=nullptr;
      CHECK(cudaHostAlloc(&mapped,65536,0)==cudaSuccess);
      CHECK(cudaMemcpyAsync(mapped,src,8,cudaMemcpyDeviceToHost,stream)==cudaSuccess);
      CHECK(gate.finish(stream)); CHECK(!gate.failed());
      gate.release_host(mapped,65536); CHECK(mapped==nullptr && gate.retained_bytes==0); clean(); }
    // Exact teardown fence used by Prefill before destroying its staging owners. A failed drain
    // must refuse cleanup; even when one wait fails all other participating streams are checked.
    reset();
    { CHECK(strata::prefill::detail::drain_prefill_streams(stream,{nullptr,stream}));
      CHECK(stream_waits==2);
      CHECK(strata::prefill::detail::drain_prefill_streams(nullptr)); // default stream is not absent
      CHECK(stream_waits==3);
      sync_fail=true;
      CHECK(!strata::prefill::detail::drain_prefill_streams(stream,{nullptr,stream}));
      CHECK(stream_waits==5); }
    // Healthy legacy blocking calls complete exactly; a zero-byte call allocates nothing.
    reset();
    CHECK(PinnedRing::direct(dst,src,8,stream)==cudaSuccess);
    CHECK(std::memcmp(dst,src,8)==0); clean();
    { const int a=alloc_calls;
      CHECK(PinnedRing::direct(nullptr,nullptr,0,stream)==cudaSuccess); CHECK(alloc_calls==a); }
    // The legacy blocking entry point has a persistent per-thread owner, so repeated errors
    // cannot construct new staging allocations after an unconfirmed completion.
    reset();
    submit_fail=true; sync_fail=true;
    CHECK(PinnedRing::direct(dst,src,8,stream)!=cudaSuccess);
    { const int a=alloc_calls, n=submit_calls;
      for(int i=0;i<100;++i)CHECK(PinnedRing::direct(dst,src,8,stream)!=cudaSuccess);
      CHECK(alloc_calls==a && submit_calls==n); clean(); }
    std::puts("pinned_ring_test: all deterministic ownership and grouping-gate cases passed");
}
#endif
} // namespace

cudaError_t cudaGetDevice(int* p) { *p=current; return cudaSuccess; }
cudaError_t cudaSetDevice(int d) { if(select_fail)return cudaErrorUnknown; current=d; return cudaSuccess; }
cudaError_t cudaGetLastError() { return cudaSuccess; }
const char* cudaGetErrorString(cudaError_t) { return "injected"; }
cudaError_t cudaHostAlloc(void** p,size_t n,unsigned) {
    ++alloc_calls; if(alloc_fail)return cudaErrorMemoryAllocation;
    allocations.push_back({std::unique_ptr<unsigned char[]>(new unsigned char[n]),n,false});
    *p=allocations.back().data.get(); return cudaSuccess;
}
cudaError_t cudaFreeHost(void* p) {
    for(auto& a:allocations) if(a.data.get()==p) { if(busy(a))++premature_frees; CHECK(!a.freed); a.freed=true; }
    return cudaSuccess;
}
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e,unsigned) {
    if(create_fail)return cudaErrorMemoryAllocation;
    events.emplace_back(new FakeEvent); *e=events.back().get(); return cudaSuccess;
}
cudaError_t cudaEventRecord(cudaEvent_t e,cudaStream_t s) {
    if(record_fail)return cudaErrorUnknown; e->stream=s; e->seq=seq; return cudaSuccess;
}
cudaError_t cudaEventSynchronize(cudaEvent_t e) {
    ++event_waits; if(event_fail)return cudaErrorUnknown; complete(e->stream,e->seq); return cudaSuccess;
}
cudaError_t cudaEventDestroy(cudaEvent_t) { return cudaSuccess; }
cudaError_t cudaStreamSynchronize(cudaStream_t s) {
    ++stream_waits; if(sync_fail)return cudaErrorUnknown; complete(s); return cudaSuccess;
}
cudaError_t cudaMemcpyAsync(void* dst,const void* src,size_t n,cudaMemcpyKind,cudaStream_t s) {
    ++submit_calls;
    for(auto& a:allocations) if(contains(a,src)||contains(a,dst)) {
        if(a.freed || busy(a))++premature_reuses;
    }
    if(!submit_fail || enqueue_on_error)copies.push_back({dst,src,n,s,++seq,false});
    return submit_fail ? cudaErrorUnknown : cudaSuccess;
}
int main(int argc, char** argv) {
#ifdef PINNED_RING_BASELINE
    if(argc>1)baseline_extra(std::atoi(argv[1])); else baseline_case();
#else
    (void)argc; (void)argv;
    tests();
#endif
}
