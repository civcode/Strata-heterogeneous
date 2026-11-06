// Small model-independent readback/capture tests. No GPU exhaustion or device-error injection.
#include "conversation_readback.hpp"
#include "conversation_copy.hpp"
#include "strata/core/conversation_snapshot.hpp"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>

#if defined(STRATA_READBACK_LINK_WRAP)
namespace {
enum class Fault { none, pin, submit, sync };
thread_local Fault fault = Fault::none;
thread_local int fail_at = 0, calls = 0;
bool fail(Fault at) { return fault == at && ++calls == fail_at; }
}
extern "C" hipError_t __real_hipHostMalloc(void**, size_t, unsigned int);
extern "C" hipError_t __wrap_hipHostMalloc(void** p, size_t n, unsigned int flags) {
    if (fail(Fault::pin)) { *p = nullptr; return hipErrorOutOfMemory; }
    return __real_hipHostMalloc(p, n, flags);
}
extern "C" hipError_t __real_hipMemcpyAsync(void*, const void*, size_t, hipMemcpyKind, hipStream_t);
extern "C" hipError_t __wrap_hipMemcpyAsync(void* d, const void* s, size_t n, hipMemcpyKind k, hipStream_t stream) {
    if (fail(Fault::submit)) return hipErrorOutOfMemory;
    return __real_hipMemcpyAsync(d, s, n, k, stream);
}
extern "C" hipError_t __real_hipStreamSynchronize(hipStream_t);
extern "C" hipError_t __wrap_hipStreamSynchronize(hipStream_t stream) {
    const hipError_t e = __real_hipStreamSynchronize(stream);
    // Drain real work first. Inject a reported failure without poisoning a GPU context.
    return fail(Fault::sync) ? hipErrorOutOfMemory : e;
}
#endif

using namespace strata::core;
using namespace strata::core::conversation_detail;
namespace {
void check(bool ok, const char* label) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
void ck(cudaError_t e) { check(e == cudaSuccess, cudaGetErrorString(e)); }
std::vector<uint8_t> pattern(size_t n, int salt) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = (uint8_t) ((i * 37 + (i >> 8) + salt) % 251);
    return v;
}
// Submission does not write the destination: only synchronize completes it. This detects premature CPU reads,
// staging reuse, failure to drain after submission failure, and freeing storage with unknown completion.
struct FakeState {
    int allocs = 0, releases = 0, submits = 0, syncs = 0, fail_submit = 0, fail_sync = 0;
    bool fail_alloc = false, in_flight = false;
    void* dst = nullptr;
    const void* src = nullptr;
    size_t n = 0;
    std::vector<std::unique_ptr<uint8_t[]>> allocations;
};
struct FakeOps {
    std::shared_ptr<FakeState> state = std::make_shared<FakeState>();
    cudaError_t allocate(void** p, size_t n) {
        ++state->allocs;
        if (state->fail_alloc) return cudaErrorMemoryAllocation;
        state->allocations.emplace_back(new uint8_t[n]);
        *p = state->allocations.back().get(); return cudaSuccess;
    }
    void release(void*) { check(!state->in_flight, "never free in-flight staging"); ++state->releases; }
    cudaError_t submit(void* dst, const void* src, size_t n, cudaStream_t) {
        check(!state->in_flight, "never reuse in-flight staging");
        state->in_flight = true; state->dst = dst; state->src = src; state->n = n;
        return ++state->submits == state->fail_submit ? cudaErrorMemoryAllocation : cudaSuccess;
    }
    cudaError_t synchronize(cudaStream_t) {
        if (++state->syncs == state->fail_sync) return cudaErrorMemoryAllocation;
        std::memcpy(state->dst, state->src, state->n); state->in_flight = false; return cudaSuccess;
    }
    void progress() {}
};
void deterministic() {
    const size_t n = PinnedReadback<>::piece_bytes + 4099;
    const auto src = pattern(n, 9);
    std::vector<uint8_t> out(n + 2, 0xa5);
    auto state = std::make_shared<FakeState>();
    {
        PinnedReadback<FakeOps> copy; copy.ops.state = state;
        check(copy.copy(out.data() + 1, src.data(), n, nullptr) == cudaSuccess, "chunked readback");
        check(std::equal(src.begin(), src.end(), out.begin() + 1), "all chunks exact");
        check(out.front() == 0xa5 && out.back() == 0xa5, "nonaligned destination sentinels intact");
        check(state->allocs == 1 && state->submits == 2 && state->syncs == 2, "bounded reusable staging");
        state->fail_submit = state->submits + 1;
        std::fill(out.begin(), out.end(), 0xa5);
        check(copy.copy(out.data() + 1, src.data(), n, nullptr) != cudaSuccess, "submission failure propagates");
        check(!state->in_flight && out[1] == 0xa5, "failed submission drained, no final payload published");
    }
    check(state->releases == 1, "successful drain allows staging cleanup");
    state = std::make_shared<FakeState>(); state->fail_alloc = true;
    {
        PinnedReadback<FakeOps> copy; copy.ops.state = state;
        check(copy.copy(out.data(), src.data(), n, nullptr) != cudaSuccess, "pin allocation failure propagates");
        check(state->submits == 0, "no unsafe pageable fallback");
    }
    state = std::make_shared<FakeState>(); state->fail_sync = 1;
    {
        PinnedReadback<FakeOps> copy; copy.ops.state = state;
        check(copy.copy(out.data(), src.data(), n, nullptr) != cudaSuccess, "completion failure propagates");
        check(out[1] == 0xa5, "unknown completion never copied to final destination");
    }
    check(state->releases == 0 && state->in_flight, "unknown-completion storage quarantined, not freed");
}
void capture(int dev, int salt) {
    ck(cudaSetDevice(dev));
    const size_t n = PinnedReadback<>::piece_bytes + 4099;
    auto src = pattern(n, salt);
    void* d = nullptr; ck(cudaMalloc(&d, n));
    ck(cudaMemcpy(d, src.data(), n, cudaMemcpyHostToDevice));
    std::vector<uint8_t> out(n + 2, 0xa5);
    ck(copy_nonblocking(out.data() + 1, d, n, false, true));
    check(std::equal(src.begin(), src.end(), out.begin() + 1), "concurrent pageable readback exact");
    check(out.front() == 0xa5 && out.back() == 0xa5, "readback guards");
    void* pinned = nullptr; ck(cudaMallocHost(&pinned, n));
    ck(copy_nonblocking(pinned, d, n, false, true));
    check(!std::memcmp(pinned, src.data(), n), "pinned destination exact");
    ck(cudaFreeHost(pinned));
    // Restoration consumes the still-owned pageable bytes; synchronize before freeing/reusing them.
    ck(cudaMemset(d, 0, n)); ck(cudaDeviceSynchronize());
    ck(copy_nonblocking(d, out.data() + 1, n));
    std::fill(out.begin(), out.end(), 0xa5);
    ck(copy_nonblocking(out.data() + 1, d, n, false, true));
    check(std::equal(src.begin(), src.end(), out.begin() + 1), "H2D restore round trip exact");
    ck(cudaFree(d));

    ModelGeometry g;
    g.n_layers = 3; g.ssm_state_size = 2; g.ssm_v_heads = 2; g.ssm_conv_channels = 8;
    g.n_head_kv = 1; g.head_dim = 64; g.idx_key_dim = 8;
    ConversationStateSizes z; std::string error;
    check(conversation_state_sizes(g, z, error), "small snapshot geometry");
    SessionState ss; ss.max_cells = 96; ss.layer_hi = g.n_layers;
    ss.gdn_alloc = g.n_gdn_layers(); ss.qsa_alloc = g.n_qsa_layers();
    check(ss.qsa_alloc == 0, "fixture is GDN-only");
    ck(cudaMalloc(&ss.gdn_state, z.gdn));
    auto expected = pattern(z.gdn, salt);
    ck(cudaMemcpy(ss.gdn_state, expected.data(), expected.size(), cudaMemcpyHostToDevice));
    std::vector<int32_t> ids{1, 2, 3}; std::vector<ConversationImageKey> images;
    std::vector<ConversationCheckpoint> checks;
    SavedConversation saved;
    check(conversation_snapshot_save(saved, {ids, images, checks, true}, ss, g, nullptr, error), error.c_str());
    check(saved.live.gdn == expected, "captured snapshot fully valid on return");
    SavedConversation moved = std::move(saved); // eviction/replacement cannot race a synchronous capture
    saved = {};
    ck(cudaMemset(ss.gdn_state, 0, z.gdn)); ck(cudaDeviceSynchronize());
    check(conversation_snapshot_restore(moved, ss, g, nullptr, error) == ConversationRestore::restored, error.c_str());
    std::vector<uint8_t> restored(z.gdn);
    ck(copy_nonblocking(restored.data(), ss.gdn_state, restored.size(), false, true));
    check(restored == expected, "whole restore exact after ownership move");
    ck(cudaFree(ss.gdn_state));
}
#if defined(STRATA_READBACK_LINK_WRAP)
void capture_failure(Fault which, int nth, int dev) {
    ck(cudaSetDevice(dev));
    ModelGeometry g;
    g.n_layers = 3; g.ssm_state_size = 64; g.ssm_v_heads = 128; g.ssm_conv_channels = 8;
    ConversationStateSizes z; std::string error;
    check(conversation_state_sizes(g, z, error), "failure fixture sizes");
    check(z.gdn > PinnedReadback<>::piece_bytes, "failure fixture spans two chunks");
    SessionState ss; ss.max_cells = 96; ss.layer_hi = 3; ss.gdn_alloc = 3;
    ck(cudaMalloc(&ss.gdn_state, z.gdn));
    ck(cudaMemset(ss.gdn_state, 37, z.gdn)); ck(cudaDeviceSynchronize());
    SavedConversation previous;
    std::vector<int32_t> old_ids{99}, ids{1, 2, 3}; std::vector<ConversationImageKey> images;
    std::vector<ConversationCheckpoint> checkpoints;
    // Establish an actually valid prior image on another thread, leaving this thread's staging cold.
    std::thread baseline([&] {
        ck(cudaSetDevice(dev));
        std::string why;
        check(conversation_snapshot_save(previous, {old_ids, images, checkpoints, true}, ss, g, nullptr, why),
              "baseline capture before replacement");
    });
    baseline.join();
    check(conversation_snapshot_validate(previous, ss, g, nullptr, error), "previous image is valid");
    ck(cudaMemset(ss.gdn_state, 19, z.gdn)); ck(cudaDeviceSynchronize());
    calls = 0; fail_at = nth; fault = which;
    const bool saved = conversation_snapshot_save(previous, {ids, images, checkpoints, true}, ss, g, nullptr, error);
    fault = Fault::none;
    check(calls == nth, "requested failure was reached in production capture");
    check(!saved && !error.empty(), "failed capture rejected with error");
    check(previous.live.ids == old_ids && previous.live.gdn == std::vector<uint8_t>(z.gdn, 37) &&
          conversation_snapshot_validate(previous, ss, g, nullptr, error),
          "failed capture never replaces previously valid image, including after first chunk");
    // A failed restore reports transfer_failed even when a preceding chunk already changed the session.
    SavedConversation valid;
    check(conversation_snapshot_save(valid, {ids, images, checkpoints, true}, ss, g, nullptr, error), "capture after failure");
    ck(cudaMemset(ss.gdn_state, 0, z.gdn)); ck(cudaDeviceSynchronize());
    calls = 0; fail_at = 1; fault = Fault::sync;
    const auto restored = conversation_snapshot_restore(valid, ss, g, nullptr, error);
    fault = Fault::none;
    check(calls == 1, "requested restore synchronization failure was reached");
    check(restored == ConversationRestore::transfer_failed, "restore failure is never reported as restored");
    uint8_t changed = 0;
    ck(copy_nonblocking(&changed, ss.gdn_state, 1, false, true));
    check(changed == 19, "restore failure fixture actually modified live GPU state before failing");
    ck(cudaFree(ss.gdn_state));
}
#endif
}
int main() {
    deterministic();
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return 77;
#if defined(STRATA_READBACK_LINK_WRAP)
    for (auto f : {Fault::pin, Fault::submit, Fault::sync}) {
        // Fresh threads also start with cold pinned staging. nth=2 fails after a completed first chunk.
        for (int nth : {1, 2}) {
            if (f == Fault::pin && nth == 2) continue;
            std::thread worker([&] { capture_failure(f, nth, n > 1 ? 1 : 0); }); worker.join();
        }
    }
#endif
    // Two simultaneous captures, then a second device if present; each worker owns all payloads and staging.
    for (int round = 0; round < 2; ++round) {
        std::thread a([&] { capture(0, 11 + round); });
        std::thread b([&] { capture(n > 1 ? 1 : 0, 93 + round); });
        a.join(); b.join();
    }
    std::puts("conversation_readback_test: PASS (injected failures, chunking, concurrency, capture/restore)");
}
