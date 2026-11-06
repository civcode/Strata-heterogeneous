// src/kernels/qsa_topk_parity.cpp - the QSA top-k the engine dispatches (qsa_block_topk) against the original kernel
// (qsa_block_topk_ref): the selected cells must be identical (GPU, synthetic, no model).
//
// A case is a context under a capacity (the engine's --max-context; max_blocks = capacity / 4 + 2, and without an
// active-block count the dispatch follows it), with consecutive queries at the context's end so that the tail block
// holds 0 to 3 cells.  Three score sets per case: continuous, 64 levels (many ties at the threshold) and all equal
// (the rule's ties go to the lowest index).  The dispatch is called as a decode window calls it (no block count) and as
// the prompt path does (with the call's largest block count).  The reference and the dispatched kernel are also timed.
//
// Usage: qsa_topk_parity --selftest
//        qsa_topk_parity CONTEXT [QUERIES=256] [REPS=10] [CAPACITY=CONTEXT] [COUNT=1]   (COUNT=0: no block count)
// STRATA_TOPK_PAGEABLE_READBACK=1: diagnostic reproduction of ROCm pageable D2H corruption under contention.
// Normal readbacks use pinned staging; kernel dispatch and all parity requirements are unchanged.
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: error=%d (%s)\n", w, (int) e, cudaGetErrorString(e));
        std::exit(2);
    }
}

bool run_case(int64_t ctx, int64_t nq, int reps, int64_t capacity, bool counted) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t max_blocks = capacity / s.idx_block + 2, cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    std::vector<int32_t> steps((size_t) (nq * k::kStepCount));
    for (int64_t i = 0; i < nq; ++i) {
        int32_t* st = steps.data() + i * k::kStepCount;
        const int64_t pos = ctx - nq + i;
        st[k::kStepPos] = (int32_t) pos;
        st[k::kStepNKv] = (int32_t) (pos + 1);
        st[k::kStepNBid] = (int32_t) ((pos + 1) / s.idx_block);
        st[k::kStepWidth] = (int32_t) k::qsa_selection_width(pos + 1, s);
    }
    std::mt19937 rng((uint32_t) (ctx * 31 + capacity));
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> scores((size_t) (nq * max_blocks)), tied(scores.size()), equal(scores.size(), 0.25f);
    for (int64_t i = 0; i < nq; ++i) {
        const int32_t* st = steps.data() + i * k::kStepCount;
        float* row = scores.data() + i * max_blocks;
        for (int64_t b = 0; b < max_blocks; ++b) row[b] = nd(rng);
        // the tail block as the scorers leave it: far above the others when it holds cells
        if (st[k::kStepNKv] % s.idx_block != 0) row[st[k::kStepNBid]] += 1e9f;
        for (int64_t b = 0; b < max_blocks; ++b)
            tied[(size_t) (i * max_blocks + b)] = row[b] > 1e8f ? row[b] : std::floor(row[b] * 8.0f) / 8.0f;   // ~64 levels
    }
    int32_t *d_steps = nullptr, *ids_ref = nullptr, *ids_new = nullptr;
    float* d_scores = nullptr;
    ck(cudaMalloc(&d_steps, steps.size() * 4), "malloc");
    ck(cudaMalloc(&d_scores, scores.size() * 4), "malloc");
    ck(cudaMalloc(&ids_ref, (size_t) (nq * cap) * 4), "malloc");
    ck(cudaMalloc(&ids_new, (size_t) (nq * cap) * 4), "malloc");
    ck(cudaMemcpy(d_steps, steps.data(), steps.size() * 4, cudaMemcpyHostToDevice), "upload");
    const int64_t active = counted ? steps[(size_t) ((nq - 1) * k::kStepCount + k::kStepNBid)] + 1 : -1;
    std::vector<int32_t> a((size_t) (nq * cap)), b(a.size());
    int32_t* pinned_readback = nullptr;
    // A blocking pageable D2H read returned stale slices on gfx1201 under multiprocess contention, even after
    // device synchronization. Immediate rereads of the untouched device results were correct. Keep kernel parity
    // independent of that runtime path, using one persistent pinned buffer for both outputs (CUDA and HIP).
    ck(cudaMallocHost(&pinned_readback, a.size() * sizeof(int32_t)), "pinned readback allocation");
    const bool pageable_readback = std::getenv("STRATA_TOPK_PAGEABLE_READBACK") != nullptr;
    auto download = [&](std::vector<int32_t>& dst, const int32_t* src, bool pageable) {
        ck(cudaMemcpy(pageable ? dst.data() : pinned_readback, src, dst.size() * sizeof(int32_t),
                      cudaMemcpyDeviceToHost), "readback");
        if (!pageable) std::copy(pinned_readback, pinned_readback + dst.size(), dst.begin());
    };
    int diagnostic_rows = 0;
    auto same = [&](const std::vector<float>& sc, const char* distribution) -> int64_t {
        ck(cudaMemcpy(d_scores, sc.data(), sc.size() * 4, cudaMemcpyHostToDevice), "upload");
        ck(cudaMemset(ids_ref, 0xff, a.size() * 4), "memset");
        ck(cudaMemset(ids_new, 0xff, a.size() * 4), "memset");
        k::qsa_block_topk_ref(d_scores, d_steps, nq, max_blocks, cap, s, ids_ref, nullptr);
        ck(cudaGetLastError(), "reference launch");
        k::qsa_block_topk(d_scores, d_steps, nq, max_blocks, cap, s, ids_new, nullptr, active);
        ck(cudaGetLastError(), "dispatched launch");
        ck(cudaDeviceSynchronize(), "reference/dispatched synchronization");
        download(a, ids_ref, pageable_readback);
        download(b, ids_new, pageable_readback);
        int64_t n = 0;
        for (int64_t i = 0; i < nq; ++i) {
            const int64_t w = steps[(size_t) (i * k::kStepCount + k::kStepWidth)];
            const bool matches = std::equal(a.begin() + i * cap, a.begin() + i * cap + w, b.begin() + i * cap);
            // All-equal scores have an independent exact answer: the lowest valid cell IDs. Checking both
            // outputs prevents a shared reference/dispatch error from falsely passing the tie-breaking fixture.
            const bool equal = std::strcmp(distribution, "equal") == 0;
            bool identity = true;
            if (equal) for (int64_t j = 0; j < w; ++j)
                identity = identity && a[i * cap + j] == j && b[i * cap + j] == j;
            n += matches && identity;
            if ((!matches || !identity) && diagnostic_rows++ < 6) {
                int64_t first = -1, differences = 0;
                for (int64_t j = 0; j < w; ++j) if (a[i * cap + j] != b[i * cap + j] ||
                    (equal && (a[i * cap + j] != j || b[i * cap + j] != j))) {
                    if (first < 0) first = j;
                    ++differences;
                }
                const int32_t* st = steps.data() + i * k::kStepCount;
                // Independent oracle: rank blocks by score (ties by index), take cells, then emit ascending.
                std::vector<int32_t> blocks(st[k::kStepNBid] + 1), cpu;
                for (size_t j = 0; j < blocks.size(); ++j) blocks[j] = (int32_t) j;
                std::sort(blocks.begin(), blocks.end(), [&](int32_t x, int32_t y) {
                    const float sx = sc[i * max_blocks + x], sy = sc[i * max_blocks + y];
                    return sx != sy ? sx > sy : x < y;
                });
                for (int32_t block : blocks) {
                    for (int c = 0; c < s.idx_block && block * s.idx_block + c < st[k::kStepNKv] &&
                         (int64_t) cpu.size() < w; ++c) cpu.push_back(block * s.idx_block + c);
                    if ((int64_t) cpu.size() == w) break;
                }
                std::sort(cpu.begin(), cpu.end());
                int ref_bad = 0, new_bad = 0;
                for (int64_t j = 0; j < w; ++j) {
                    ref_bad += a[i * cap + j] != cpu[j];
                    new_bad += b[i * cap + j] != cpu[j];
                }
                std::vector<int32_t> reread(a.size());
                download(reread, ids_ref, false);
                int reread_bad = 0;
                for (int64_t j = 0; j < w; ++j) {
                    reread_bad += reread[i * cap + j] != cpu[j];
                }
                download(reread, ids_new, false);
                int new_reread_bad = 0;
                for (int64_t j = 0; j < w; ++j) new_reread_bad += reread[i * cap + j] != cpu[j];
                // Rereads are diagnostic only: a mismatch always fails, even if the retry is correct.
                std::fprintf(stderr, "CPU_ORACLE row=%lld reference_errors=%d dispatched_errors=%d "
                             "pinned_reread_reference_errors=%d pinned_reread_dispatched_errors=%d pageable=%d\n",
                             (long long) i, ref_bad, new_bad, reread_bad, new_reread_bad, pageable_readback);
                std::fprintf(stderr, "MISMATCH distribution=%s ctx=%lld capacity=%lld nq=%lld counted=%d active=%lld "
                             "max_blocks=%lld cap=%lld row=%lld first=%lld expected_cpu=%d actual=%d reference=%d differences=%lld "
                             "pos=%d n_kv=%d n_bid=%d width=%d reg_min=%s stream=%s old=%s\n",
                             distribution, (long long) ctx, (long long) capacity, (long long) nq, counted,
                             (long long) active, (long long) max_blocks, (long long) cap, (long long) i,
                             (long long) first, cpu[first], b[i * cap + first], a[i * cap + first], (long long) differences,
                             st[k::kStepPos], st[k::kStepNKv], st[k::kStepNBid], st[k::kStepWidth],
                             std::getenv("STRATA_TOPK_REG_MIN_BLOCKS") ? std::getenv("STRATA_TOPK_REG_MIN_BLOCKS") : "default",
                             std::getenv("STRATA_TOPK_STREAM") ? std::getenv("STRATA_TOPK_STREAM") : "default",
                             std::getenv("STRATA_TOPK_OLD") ? std::getenv("STRATA_TOPK_OLD") : "unset");
            }
        }
        return n;
    };
    const int64_t same_tied = same(tied, "tied"), same_equal = same(equal, "equal"), same_cont = same(scores, "continuous");   // d_scores: continuous
    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "event");
    ck(cudaEventCreate(&e1), "event");
    auto timed = [&](auto f) {
        ck(cudaEventRecord(e0), "record");
        for (int r = 0; r < reps; ++r) f();
        ck(cudaEventRecord(e1), "record");
        ck(cudaEventSynchronize(e1), "time");
        float ms = 0;
        ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
        return ms / (float) reps;
    };
    const float t_ref = timed([&] { k::qsa_block_topk_ref(d_scores, d_steps, nq, max_blocks, cap, s, ids_ref, nullptr); });
    const float t_new = timed([&] { k::qsa_block_topk(d_scores, d_steps, nq, max_blocks, cap, s, ids_new, nullptr, active); });
    ck(cudaFreeHost(pinned_readback), "free pinned readback");
    ck(cudaEventDestroy(e0), "destroy start event"); ck(cudaEventDestroy(e1), "destroy end event");
    ck(cudaFree(d_steps), "free steps"); ck(cudaFree(d_scores), "free scores");
    ck(cudaFree(ids_ref), "free reference"); ck(cudaFree(ids_new), "free dispatched");
    const bool ok = same_cont == nq && same_tied == nq && same_equal == nq;
    std::printf("%s context %lld under a capacity of %lld (%lld blocks), %lld queries %s: reference %.3f ms, "
                "dispatched %.3f ms (%.1fx); identical ids: continuous %lld/%lld, 64 levels %lld/%lld, equal scores "
                "%lld/%lld\n", ok ? "PASS" : "FAIL", (long long) ctx, (long long) capacity, (long long) max_blocks,
                (long long) nq, counted ? "with their block count" : "without a block count", t_ref, t_new, t_ref / t_new,
                (long long) same_cont, (long long) nq, (long long) same_tied,
                (long long) nq, (long long) same_equal, (long long) nq);
    return ok;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--selftest") == 0) {
        // {context, capacity}: both sides of the register kernel's reach (33,792 blocks = 135,168 cells), short
        // contexts under a long capacity (fewer blocks than the kernel's threads; a prompt batch below the dispatch's
        // 4,608 blocks), and the long contexts themselves
        const int64_t cases[][2] = {{9000, 9000},    {135160, 135160}, {135176, 135176}, {2100, 262144},
                                    {9000, 262144},  {32768, 262144},  {262144, 262144}, {300001, 524288},
                                    {524288, 524288}};
        bool ok = true;
        for (const auto& c : cases) {
            ok = run_case(c[0], 8, 3, c[1], false) && ok;     // a decode window's call
            ok = run_case(c[0], 64, 3, c[1], true) && ok;     // the prompt path's
        }
        std::printf("%s\n", ok ? "qsa_topk_parity: PASS" : "qsa_topk_parity: FAIL");
        return ok ? 0 : 1;
    }
    if (argc < 2) {
        std::fprintf(stderr, "usage: qsa_topk_parity --selftest | CONTEXT [QUERIES=256] [REPS=10] [CAPACITY=CONTEXT] "
                             "[COUNT=1]\n");
        return 2;
    }
    const int64_t ctx = std::atoll(argv[1]);
    const int64_t nq = argc > 2 ? std::atoll(argv[2]) : 256;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 10;
    const int64_t capacity = argc > 4 ? std::atoll(argv[4]) : ctx;
    if (ctx < nq || nq <= 0 || reps <= 0 || capacity < ctx) {
        std::fprintf(stderr, "qsa_topk_parity: CONTEXT >= QUERIES > 0, REPS > 0, CAPACITY >= CONTEXT\n");
        return 2;
    }
    return run_case(ctx, nq, reps, capacity, argc <= 5 || std::atoi(argv[5]) != 0) ? 0 : 1;
}
