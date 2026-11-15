# Report 0045 — Oberon M4 resident performance baseline

## Status and scope

**PASS for the bounded M4 measurement matrix**, not a numerical parity or production-readiness claim. Native resident A/B/B/A blocks completed at 32K for HIP Q8/F16, mixed Q8/F16, and Q8 topology, each for short and substantial prefill workloads. A Q8 near-capacity 256K topology A/B/B/A also completed. A separately balanced, same-token-ID client-adapter comparison includes the pinned vanilla reference. Each block used one warmup and three measured requests per workload; six measurements per arm. All reported blocks were eligible; no early-stop, failed or discarded benchmark block occurred. Synthetic repeated text is a controlled workload, not a representative application distribution.

## Provenance and safety

Starting branch `feature/heterogeneous-hip-only`, revision `b2840984af3d8d00e9fcfd1c263e62b493ccc848`. Native source and CMake inputs match 0044's binary provenance: separate HIP gfx1201 ROCm `/opt/rocm/core-10.1`, CUDA sm_89, and host Release binaries built with 28 jobs. SHA-256 coordinator `75fea7a7b41a6cddc5604706207a452c42e0d0dcf7bb79842af06f614418c563`, HIP `957e51822d090c4e2a790be35a7ff6b60c617c329b72434015b55e0c61df2f24`, CUDA `9f6824e339de2d177ad41c77d63188174f93376c17b76d1cf27317869a226cbe`. Checkout `.venv/bin/python` 3.12.3, `pip check` passed; analyzer tests 5/5 and selected resident/worker/admission CTests 6/6 passed. The first module-style analyzer-test invocation failed because its test imports from the tools directory; `PYTHONPATH=tools ... unittest discover -s tools` passed without installing dependencies.

Same Q2_0 pack/index SHA-256 `c344c01f59f547ed8a59003857a82eb90499e5299154241dbca787acb54b5b7c`, fingerprint `2026111301`, native resident KV, max_t=8, all 512 experts per owned layer and >=1024 MiB configured reserve. HIP placement `hip:0:0-24,hip:1:24-48`; mixed `cuda:0:0-15,hip:0:15-33,hip:1:33-48`; physical NVIDIA 04:00.0, AMD R9700s 03:00.0 and 07:00.0. Shared transport, grouped HIP, chunk 128, streaming, greedy, `HSA_ENABLE_SDMA=0`; no prompt/conversation cache reuse, MTP, lookup, vision, control vector or restart. Measured process environments cleared inherited `STRATA_*` trace/probe/override variables. Routine capacity/residency and request timing logs remained enabled; no window or transfer timing instrumentation was enabled. No GitHub Actions, downloads, repack, reserve reduction, fallback/spill, clock changes, forced GPU kills, or unrelated service changes.

Native comparisons used ordinary workload mode at 32768 capacity, not INT8-only long-context mode, so F16 was admitted. The short prompt had 7 tokens, the substantial prompt 4096, each with 64 actual output tokens and 63 committed decode feedback tokens. The 256K comparison used the qualified 262136-token prompt and four outputs/three feedback tokens at 262144 capacity; F16 was not attempted at 256K. Each topology/precision comparison ran in chronological A/B/B/A order, four separate process sets and four requests per set. Exact prompt IDs, equal work, source/settings, timings and output hashes were validated by `tools/heterogeneous_resident_bench_summary.py`. Independent GPU memory monitoring ran at two-second intervals.

## Native host timing results

Each cell is **median [minimum, maximum]**, six measured requests per arm. Prefill rate = committed prompt tokens / host prefill seconds; decode rate = committed feedback tokens / host decode seconds, excluding the last un-fed output. TTFT is first streamed output on the native host path, not HTTP TTFT. Elapsed is host request start to completion. Rates, including the seven-token prefill rate, are not kernel-only timings.

| Workload / comparison | Arm | Prefill tok/s | Feedback tok/s | TTFT ms | Elapsed ms |
|---|---|---:|---:|---:|---:|
| 7-token, HIP KV | Q8 | 58.47 [56.39, 60.19] | 44.40 [43.49, 45.40] | 119.72 [116.30, 124.14] | 1540.78 [1509.46, 1566.57] |
| | F16 | 60.40 [59.52, 60.66] | 45.34 [44.92, 45.75] | 115.90 [115.40, 117.61] | 1504.93 [1492.92, 1518.39] |
| 7-token, mixed KV | Q8 | 51.45 [50.96, 51.71] | 43.49 [43.14, 44.59] | 136.07 [135.36, 137.36] | 1585.90 [1549.44, 1595.93] |
| | F16 | 51.04 [50.95, 51.80] | 44.71 [44.39, 45.02] | 137.14 [135.14, 137.40] | 1546.20 [1536.75, 1554.55] |
| 7-token, Q8 topology | HIP | 60.18 [59.81, 60.67] | 45.56 [45.10, 45.83] | 116.32 [115.38, 117.05] | 1499.58 [1491.33, 1512.21] |
| | mixed | 51.68 [51.22, 52.16] | 44.64 [43.82, 45.36] | 135.45 [134.19, 136.66] | 1546.10 [1524.59, 1574.39] |
| 4096-token, HIP KV | Q8 | 1762.88 [1751.47, 1773.09] | 45.22 [44.69, 45.37] | 2323.48 [2310.09, 2338.61] | 3723.02 [3709.55, 3730.56] |
| | F16 | 1569.43 [1563.25, 1574.65] | 44.55 [44.46, 45.24] | 2609.86 [2601.21, 2620.18] | 4020.10 [4004.71, 4035.06] |
| 4096-token, mixed KV | Q8 | 2165.65 [2160.58, 2170.12] | 44.22 [43.70, 44.65] | 1891.35 [1887.46, 1895.79] | 3314.66 [3306.32, 3337.61] |
| | F16 | 1900.12 [1896.33, 1904.09] | 44.16 [42.66, 44.32] | 2155.66 [2151.16, 2159.96] | 3584.60 [3577.30, 3629.24] |
| 4096-token, Q8 topology | HIP | 1743.53 [1737.30, 1749.64] | 44.97 [44.74, 45.17] | 2349.26 [2341.06, 2357.68] | 3752.38 [3736.57, 3759.48] |
| | mixed | 2163.35 [2160.06, 2168.97] | 44.01 [43.89, 44.58] | 1893.36 [1888.45, 1896.24] | 3323.22 [3308.68, 3330.35] |
| 262136-token, Q8 topology | HIP | 1780.12 [1779.49, 1781.58] | 41.14 [41.03, 41.36] | 147257.73 [147136.60, 147309.95] | 147330.77 [147209.51, 147382.97] |
| | mixed | 2381.81 [2379.35, 2383.50] | 40.24 [39.83, 40.66] | 110057.25 [109979.23, 110171.04] | 110131.30 [110054.13, 110245.77] |

The 256K feedback figures contain only three feedback tokens per request and are not a robust long-decode benchmark. Native block startup was 8.86–11.02 s, shutdown 0.31–0.39 s, excluded from warmed timings. Output hashes were stable within each arm. The short comparisons had two hashes across arms; all 4096-token and 256K comparisons had one. Differences are diagnostic observations, not parity failures or permission to relax full-logit criteria.

## Three-way client-adapter observations

The pinned clean vanilla reference from report 0041 (`8db7ddc46cd2575514408c4ab7dd878f69df2f5f`, binary SHA-256 `686bbece567a7146935d0b4881577a46627edb7652e0be165e5ded3687eedce7`) ran on HIP 24/24, the same pack/actual prompt IDs and 32768 capacity, chunk 128, `--no-prefill-borrow`, `--short-read 0`, `--kv-resident 0`, 1024 MiB reserve, no cache, MTP, lookup, vision or control vector. Its `--serve` requires `--spec 2`; heterogeneous uses max_t=8. Vanilla reported 12288/12288 experts on each owner and zero experts streamed on every measured request. The adapter required `REUSED 0`, 64 actual outputs, length finish and clean QUIT. Q8/F16 vanilla blocks ran Q8/F16/F16/Q8; a separate Q8 sequence ran vanilla/HIP/mixed/mixed/HIP/vanilla. Each process served both workloads, one warmup and three measured requests each. No HTTP service was started.

The following are **client-side monotonic request-to-first-token and request-to-completion**, median [min,max] ms, six samples per arm. These are comparable end-to-end boundaries for the controlled token requests, but vanilla uses its raw native GEN protocol and heterogeneous uses the Python ResidentEngine adapter, so adapter/IPC overhead and engine settings differ. They are not kernel-only measurements or HTTP numbers and should not be divided into claims of exact engine speedup.

| Workload | Vanilla Q8 TTFT / elapsed | HIP Q8 TTFT / elapsed | Mixed Q8 TTFT / elapsed |
|---|---|---|---|
| 7 tokens | 80.0 [79.7, 80.1] / 996.1 [994.5, 996.9] | 119.7 [118.3, 121.6] / 1507.4 [1504.8, 1524.2] | 135.4 [134.6, 138.9] / 1549.2 [1533.8, 1561.5] |
| 4096 tokens | 3327.1 [3306.9, 3349.1] / 4263.7 [4241.2, 4285.6] | 2341.1 [2329.1, 2354.3] / 3739.8 [3728.8, 3766.3] | 1900.7 [1891.0, 1902.2] / 3323.7 [3303.8, 3345.0] |

Independent vanilla Q8 versus F16 blocks: short Q8 TTFT/elapsed 79.9 [79.7, 80.8] / 992.7 [990.8, 993.9] ms, F16 80.8 [80.3, 81.1] / 990.9 [985.5, 996.7] ms; 4096-token Q8 3331.6 [3319.5, 3435.4] / 4264.8 [4253.0, 4368.6] ms, F16 3411.1 [3391.1, 3473.9] / 4341.4 [4326.3, 4403.6] ms. Vanilla startup was 6.34–6.39 s. Across the three-way Q8 sequence, short vanilla/HIP outputs shared digest prefix `8191b66a103a`, mixed `5579ed6cbe65`; all three 4096-token outputs shared `8191b66a103a`. Prompt digest prefixes were `a55c9c3a513f` and `ab0b11dd8737`. Matching/differing greedy outputs are not a full-logit parity test.

## Memory and next step

Independent peak used MiB (two-second samples, plan order; maximum across the relevant four blocks): 32K HIP Q8 **21723/21885**, HIP F16 **21908/22071**; mixed Q8 **13615/17074/14776**, mixed F16 **13709/17230/14900**. Vanilla 32K Q8 **19637/20071**, F16 **19822/20257**. At 256K, HIP Q8 **23393/23555**, mixed Q8 **14507/18486/15928**. These monitors do not measure instantaneous peak and differ from runtime accounting.

For 32K representative graph-capture phase snapshots, HIP Q8 free was 11324/11164 MiB, F16 11138/10978; mixed Q8 2611/15974/18272, F16 2517/15818/18148. For 256K, the minimum native allocator-boundary free across measured blocks was HIP **9306/9146 MiB**, mixed **1437/14224/16774 MiB**; all logged admissions fit and the tightest CUDA phase was only 413 MiB above reserve. These are sampled phase observations, not guaranteed transient headroom. Vanilla reported 12998/12696 MiB free after loading in the Q8 client block. All worker sets drained; no workers survived.

The observed short decode cost dominates small requests, while 4096-token and 256K prefill benefit from the mixed placement; Q8 improves substantial-prefill rate here but not necessarily short decode. These measurements alone cannot isolate PCIe transport, host orchestration, graph execution or layer imbalance as the cause. **M5 next:** use a separate bounded trace/profile after the baseline to measure per-stage decode execution and synchronized CUDA↔HIP handoff/host wait, then evaluate a bounded asynchronous handoff or overlap only if ownership, cancellation, reserve and fail-closed semantics remain intact. Do not infer a throughput gain from instrumentation or enable unqualified async in this baseline.

Raw private commands, configs, block JSON, analyzer outputs, logs, monitoring, and the small client adapters are under `/home/chris/tmp/strata-heterogeneous/handoff-0045/`. The prior M1–M3 evidence remains in reports 0041/0044. The unrelated untracked `tools/heterogeneous_pool_probe.cpp` and dirty vanilla checkout were preserved. HTTP service production behavior, arbitrary prompts, broader 256K decode, transient VRAM peaks and strict numerical parity remain outside this result.
