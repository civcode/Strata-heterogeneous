# 0036 — reference and common-path performance diagnosis

**Status: PARTIAL.** The new native-prefix timing passed integrated tests and matched actual HIP-only and mixed traces. Reversed independent heterogeneous runs and a separate controlled flag study identify a practical **existing opt-in**, not a new implementation: compact results plus fast fence and fanout raised warm CLI story decode from about 44–45 to 59 tok/s on grouped HIP and 63 tok/s on preferred mixed 15/18/15, with identical output IDs within each placement for the tested prompts. These observations do not qualify production, establish numerical parity, or justify a default change. The original Python control still fails on every architecture and mode; the cause remains unresolved. Matched authorized vanilla and near-32K testing remain NOT_RUN.

## Source, integration and safeguards

Tested branch `feature/heterogeneous-hip-only` at `ff97e6ad694a373c0509478bea33e574a8598109`. No inference implementation or default was changed during this iteration. Existing host/CUDA/HIP CMake caches and checkout `.venv` were used, with the same toolchain as the existing vanilla setup. Each `cmake --build ... -j28` succeeded; host CTest `-j28` 99/99; targeted critical-path, native-prefix, HIP pair, architecture-comparison and resident suites 38/38. Coordinator/CUDA worker/HIP worker SHA-256: `5681204b02058a9fc1f5bdab7812cf700a669d4fbaeec77c3b817e43d8850260`, `08359d79518d0a7057887b7c161efab7b5a4adbc2da49f65c466a2223a09937a`, `07cf106e6d3b98458b3dd33eacbc3dce1b86e479b39c34677751884dd840759b`. No Actions, downloads, repacks, reserve reductions, spill, forced GPU-worker kills, clock/power changes, or branch merge. The unrelated untracked probe was preserved.

Original Q2_0 model assets, profile and entry PLE identity from report 0035 were retained: pack index SHA-256 `c344c01f59f547ed8a59003857a82eb90499e5299154241dbca787acb54b5b7c`, fingerprint `2026111301`. This is not a new complete asset checksum. All owner configurations used full expert residency, max_t 8, no arena cap, 1 GiB reserve per owner, FP16 KV default, shared transport, grouped HIP, no MTP/lookup, greedy output and the existing exact model tokenizer/template. HIP ordinals and physical PCI bindings were confirmed in startup logs. Primary matrix and trace runs did not set optimization or tracing flags except where stated; RDNA4 default HC projections are Q8 as logged. No CUDA worker was configured in HIP-only arms. Existing mixed defaults were not promoted or altered.

2K config SHA-256 in stage order:

- HIP 24/24: `97c180088dfb5dba6f627b781952164248fa0fdb8709f7ba2990227706d89f97`, `e6de8f20ab00ea1e1755e31512f7a5c380d3f6a26d13a9e7e03e7039a765e59c`.
- Mixed 12/18/18: `d3886589ae39aa64d93bff5f1f743d2ce8f1c709eac8e5071355ca293b6eee2f`, `fad83d9269ff860f1acce17ac5c1042fc0d7b89ac9708765ee3266bd6c282190`, `761cfa62cb5792bcd7aa3e3927ef6fff983539d113f9edd8bbbde9f3d0239f80`.
- Preferred mixed 15/18/15: `412e458e4e1cfb7703d206bf131b9e73190004939dec119e02ec93c5174dbf17`, `3dde8fd4c33dcb3c390dcea0f9f318f9053e6201df9c0ab51954808cfc292926`, `76670b44271c8ba4ed207175e2f0a284573789097ec6a123762e2dae9098144c`.

## Timed native prefix and legacy coverage

With `STRATA_HETERO_PERF_TRACE=1`, the same 195-token source-review prompt and 32-token budget ran in grouped HIP bulk and both mixed bulk placements. Analyzer endpoints were physical worker endpoints `1` for grouped HIP and `0,2` for mixed; each analyzer accepted 32/32 ordinary windows with matching positions, benchmark phase bounds and complete native-prefix timing. The 194-row native-prefix round trips were 1.225 s HIP, 1.181 s mixed 12, 1.152 s mixed 15, respectively, in separate cold single observations. These are host round trips, not GPU kernel timings. The ordinary window followed the successful prefix ACK at window 2. A pre-0036 untimed log still gives position counts matching but analyzer exit 1 and `matched=false`, correctly refusing full timing coverage. Traces are intrusive diagnostics, not throughput runs.

## Quality diagnosis and reference blocker

The unchanged frozen coding case asks for `print(sum(i*i for i in range(4)))`, with exact expected answer `14`. The trusted fixture evaluates 0 + 1 + 4 + 9 = 14 using the standard library; no generated code was executed. Grouped HIP serial and bulk and mixed 12/15 bulk, in fresh resident CLI processes and repeated resets, each returned `30`, stop, output IDs `[18, 15]`; all used the same 38 CLI input IDs. Independent reconstruction through the actual Service OpenAI message conversion and tokenizer gives the same 38 IDs as CLI (SHA-256 of JSON IDs `8c44409e3c2a262ee230b85636fc95d0484858c1bc2589e73116c0a8fa07f964`). The matching HTTP matrix returned `30` in all arms and repetitions. This narrows a simple HTTP-versus-CLI rendering discrepancy and one reset-specific explanation; it does **not** distinguish a model limitation from a numerical divergence shared by these paths. Existing reports 0027.15/0030 retain their numerical failures. No threshold was weakened.

**Matched vanilla: NOT_RUN.** An authorized, runnable existing dual-R9700 vanilla executable and its exact invocation/model identity were not established; no unmatched rate is inserted in the table. To unblock, provide/confirm the authorized executable and config paths and permission to use them, plus the exact model/shard and pack/profile identities, tokenizer/template, FP16 KV/context, GPU mapping, greedy sampling, chunk/prefill mode, MTP/cache settings and timing denominator. Do not infer identity from an unrelated worktree or the historical ~4000/~120 operator observations. The same missing oracle prevents a definitive model-vs-native-numerical classification and meaningful matched vanilla MTP. General HIP-only MTP remains unsupported; separate speculative comparison NOT_RUN.

## Balanced trace-off HTTP comparison

Frozen manifest SHA-256 `8e4f64ee29ff21b6d2478856d6c96f7507b9b985bc7087dedf5921d52e38df76`: exact arithmetic and Python controls, a natural story, and 1,378-token real-source review. Two independent launch blocks reversed the order: HIP serial, HIP bulk, mixed12 bulk, mixed15 bulk; then mixed15, mixed12, HIP bulk, HIP serial. One warmup plus five measured repetitions per case in each launch, one server at a time; 4 Hz passive telemetry. Prefill chunk 1 for HIP serial, 128 for bulk, 2K capacity, 512 completion budget. Every launch had full residency, zero reported host misses and clean close. The service's prompt/decode rates are `heterogeneous_host_commit_progress` proxies, not GPU kernel rates; HTTP TTFT and total are client wall time. Actual prompt counts were 27 / 38 / 43 / 1,378. Service prompt reconstruction was checked for these inputs; individual HTTP request token IDs were not directly logged.

All 10 measured arithmetic answers per arm passed; all 10 coding answers per arm failed as above, so benchmark exit status was nonzero for the known quality failure. Story was UNSCORED but naturally stopped; review was UNSCORED and truncated at 512 tokens in all arms. Within each arm/case, outputs repeated identically across both launches, but story and review outputs differed among arms. Story completion counts differ, so total-time or decode comparisons across arms are **not paired same-output work**. Do not call this quality parity or use its E2E ratios as a speedup claim.

| Arm, block | Story tokens / finish | Story TTFT / total s | Story decode proxy tok/s | Long review TTFT / total s | Review prefill / decode proxy tok/s |
| --- | --- | --- | ---: | --- | --- |
| HIP serial 1 | 189 / stop | 0.979 / 5.324 | 43.6 | 31.138 / 42.654 | 44.3 / 44.3 |
| HIP serial 2 | 189 / stop | 0.976 / 5.249 | 44.4 | 31.384 / 42.988 | 43.9 / 44.2 |
| HIP bulk 1 | 181 / stop | 0.323 / 4.408 | 44.4 | 2.591 / 14.300 | 532.2 / 43.7 |
| HIP bulk 2 | 181 / stop | 0.324 / 4.431 | 44.1 | 2.596 / 14.166 | 531.2 / 44.2 |
| Mixed 12 bulk 1 | 168 / stop | 0.314 / 4.125 | 44.1 | 2.047 / 13.924 | 673.6 / 43.1 |
| Mixed 12 bulk 2 | 168 / stop | 0.314 / 4.156 | 43.8 | 2.051 / 13.775 | 672.2 / 43.7 |
| Mixed 15 bulk 1 | 187 / stop | 0.312 / 4.556 | 44.1 | 1.933 / 13.830 | 713.5 / 43.0 |
| Mixed 15 bulk 2 | 187 / stop | 0.313 / 4.602 | 43.6 | 1.935 / 13.633 | 712.6 / 43.8 |

Values are each block's median of five warm repetitions. Long-review completion counts/finish match but generated content differs; HIP serial/bulk changes algorithm, mixed versus HIP changes hardware and placement. On this specific prompt, preferred mixed prefill proxy was about 6% above mixed12 in both blocks, with no substantive decode change. No matched vanilla arm, full agent task, or general quality claim.

Whole-launch maximum sampled used MiB CUDA/HIP0/HIP1, both blocks: HIP serial 447/21,145/21,307; HIP bulk 447/21,484/21,648; mixed12 11,603/16,863/16,946; mixed15 13,939/16,869/14,605. CUDA's minimum sampled free for mixed15 was 2,437 MiB. Samples can miss transient peaks and include unrelated background usage. Sampler was stopped with normal termination after each owned server closed.

## Existing opt-ins and measured common path

Historical report 0007/0008 used the existing compact-result, fast-fence and fanout opt-ins, whereas report 0035 and the primary matrix left them unset. We tested one change at a time, then their combination, with trace and telemetry off. Each entry is an independent launch, one warmup plus five warm story repetitions; base1 and base2 bracket the interventions. All completions naturally stopped, with identical output IDs across all flag variants within a placement (180 CLI output IDs HIP, 186 mixed15); these CLI token counts omit the stop token and differ from HTTP usage. No cross-placement output equality is asserted.

| Placement | Baseline 1 | Compact only | Fast fence only | Fanout only | Combined | Baseline 2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HIP 24/24 | 44.44 | 52.65 | 49.45 | 45.02 | 59.21 | 44.82 |
| Mixed 15/18/15 | 44.58 | 51.16 | 52.55 | 44.64 | 62.64 | 44.61 |

Numbers are median CLI host-commit decode feedback tok/s on the same naturally terminating story. Combined is `STRATA_HETERO_COMPACT_RESULT=1 STRATA_HETERO_FAST_FENCE=1 STRATA_HETERO_FANOUT_CONTROL=1`; no other optimization environment overrides. The native nonfinite guard was left enabled; worker source explicitly rejects compact result when it is disabled. Every flag run closed cleanly and reported full residency/zero misses. A separate two-repeat resident sequence of 1,378-token review plus coding control compared baseline to combined on each placement: every corresponding input/output ID sequence matched; review still truncated at 512 and coding still returned the wrong answer. This is bounded repeatability evidence, not broad numerical qualification or a new implementation. A separate code change would duplicate already available controls; no safe new default or optimization is promoted. The ~60 versus ~44 difference is consistent with these explicitly controlled settings on this workload, not proof that all historical settings or remaining bottlenecks are identical.

Sampler controls on the same default story used separate launches: off baseline brackets HIP 44.44/44.82, mixed15 44.58/44.61; 1 Hz HIP 45.19, mixed15 44.67; 4 Hz HIP 44.89, mixed15 44.98. Not balanced and not sufficient to resolve sub-percent overhead. Combined at 4 Hz yielded 59.35 HIP and 62.73 mixed15; minimum sampled free MiB CUDA/HIP0/HIP1 was 15,929/11,166/11,002 and 2,711/15,781/18,045 respectively. These samples cannot guarantee transient free VRAM.

Separate trace-on comparison, same 195-token prompt and 31 decode windows, had full matched native-prefix coverage. Median decode critical/commit wall spans in ms were HIP default 22.03/2.13 versus combined 16.74/0.59; mixed15 default 21.94/2.06 versus combined 15.87/0.46. Worker covered spans and coordinator/commit spans may nest or overlap; do not sum medians or infer GPU kernel times. This points to a shared checkpoint/result/fence path worth deeper investigation, not proof of one isolated cause.

ROCm `rocprofv3` 1.4.1 captured kernel and HIP API traces in a separate short grouped-HIP run; profiling perturbs timing. Within its single 24.73 ms host decode interval, 742/734 dispatched GPU kernels overlapped the interval on the two agents; per-agent unions were 5.67/7.05 ms. The agents may overlap and the interval includes host/control/queue gaps; these are not occupancy, a full timeline attribution, or a representative sustained-decode measurement. Top inclusive recorded kernel families included Q3K matvec, Q8 split GEMV and BF16 GEMV. Raw profiler CSVs remain private. Longer phase-aligned GPU profiling and matched vanilla traces remain needed.

## Staged capacity observation

Preferred mixed 15/18/15, same 1 GiB reserve, full residency and bulk chunk 128, passed CPU model/readiness and no-run preflights and two separate real replay controls:

| Configured capacity | Actual prompt | Prefill proxy tok/s | Outcome | Minimum sampled free MiB CUDA / HIP0 / HIP1 |
| --- | ---: | ---: | --- | --- |
| 8,192 | 2,163 | 557.1 | tool call, UNSCORED, 48 completion tokens | 2,387 / 15,701 / 17,979 |
| 32,768 | 20,748 | 667.8 | tool call, UNSCORED, 69 completion tokens | 2,221 / 15,435 / 17,763 |

Both committed native prefix, reported full residency and zero host misses, then closed cleanly. Replays are scripted history, not autonomous tools; generated calls were not executed. 8K replay manifest SHA-256 `b5e178a7b90eed94d191d3100be2ddbee408d4a0a42efcd2aeb42a65a9190fe9`; 32K `8b54ddd122f4e88f4bf78f3a470d8fbd8256a757d24dddb8fcf809b0c786c938`. One observation each, 250-ms sampling, no near-32K prompt and no guarantee of actual peak. Q8 KV/256K is outside this runtime and remains NOT_RUN.

## Remaining work and evidence

Resolve authorization and run matched vanilla with exact prompt IDs/settings; compare the Python logits/state against the bounded oracle without weakening numerical thresholds. Qualify longer naturally terminating scored tasks, near-capacity 32K, bulk cancellation/rollback and opt-in flags over broader workloads and failure paths. Repeat balanced flag/monitor trials and use longer GPU/queue timelines before proposing new common-path code. Preserve full residency, finite checks, reserves and defaults. All private configs, logs, raw output, analyzer JSON, sampling and profiler data are under `/home/chris/tmp/strata-heterogeneous/m7-0036/`.
