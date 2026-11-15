# 0034 — 15/18/15 placement (partial qualification)

**Decision: promising for this one prefill workload, not qualified as the default.** At 32K configured capacity, the candidate completed a short native-prefill control and a matched A/B/A source-review experiment with full expert residency. Its warm median prompt rate was about 5.9% higher than either original-placement launch; decode was essentially unchanged. The candidate's lowest sampled CUDA free memory in the comparison was 2,229 MiB, not a guaranteed peak bound. All source-review outputs hit the completion limit and differed across placements, so quality equivalence is not established. Neither near-32K inference nor 256K Q8 was tested. Prior diagnostic failures and report 0032's quality limitations remain in force. No default change or production admission is recommended on this evidence.

## Status

| Gate | Result |
| --- | --- |
| Branch/source | PASS: fast-forwarded to `1adb08eeaef9e7ccacfd230797646bcdf9aaff41`; no implementation changes. |
| Build / calculator | PASS: existing host, CUDA and HIP builds with `-j28`; `PYTHONPATH=tools .venv/bin/python -m unittest -v test_heterogeneous_vram` passed; calculator JSON and sweep saved. |
| 2K candidate | PARTIAL: serial and native-bulk short arithmetic returned `4`, natural stop, clean close, full residency and zero reported host misses. No cancellation/reset proof. Serial sampler failed due to missing `PYTHONPATH`; no serial memory claim. Bulk telemetry worked. |
| 32K capacity candidate | PARTIAL: short native-bulk arithmetic control succeeded; configured capacity is not actual prompt length. Serial 32K control, near-capacity test, cancellation and comprehensive finiteness/state audit NOT_RUN. |
| Matched workload | PARTIAL: A/B/A independent launches, one fixed real-source review, one warmup plus three repeats per launch, native chunk 128. HTTP TTFT, naturally terminating long output, frozen replay, additional task quality, balanced B/A/B and telemetry-off/1 Hz controls NOT_RUN. |
| Q8 / 256K | NOT_RUN: this checkout does not provide explicit heterogeneous worker Q8 selection or a 256K workload/native-spec path. Do not infer runtime support from the calculator or change only `max_cells`; implementation and staged qualification are needed. |

## Identity, configuration and safety

The existing checkout `.venv` and vanilla build caches were used; no downloads, repack, spill, forced worker kills, GPU clock/power changes or GitHub Actions. Coordinator SHA-256 `6c455bde4334b9e647edd5eefff8ad006bbb169df29b22e4cd55383f7c92cda4`; CUDA worker `725d62b69d744856b6d855a6a8e81f3673c153265efbf39619672c0d5919f3d5`; HIP worker `76d25eaddb46af45619efbab893d160be38dd8eabbb29d5266af76d8f0f83fab`. All match report 0032's rebuild. Source-bound CLI no-run preflights passed. Original configs remained unchanged and their hashes match report 0032. Fingerprint `2026111301`; configured pack index SHA-256 `c344c01f59f547ed8a59003857a82eb90499e5299154241dbca787acb54b5b7c`; not a new raw asset audit.

Both comparison arms used 32768 cells, max_t 8, shared transport, grouped HIP, streamed commits, greedy chat, chunk 128, same model/shards/profile and PLE on CUDA0, full slots, no arena cap, and **1 GiB reserve on each owner** (higher than the original 512 MiB). Precision and optimization environment variables were unset for every arm; no trace requested. Defaults, including reported HIP Q8 HC projections, remained as implemented. No MTP. Runtime logs confirmed CUDA0 PCI `0000:04:00.0`, HIP0 `0000:03:00.0`, HIP1 `0000:07:00.0`. Candidate slots were 7680/9216/7680 for `[0,15)/[15,33)/[33,48)`; original 6144/9216/9216 for `[0,12)/[12,30)/[30,48)`.

32K config SHA-256 in stage order:

- Original: `52d12346562a956c69ea6c7fd4b7eb513de0a2924a6e742a568a863ae8cd9484`, `d1dc9451f0cf1e57857af70876ae4783ab066323d44d8ab67a00723eaf6adc5b`, `bebf4ee0f26b78d55a538a5187eba4e8688c41c633c28e32399dbd0900813a34`.
- Candidate: `f9566bf58818ccb2b187ca8698b90f12d2c4f64e7e0438fee8e88034959e457d`, `94a51536bd2e132e70b82820de226f9715d7b3a7f834cd0dd66a273371fef433`, `eb60718cd5e638c64a35ac82ce1323a2af2e8cc7c222cfd68dd24ed43e9ac035`.
- Candidate 2K: `412e458e4e1cfb7703d206bf131b9e73190004939dec119e02ec93c5174dbf17`, `3dde8fd4c33dcb3c390dcea0f9f318f9053e6201df9c0ab51954808cfc292926`, `76670b44271c8ba4ed207175e2f0a284573789097ec6a123762e2dae9098144c`.

## Memory and capacity evidence

Passive readiness inventory for 2K candidate reports packed selected-weight bytes 2,357,877,248 / 2,628,297,088 / 2,438,140,544 and canonical expert cache plus table 10,616,930,304 / 12,740,296,704 / 10,616,930,304 bytes, CUDA/HIP0/HIP1. These are inventory categories, not a complete device peak: session, native dense/PLE, verifier, prefill/staging, allocator and dynamic overhead are omitted. Startup guards were retained with 1 GiB reserve; they cannot establish a bound on later dynamic peaks.

| Candidate test | Sample interval | Max sampled used MiB CUDA / HIP0 / HIP1 | Min sampled free MiB CUDA / HIP0 / HIP1 |
| --- | --- | --- | --- |
| 2K short bulk | 250 ms | 13,891 / 16,822.7 / 14,524.5 | 2,485 / 15,801.3 / 18,099.5 |
| 32K short bulk | 250 ms | 14,097 / 17,154.7 / 14,792.5 | 2,279 / 15,469.3 / 17,831.5 |
| 32K review B | 250 ms | 14,147 / 17,203.4 / 14,875.4 | 2,229 / 15,420.6 / 17,748.6 |

Each short bulk run committed 25 native-prefill rows at chunk 128; each finished with zero host expert misses and `HETERO_CLOSED`. The 32K candidate's graph capture reported 2,519 MiB CUDA free in the short run. Sampling can miss transient peaks; the 1 GiB actual-free goal is supported only for observed sample instants, not proven for all peaks or other workloads. No refusal was bypassed.

The 256K Q8 calculator estimates 14.50 / 18.03 / 15.83 GiB used and 1.50 / 13.97 / 16.17 GiB free for CUDA/HIP0/HIP1. It assumes the original 32K sampled peaks, FP16 baseline KV, a 64 MiB per added dense layer allowance and extra workspace; these are not measured bounds. Source `src/core/layer.cpp` initializes `g_kv_int8` false and worker stage initialization does not select Q8, supporting the baseline FP16 assumption, but per-state runtime format was not directly logged. Q8 geometry yields 3,321,888,768 total KV bytes at 256K; it is an estimate, not an allocation. Existing heterogeneous workload limit is 32768. Explicit per-owner Q8 `kv_int8` and resident `kv_mode=0` must be verified before sizing/allocation in any future implementation; no host mirror, streaming, fallback, or cap-only workaround is qualified here.

## Matched A/B/A source-review results

Fixed prior `heterogeneous_contract.py` source-review prompt SHA-256 `41eae5dd2d090b0b49783084495394db6a08ffe6eeb61763eb093722927cfa6f`. Each independent resident CLI launch used 1,378 prompt tokens and a 512-token completion budget, four identical requests with the first excluded as warmup. All 12 completed with `finish=length`, 512 returned tokens, no host misses and clean close. Rates are medians of three warm repetitions from `heterogeneous_host_commit_progress` timing: prefill includes reset/IPC; decode is feedback checkpoints and consumer pacing. First output is CLI timing, **not HTTP client TTFT**. Startup is separate. This is one task and one order, not a statistical generalization.

| Launch | Placement | Startup s | Prefill tok/s | Decode feedback tok/s | First output ms | Result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| A1 | 12/18/18 | 7.60 | 675.5 | 44.13 | 2,040 | truncated 3/3 |
| B | 15/18/15 | 8.82 | 715.5 | 44.32 | 1,926 | truncated 3/3 |
| A2 | 12/18/18 | 7.51 | 675.6 | 43.95 | 2,040 | truncated 3/3 |

All four outputs within each launch were identical; the candidate output differs from both identical baseline-launch outputs. All answers reached the failure-case section, but the example was cut off; candidate names exceeding context as a failure before truncation. This is not a completed rubric pass and cannot establish acceptable quality or parity. Existing report 0032 identifies a separate incorrect Python answer. No generated code/tools were executed. Operator-reported vanilla ~4,000 prefill/~120 decode is an unmatched reference, not a controlled comparison.

### Phase-aligned 4 Hz telemetry

Mean reported busy % and power W for the three warm requests, per CUDA/HIP0/HIP1; 23–24 prefill and 139–140 decode samples per device per launch. Busy is a device-reported utilization metric, not occupancy; short prefill phases are sensitive to sample alignment. One sampler per launch; memory includes other allocations. Telemetry overhead controls were not run.

| Arm | Prefill busy % | Prefill power W | Decode busy % | Decode power W | Max sampled used MiB CUDA / HIP0 / HIP1 (whole launch) |
| --- | --- | --- | --- | --- | --- |
| A1 | 38.5 / 82.5 / 82.6 | 86.3 / 184.7 / 193.2 | 10.5 / 26.8 / 34.6 | 60.1 / 73.6 / 87.8 | 11,809 / 17,133.4 / 17,279.6 |
| B | 37.6 / 83.1 / 74.1 | 98.9 / 193.7 / 171.4 | 13.1 / 27.6 / 31.4 | 61.7 / 74.7 / 81.1 | 14,147 / 17,203.4 / 14,875.4 |
| A2 | 52.0 / 80.9 / 84.7 | 91.1 / 186.2 / 202.0 | 10.3 / 26.5 / 34.8 | 59.5 / 74.5 / 85.7 | 11,809 / 17,133.4 / 17,279.6 |

## Remaining work and evidence

Private evidence: `/home/chris/tmp/strata-heterogeneous/m7-0034-placement/`, including logs, configs, readiness, calculator output, raw outputs and samples. 2K serial first attempt correctly refused workload mode without `--resident`; retry succeeded. The short serial telemetry attempt failed import and is not used. The unrelated untracked `tools/heterogeneous_pool_probe.cpp` was preserved.

Before promotion: verify clean reset/cancel and serial 32K; run longer real prompts and naturally completed outputs, frozen replay and actual task tests, repeat balanced launch orders and telemetry controls, and establish safe peak headroom including prefill/transients. Compare quality explicitly rather than accepting differing answers. For the ultimate goal return to implementation of opt-in Q8 and extended context, then stage 2K/32K and 64K/128K/256K capacity qualification with actual per-owner state and reserve evidence; do not expand on an estimate. No 256K inference or VRAM-resident Q8 result is claimed.
