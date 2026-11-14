# Report 0009 — starvation investigation (partial)

**Status: PARTIAL / UNVERIFIED.** Branch was fast-forwarded from `f6798aa88e482bec076e05cda9dfd53fcb95a21e` to the requested `e609d2901ce4abb96be25b68eb0d910d61eca237`. No inference code was changed or newly tested. This report re-analyzes existing report 0008 artifacts; it does not complete the requested GPU experiment matrix. The exact source tested by those historical runs was `54011bed3d345c27e43657b5af6ce1c081f0a3d2`. The report commit itself is identifiable in Git history; it contains no production code changes.

## Existing heterogeneous evidence

Raw staging: `/home/chris/tmp/strata-heterogeneous/m6-0008-XQLzkX2c/`, especially `replay-off.json`, `replay-off-utilization.json`, `replay-off-telemetry.jsonl` and `trace-A02.log`. Original full resident Q2_0 12/18/18, chunk 8, lookup off; original report contains device mapping, pack identity, safeguards and limitations. This is historical evidence, not a rerun at the handoff SHA.

New extraction from the original 72 replay requests: 52,482 prompt tokens / 330.370 seconds of `prefill_including_reset` = **158.86 prompt tokens/s pooled**, including reset and all requests, including warm-ups; it is not pure kernel throughput or an HTTP TTFT. Per-device sample-weighted prefill means (CUDA/HIP0/HIP1) were busy **12.73/27.00/50.10%**, power **57.37/84.89/129.64 W**, temperatures **44.34/51.88/50.51 °C**, with 1306/1318/1318 valid samples respectively at nominal 250 ms. Sample counts may include short-phase boundary effects; busy readings have vendor-specific averaging and are not occupancy or precise overlap. Original native first-prediction turn medians ranged 2.696–6.259 s. Clock data was not extracted.

Report 0008 warm decode: equal-task median **58.987 native feedback tok/s**, pooled 59.030; sampled busy **12.94/33.59/45.87%**, power **56.72/85.15/102.15 W**, limits 285/300/300 W. 598 aligned complete cycles, zero cycles all devices >=90%. All completions hit the 128-token cap; real agent quality, completion, streaming throughput and the 160 tok/s target remain **UNVERIFIED**.

## Comparison and critical-path interpretation

The operator's vanilla dual-R9700 observation (decode ~50/100% busy, prefill ~100/100%) is **unmatched**, not independently reproduced; workload, launch mode, exact asset identity, phase boundaries, sampling, MTP and memory placement are not established. Do not attribute the difference to a particular hardware link or kernel on this basis.

The source audit in `experimental/heterogeneous/docs/M6_GPU_BOTTLENECKS.md` identifies a serial blocking stage loop, one pending window and commit barrier, and prefill using <=8-row verifier windows with commit/checkpoint rather than vanilla bulk-prefill. Serial stage dispatch is a concrete architectural explanation for substantial cross-device starvation, but not a quantified causal comparison against vanilla. `CMakeLists.txt` declares optional HIP `STRATA_PREFILL_MMQ` default OFF; the actual historical cache value was not verified in this iteration. Native bulk-prefill, graph, transport and MTP comparison remains incomplete.

Historical separate A02 trace reports median stage roundtrips **2.675/5.873/7.630 ms** and commit **0.699 ms**; final execute **6.927 ms inclusive**. These are independent medians, not a timestamp-joined window breakdown or GPU kernel timings; no additive per-window attribution, head-vs-layer conclusion, transfer cost, dispatch gap or native-kernel cause is established. An illustrative sum of stage and commit medians is 16.877 ms, versus 6.25 ms per token needed for 160 tok/s; this is not a hardware lower bound. Removing only the illustrative commit median cannot close the gap. Safe independent slot/state ownership and restoration are prerequisites for pipeline overlap; MTP assets and lifecycle are not qualified. Original placement diagnostic remains FAIL as described in report 0008; do not relax parity.

## Not completed / follow-up

- **UNVERIFIED:** requested fresh original-plan replay, five warm balanced controls, chunk 1 versus auto, per-device clocks, output/position checks, full residency and cleanup. No fresh GPU inference was performed.
- **UNVERIFIED:** authorized runnable vanilla reference and matched no-draft/draft baseline; HIP+HIP worker isolation was not assessed.
- **UNVERIFIED:** successful timestamp-joined per-window decomposition, copy/host accounting, decode/prefill device activity and HIP1 layer versus head attribution. `nsys` 2022.4.2 and `/opt/rocm/bin/rocprofv3` 1.4.1 are installed; their presence is not evidence of a working capture. No profiler trace was collected.
- **UNVERIFIED:** full source review, evidence-backed optimization, fresh native builds, tests and real-path parity/commit/cancel/cleanup. No safe speedup is claimed. Rank subsequent work by measured critical-path recovery after obtaining joined traces and separate GPU timelines; first establish matched baselines, then evaluate native bulk prefill and coordinated pinned-ring/event ownership, followed by independently owned multi-window lifecycle and rollback. MTP requires approved matching assets and state ownership.
- Full representative agentic quality, 95–100% productive utilization, sustained useful power and 160 tok/s are **not demonstrated**. Historical 14/19/15 placement parity failure and remaining all-owner/restore hardening blockers persist.

No driver, clock, power, precision, context, memory budget or GPU settings were changed. Existing untracked `tools/heterogeneous_pool_probe.cpp` was left untouched and excluded from this report commit.
