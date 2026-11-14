# M5 — Performance optimization, target 160 tok/s on Oberon

Baseline: M4.7 bounded full Q2_0 routed-expert residency PASS at 6e295857; report 0004. 24,576 verified slots on CUDA and two HIP workers, zero host expert fallback. Matched 42-prompt-token / 64-completion-token warm feedback decode 36.82/36.85 tok/s, hybrid 21.31/21.37. Operator reports vanilla Strata on 2x R9700 ~120 tok/s using the same model. Exact vanilla launch options, quantized artifacts, draft mode, sampling and measurement denominator remain to be matched; treat 120 as operator-reported, not an independently reproduced result. Target **160 sustained accepted single-request decode tok/s**, with 120 intermediate. Do not count draft proposals, GPU rows that were rejected, or combined multi-request throughput. 160 is a target, not a prediction.

| Milestone | M5 source change | Remaining qualification |
| --- | --- | --- |
| M5.1 | Opt-in per-stage coordinator round trips, commit/reset barriers, native upload/execute/download/commit phases, final logits download/result send wall clock. | Isolate latency; profile separate runs from throughput samples. |
| M5.2 | Opt-in compact STFR result (predictions + nonfinite rows); no full vocabulary D2H/serialization/IPC/host scan. Final native GPU finite guard must be enabled, and coordinator rejects unexpected full-logit reply. | Compare compact vs full logits at same residency, prompt/chunk/sampling; verify guard and fail-closed refusal; retain diagnostic full-logit mode. |
| M5.3 | Opt-in 50 us polling vs existing 1 ms worker fence wait. All fences, deadlines, GPU status checks remain; no unsafe synchronization removal. | Measure CPU and wall times; further fence coalescing only after dependency audit. |
| M5.4 | Opt-in dispatch commit/reset to every independent stage before collecting ACKs. Validate every matching ACK before publishing or starting next request. Failure poisons all peers; no rollback promise. | CPU socket/shared barrier tests, failure cases and real GPU parity/cancellation. |
| M5.5 | Reuse original frame validation without throwing away serialized bytes; generation discards redundant provisional result copy. No change to actual wire checksum, shared-slot ownership or float32 handoff. | Compare socket vs shared; measure staged copies. Actual direct/overlapped cross-device transfers deferred. |
| M5.6 | Optional existing Strata CPU SuffixDrafter linked into coordinator, verified positive greedy prefix, bounded by prefill chunk <=8 and completion budget. Buffered generation/resident only, readiness negotiated LOOKUP, no tentative drafts published. | CPU and native partial-prefix tests, no-match fallback, varied workloads, accepted/proposed counts and correctness. Existing GPU MTP drafter still requires validated assets and final-stage residual/head/session integration; not implemented. |
| M5.7 | Existing resident repeat benchmark, per-stage traces, native lookup counters and passive telemetry. | Matched controls, documented numerical parity and speed; target not yet achieved. |

## Controls

Set environment per launch (unset or 0 is baseline, 1 opts in):
- STRATA_HETERO_COMPACT_RESULT=1 (generation only; requires enabled STRATA_NAN_GUARD, which must not be disabled; do not set for parity/tap/logit capture).
- STRATA_HETERO_FAST_FENCE=1 (50 us poll sleep; not a hard scheduling latency guarantee).
- STRATA_HETERO_FANOUT_CONTROL=1 (parallel GPU-local positive prefix commits and request resets, socket sends are serialized but GPU work can overlap).
- STRATA_HETERO_PERF_TRACE=1 (stderr lines HETERO_PERF component=... phase=... stage=... request=... window=... rows=... us=...; spans measure inclusive wall time and can log failed operations. Coordinator barrier stage=N is an aggregate sentinel, not a device; fanout ACK waits are order-dependent, not commit kernel timings. Use successful committed windows, separate traced and untraced runs, don't sum overlapping or nested spans.)

For lookup add explicit `--lookup-draft --prompt-chunk 0` to the existing heterogeneous-generate command, or generate server JSON with lookup_draft true. Requires effective chunk >=2 and buffered mode; `--stream-tokens` with lookup is rejected before workers start. No MTP weights, synthetic inference, approximate acceptance or alternate tokenizer. It indexes committed prompt/feedback history per request, proposes continuations after already-known next token, compares row i's actual model prediction against row i+1's candidate, commits only matching positive prefix and uses prediction at last accepted row. Stop proposals are truncated; final capped output is not fed. Failure and cancellation preserve existing fail-closed semantics. The ordinary model verification window still decides output, but multirow vs one-row native numerical behavior must be qualified; do not claim bitwise parity from CPU tests. HETERO_LOOKUP logs proposed/accepted draft counts and multirow windows on normal finish or decode cancellation; no-match uses ordinary one-row path. A short factual prompt may receive no useful proposal; test repeat/copy/code plus nonrepetitive chat.

For a full optimized nonlookup control set the first three flags to 1 and leave trace off. Use the same full-residency stage configs/plan/model/precision/reserve/context as report 0004. Start with no flags, then each flag independently and combined; compare outputs and position/ACKs, record actual startup, warm decode and prefill. Test shared vs socket and both chunk 1 and 8. Then qualify lookup separately with matched buffered controls, repeated prompt copy/code and ordinary chat, >=128 completion tokens when model naturally continues, report actual lengths and optional 256 cap. Do not force EOS suppression or omit quality checks. Warm runs >=3; actual feedback/accepted and completion counts, duration, median/range, prefill and startup separately. Profile kernel and handoff costs only with appropriate existing nonintrusive diagnostics; do not remove checksum/finite guard or device/process isolation.

## Oberon qualification — partial (report 0005)

[Report 0005](../../../.ai/reports/report-0005.md) tested source `2d8babd4ce4f481d2f9e1496b5a2a748b945002d`. 39/39 CTests, 413 Python tests and independent native CUDA/HIP builds passed. The agent made no production changes. Full routed-expert residency and zero host fallback remained verified. All 40 matched 42-prompt/128-output requests had identical token IDs across transports and flags.

| Warm median feedback tok/s | Socket | Shared |
| --- | ---: | ---: |
| Baseline | 36.520 | 37.049 |
| Compact only | 40.649 | 41.360 |
| Fast fence only | 48.339 | 52.368 |
| Fanout only | 38.228 | 38.816 |
| Combined | 59.018 | 60.373 |

Combined nonlookup is ~64% faster than M4.7's ~36.8 tok/s; it is not close to the general 160 target. A favorable repeated-copy lookup test (212 prompt, 128 output, 127 feedback rows) reported warm median 157.851 (range 157.425–157.984) vs matched ordinary control mean 59.015; 111 proposed and 111 accepted drafts across 16 multirow windows. One-shot and resident IDs matched on that workload. Factual chat exercised complete rejection and EOS. Real native intermediate acceptance, diverse code and a specifically verified no-proposal case remain unqualified. Do not claim 160 reached or extrapolate favorable lookup to general workloads.

Combined median coordinator stage roundtrips 2.682/5.795/7.575 ms; commit barrier 0.700 ms (baseline 3.541). Worker enqueue execution stage0/1 2.231/4.942 ms and final execute 6.897 ms. Final result send 1.008 -> 0.001 ms and compact removes the separately timed full logits download. These inclusive spans are not kernel-only times. Baseline cold startup once failed while earlier CUDA waited for slower HIP; identical retry passed, root cause unconfirmed. GPU peak busy sensor values 87/79/66%; no decode-phase average/overlap measured. Existing optimized roundtrip still serial. Native MTP assets unverified; vanilla 120 user-reported and exact mode not established.

Proceed to [M6](M6_PIPELINED_PERFORMANCE.md) for startup robustness, mandatory phase-aligned utilization and measured placement/MTP prerequisites. M5 remains partial; no historical KL or broader numerical parity claim.

## Pre-handoff local evidence and limitations

GitHub direct checkout DNS lookup failed in Chat AI container (remote API connector available). Local strict GCC C++17 tests compiled byte-identical performance.hpp (b5b76cd8...) and speculation.hpp (ebc96aa3...): 255 exhaustive 1–8-row prefix acceptance cases in each poll mode, invalid inputs, boolean parser, default vs 50us poll interval and trace format PASS. All **11** resident Python unit tests passed on byte-identical adapter/contract/test blobs (53037ef3, c5b8f281, 9e82d62f). They are CPU wire doubles; not proof of GPU correctness. Extended final pipeline CPU tests and native builds, actual M5 performance and numerical parity pending Oberon. No GitHub Actions. Target 160 remains unverified.
