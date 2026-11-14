# Report 0005 — M5 controlled profiling and decode optimization

**Status: Partial.** Tested `2d8babd4ce4f481d2f9e1496b5a2a748b945002d` on `feature/heterogeneous-cuda-window-admission`. 160 sustained accepted tok/s was **not reached** in the measured warm samples. Raw logs, configs, outputs, traces, telemetry and measurements: `/home/chris/tmp/strata-heterogeneous/m5-0005-eAYp3SbC/`. No production source changes were made.

## Build, setup and startup failure

Isolated checkout `.venv/bin/python` passed `pip check`; all three CMake caches use it. Host build, **39/39 CTests**, separate CUDA coordinator/worker and HIP worker builds passed with <=28 jobs. Resident plus frontend/server suite: **413 tests PASS** (`server-tests.log`). Actual coordinator link command includes original `src/spec/suffix_drafter.cpp`. CPU fanout/compact tests cover socket/shared, bad ACK and poisoning, but no independent withholding-first-ACK peer test was added.

New staging copies change only 16 to 6144/9216/9216 slots; max_cells=2048, max_t=8, reserve, model and plan unchanged. Model gate passed. Actual logs show CUDA 6144/8493465600 bytes, HIP0/HIP1 each 9216/12740198400 bytes, three zero-doorbell graphs, and zero host callbacks/misses on normal full-resident shutdown. GPU identities: RTX 4070 Ti SUPER `0000:04:00.0`, R9700 `0000:03:00.0` and `0000:07:00.0`; plan `cuda:0:0-12,hip:0:12-30,hip:1:30-48`, caller fingerprint 2026111301, pack index SHA-256 `c344c01f59f547ed8a59003857a82eb90499e5299154241dbca787acb54b5b7c` (not attestation of all GPU weights).

The first baseline live startup failed: CUDA timed out waiting while HIP full-cache startup was still finishing; supervisor then logged broken pipe (`socket-baseline.log`, `socket-baseline.json`). Workers exited and free CUDA memory returned to baseline; an identical retry passed. Existing configs use 30000 ms worker I/O timeout. A reproducible root cause or cold-start fix was not established; startup robustness remains **UNVERIFIED**, not hidden by the retry. No timeout/reserve/precision guard was changed.

## M5.1 — PASS, bounded trace

Separate traced baseline vs combined, warm request 2, 63 one-row decode windows (`profile-summary.log`; inclusive wall-clock spans, do not add nested/overlapping components):

| Median ms | Baseline | Combined |
| --- | ---: | ---: |
| coordinator stage 0 / 1 / 2 roundtrip | 3.631 / 8.465 / 11.621 | 2.682 / 5.795 / 7.575 |
| coordinator commit barrier | 3.541 | 0.700 |
| worker stage 0 / 1 enqueue execution | 2.234 / 4.948 | 2.231 / 4.942 |
| worker final execute | 7.856 | 6.897 |
| final logits download | 0.411 | absent in compact mode |
| final result send | 1.008 | 0.001 |
| nonfinal D2H wait stage 0/1 | 1.053 / 1.053 | 0.102 / 0.102 |

Sequential stage execution/roundtrips, especially final and HIP stage 1, dominate combined per-token latency; worker fence sleeps and commit/logit transport also substantially affect baseline. Precise kernel-only attribution is not established. The traced runs are separate from throughput samples.

## M5.2–M5.5 — PASS for bounded controls

42 prompt tokens, 128 output tokens, 127 committed feedback rows, length stop; four runs per mode, first excluded as cold. Table gives warm feedback tok/s mean / median / range:

| Transport | Baseline | Compact only | Fast fence only | Fanout only | Combined |
| --- | --- | --- | --- | --- | --- |
| socket | 36.462 / 36.520 / 36.338–36.527 | 40.643 / 40.649 / 40.617–40.664 | 48.448 / 48.339 / 48.086–48.919 | 38.202 / 38.228 / 38.067–38.312 | 59.079 / 59.018 / 58.934–59.285 |
| shared | 37.255 / 37.049 / 37.025–37.690 | 41.335 / 41.360 / 41.277–41.368 | 52.278 / 52.368 / 52.004–52.462 | 38.905 / 38.816 / 38.797–39.101 | 60.299 / 60.373 / 60.119–60.405 |

Every 128-token sequence matched exactly across all 40 benchmark requests. Separate 42/64 baseline/combined chunk 1/auto controls also matched each other (`64-*.json`), with warm baseline auto 36.76–37.00 and combined auto 59.41–59.66 tok/s. Per-mode untraced JSON/logs, preflights and `measurement-summary.log` retain the full IDs and startup/shutdown/TTFT values. Warm best nonlookup **shared combined mean 60.299, median 60.373, range 60.119–60.405**; below 120 and 160.

Compact final verifier refuses disabled `nan_guard_on`; `Verifier::run` invokes the GPU logits nonfinite-row guard and `note_nonfinite`, and coordinator refuses diagnostic logits when compact is requested. Traced compact has no `final_logits_download`. Native full-logit diagnostic capture remained functional: two STFR windows each had 248320 finite logits (`diagnostic-validation.log`). This does not establish equivalence of every possible output or historical cross-backend KL parity.

## M5.6 — PARTIAL: verified lookup, bounded correctness gaps

Combined buffered resident control vs lookup with repeated copy prompt: 212 prompt tokens, 128 output tokens, 127 accepted feedback rows; four requests matched exact output IDs, including one-shot control vs lookup (`control.json`, `lookup.json`, `one-control.json`, `one-lookup.json`). Each lookup request reported `proposed=111 accepted=111 multirow_windows=16`; average is 127/16 = 7.94 feedback rows per multirow window for this workload. Three warm lookup samples **mean 157.753, median 157.851, range 157.425–157.984** accepted feedback tok/s; corresponding control mean 59.015. This is a highly repetitive favorable input, not general throughput; still below 160.

Ordinary chat produced identical `The capital of France is **Paris**.` and stop in two lookup/control requests; lookup proposed 1 and accepted 0, exercising first rejection and EOS. 128-token output cap was observed. CPU preflight refused lookup+stream and lookup+chunk1 (`refusal-*.json`). Existing CPU prefix tests cover intermediate partial acceptance; **a real GPU case with a nonzero but not fully accepted draft prefix, varied code prompts, and a specifically demonstrated no-proposal run remain unverified**. No unverified suffix was intentionally published. Buffered lookup through existing loopback server returned the same chat response and closed cleanly (`lookup-http.log`, `lookup-server.log`).

## M5.7 — PASS for bounded measurements; target and wider quality UNVERIFIED

Combined nonlookup streaming server: disconnect after streamed `The`, cancellation ACK, subsequent A matched, stable worker PID/start records and ordinary owner shutdown (`stream-client.log`, `stream-after.log`, `stream-workers-{before,after}.log`, `stream-shutdown.log`). Per-device passive sampler 45 one-second samples in `telemetry.jsonl`: peak used bytes CUDA 11879317504, HIP0 17327632384, HIP1 17411821568; sampled max utilization 87%/79%/66%; mapping in `telemetry-map.log`. Post-run samples returned to idle VRAM (`post-memory.jsonl`). `/usr/bin/time -v` around combined coordinator and its children showed 24.23 s user, 2.57 s system and 174% CPU over 15.37 s wall (`combined-cpu-time.log`); this includes loading and all children, not a decode-only CPU profile.

## MTP and next steps

Existing `MtpDrafter::load` expects its prepared `dense.txt`, `dense.bin` and `experts.bin`, a compatible native final head/output weights and stage-local draft K/V/session state; verifier exposes final-stage `final_R`. Integrating it safely requires final HIP-stage ownership and lifecycle/positive-prefix wiring rather than foreign device pointers. The presence/compatibility of MTP runtime assets in the approved artifacts was not established; no alternate model was fetched. The claimed vanilla two-R9700 ~120 tok/s is user-reported, not independently reproduced: exact CLI, model/quantization, context, draft mode, sampler and timing denominator are needed.

Next bounded investigation: improve cold-start readiness without relaxing I/O failure guards, and measure final-stage graph/head and stage-1 execution with existing nonintrusive profiling, then evaluate compatible final-stage MTP assets and verified accepted-prefix state. The trace suggests the remaining ~2.2/4.9/6.9 ms worker execution components alone are substantial; no guarantee of reaching 160 is inferred. Native fault recovery, full lookup partial-acceptance coverage, long contexts, optimized bulk prefill, MTP, broad cross-backend numerical parity and production readiness remain unverified. No GPU resets, forced GPU worker kills, driver changes, Actions, alternate model, new kernels or precision changes were introduced.
