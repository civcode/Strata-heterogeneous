# Report 0006 — M5 closure and M6 utilization/placement

**Status: Partial.** Tested `e9476b9aff6dacaad0ba05c48a0c79bfd3362dce` on `feature/heterogeneous-cuda-window-admission`. Evidence: `/home/chris/tmp/strata-heterogeneous/m6-0006-6gXSWyR8/`. No production source changes were made. Bounded startup, actual lookup edge cases, utilization and two placement candidates were exercised. Native fault recovery, cold page-cache startup, full placement numerical diagnosis and M6.2–M6.7 implementation remain unverified.

## Environment and startup — bounded PASS

Checkout-local `.venv/bin/python` passed `pip check` with `PYTHONHOME/PYTHONPATH` unset; all CMake caches select it. Host build and **40/40 CTests**, 413 resident/frontend/server Python tests, separate CUDA worker/coordinator and HIP worker builds passed, <=28 jobs. Logs: `environment.log`, `host-build.log`, `ctest.log`, `server-tests.log`, `cuda-build.log`, `hip-build.log`. CPU tests include late startup/partial READY/EOF; no independent withholding-first-ACK fanout test was added.

Five separate original-plan full-resident sessions started and shut down without retry or induced cache eviction: startup 7310.61, 7322.76, 7314.68, 7277.91, 7304.16 ms; shutdown 252.66–273.01 ms. Each logged all three workers ready and six request resets, exact full cache and zero host callbacks/misses; `verification.log`, `baseline-on.log`, `combined-on.log`, `baseline-off.log`, `combined-off.log`, `shared-off.log` and matching JSON. No assertion about true cold page-cache startup or all hardware stalls: the earlier M5 30-second startup failure was not reproduced or causally proven. New separately bounded 120-second startup wait was not tested by intentionally delaying or faulting GPU workers; worker I/O/fence stays 30 seconds.

Original Q2_0 model gate and passive readiness passed (`model-gate.log`, `readiness.json`); stage-config copies change only max_cache_slots 16 → 6144/9216/9216. Model fingerprint 2026111301, pack-index SHA-256 `c344c01f59f547ed8a59003857a82eb90499e5299154241dbca787acb54b5b7c`; caller fingerprint is not GPU weight attestation. Plan CUDA RTX 4070 Ti SUPER `0000:04:00.0` → HIP R9700 `0000:03:00.0` → HIP R9700 `0000:07:00.0`; 2048 max cells, max_t 8, original reserves/precision. Monitor indices independently mapped in `telemetry-map.log`.

## M6.0 throughput and sampled utilization — bounded PASS; objectives not generally met

Same 42-token prose prompt, 128 returned tokens, 127 feedback rows, six sequential requests, first excluded. Original 12/18/18:

| Mode | Warm mean / median / range feedback tok/s | Sampler |
| --- | --- | --- |
| Socket baseline | 36.317 / 36.238 / 36.171–36.588 | on |
| Socket baseline | 36.662 / 36.720 / 36.382–36.805 | off |
| Socket combined | 59.241 / 59.188 / 58.793–59.609 | on |
| Socket combined | 59.186 / 59.205 / 59.005–59.389 | off |
| Shared combined | 60.814 / 60.802 / 60.732–60.927 | off |

All five modes' 128 output IDs matched. On/off measurements are separate sequential runs, not a controlled overhead confidence interval. JSON contains startup, prefill/reset, first committed prediction, decode, tail and shutdown timestamps. Raw sampler at 250 ms and analyzer outputs (`baseline-on-utilization.json`, `combined-on-utilization.json`) separate phases; very short idle/tail/prefill phases have low-sample warnings and are not sustained evidence. Warm decode pooled over five requests:

| Mode | CUDA mean/median/peak % | HIP0 mean/median/peak % | HIP1 mean/median/peak % | Complete cycles (expected approx.) |
| --- | --- | --- | --- | --- |
| baseline | 8.09 / 7 / 10 | 22.61 / 23 / 32 | 30.19 / 30 / 43 | 69 (69.94) |
| combined | 12.91 / 12 / 15 | 33.60 / 34 / 36 | 45.72 / 45 / 52 | 43 (42.88) |

Each device had 0% of samples >=90% or >=95%; CUDA below 10% in 60.87% baseline, 0% combined, HIP devices 0% below 10%. Complete same-cycle all-device >=10%: 39.13% baseline / 100% combined; >=90% and >=95%: **0%** both. These are 250-ms sampled busy/overlap proxies, not compute occupancy or true simultaneous kernel execution; acquisition skew and sample coverage checks are in analyzer JSON. Warm combined average sampled GPU powers 60.41/81.12/96.02 W imply roughly 4.01 GPU-only J per feedback token at pooled 59.24 tok/s, not wall power and subject to sensor windows and sample coverage. VRAM used during warm combined was 11879317504 / 17327640576 / 17411833856 bytes. Idle before/final samples and remaining-worker check: `idle-before.jsonl`, `idle-final.jsonl`, `remaining-workers.log`.

Favorable repetitive copy task (212 prompt, 128 output, 127 accepted feedback rows): six lookup runs, five warm mean **160.660**, median **161.652**, range **155.883–162.328**, pooled **160.623 tok/s** over ~3.95 s, 14 complete telemetry cycles. All 6 matched two nonlookup controls exactly (`lookup-repeat-compare.log`); each reported 111/111 proposed/accepted and 16 multirow windows. This exceeds 160 pooled for this highly repetitive bounded workload, **not** a general sustained decode result. Warm sampled GPU mean/median/peak CUDA 12.33/12/15%, HIP0 25.44/27/30%, HIP1 52.25/51/70%; all-device >=90/95% 0%. Thus **95–100% productive utilization is not demonstrated**, and busy sensors cannot prove productive compute.

## M5 lookup gaps — bounded PASS with limits

Actual mixed prose/pattern/code prompts in `diverse-{control,lookup}.{json,log}`: two independent repeats each, exact 64-token control/lookup IDs, positions and full-stage positive commits verified in `verification.log`. Lookup request 1 proposed 14/accepted 3, request 2 37/6, request 3 14/0; these establish nonzero partial acceptance and first rejection, not every possible rejection location. Short France completion had zero proposals in both requests (`no-proposal.log`); prior report 0005 documented EOS, capped output and 111/111 full acceptance. Buffered lookup loopback matched France response; optimized nonlookup SSE disconnect after streamed `The`, cancellation ACK, subsequent A and stable worker IDs succeeded (`lookup-http.log`, `stream-client.log`, `stream-after.log`, `stream-workers-{before,after}.log`, `*-shutdown.log`). No native fault injection.

## M6.1 placement — measured, quality qualification PARTIAL

Original, 12/20/16 and 14/19/15 kept device order, geometry, PLE-owning layer 1 on CUDA, budgets/reserves and artifacts. New staged configs, CPU model gate/readiness and preflight passed; memory estimator omissions were retained; actual full allocations/reserves succeeded. Every benchmark stage logged all-resident zero-doorbell and no host misses. Six requests per candidate, same 42/128 workload, first excluded; 250-ms telemetry:

| Split | Warm median tok/s | CUDA/HIP0/HIP1 mean busy % | Complete cycles | Output vs original |
| --- | ---: | --- | ---: | --- |
| 12/18/18 | 59.188 | 12.91 / 33.60 / 45.72 | 43 | control |
| 12/20/16 | 59.518 | 13.02 / 37.64 / 42.26 | 42 | identical across six |
| 14/19/15 | 59.588 | 16.36 / 35.21 / 41.21 | 42 | reproducibly differs at output index 40: 10842 → 17661 |

All-device >=90/95% samples were 0% for both candidates. Full warm VRAM bytes for 12/20/16 were 11879317504/18968121344/15771312128; 14/19/15 13508804608/18158592000/14957535232. These are sampled utilization/VRAM, not proof of all possible memory peaks. Same-candidate six-session-request outputs were repeatable; fresh process repeats were not separately run. `quality-comparison.log` records actual text and first difference. Existing non-generation STFR smoke allows at most 64 explicit tokens/windows; this first divergence requires 42 prompt + 40 output tokens, so teacher-forced logits at that point were not obtained with that existing path. The 14/19/15 difference is **unexplained**, not evidence by itself of a transfer bug. No new default is recommended from these tiny timing differences; 12/20/16 matches this one prompt but is not broadly qualified.

Separate traced warm one-row medians, inclusive (`trace-summary.log`): original combined stage 0/1/2 roundtrips 2.727/5.832/7.562 ms and barrier 0.703 ms; 12/20/16 2.677/6.478/6.930 and 0.713 ms; 14/19/15 3.061/6.258/6.688 and 0.700 ms. Worker native execute spans were original 2.235/4.952/6.900 ms versus 12/20/16 2.232/5.565/6.264; do not sum nested spans or call them isolated kernel durations. Python monotonic request bounds contained 1701+ C++ steady-clock spans per traced optimized request on this host/boot, confirming practical epoch alignment; no generic cross-host claim. Idle PCIe sysfs snapshot (`pcie-links.log`): CUDA current 2.5 GT/s x4, max 16 GT/s x16; AMD both current/max 32 GT/s x16. Idle link state is not active bandwidth or negotiated-load measurement.

## M6.2 design prerequisite — UNVERIFIED assets; not implemented

Approved config-named GGUF shard metadata inspected read-only: 37623740192-byte shard (1223 tensors) and 28800138432-byte shard (1 tensor) had no names containing `mtp` or `nextn`; configured Q2_0 pack has no MTP runtime path (`mtp-asset-audit.log`). This name check does not prove model absence; no broad artifact search was done. Existing `MtpDrafter::load` needs an explicitly authorized prepared `dense.txt`, `dense.bin`, `experts.bin`, compatible native head/output weights, draft K/V/state and memory budgeting; runtime files, hashes, headers, sizes and Q2_0 compatibility remain unknown. Source exposes final-stage `Verifier::final_R`, MTP load/bind/prefill/draft and K/V restore; existing `--pipeline-windows` and verifier snapshot/restore code are same-backend designs, not evidence of cross-process transactional safety. A bounded next design is final-stage-owned MTP with explicit lifecycle/asset identity, accepted-prefix catch-up and reset, then independently owned multi-window state/slots and coordinated restore before any publication. Ordinary suffix/MTP proposal generation is not cross-window lookahead. First-stage placement and long-prompt pipelining are hypotheses; no foreign GPU pointers, speculative kernels or second engine are authorized by these results.

Vanilla dual-R9700 ~120 tok/s remains user-reported; exact executable/revision, CLI, model/quant, context, MTP/pipeline, sampler and throughput denominator are needed. M6.2–M6.7 not implemented. No GPU resets, driver changes, forced worker kills, Actions, precision/budget/reserve changes or unrelated worktree access. Existing stash and untracked probe preserved. Broad numerical parity and production readiness are not claimed.
