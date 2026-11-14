# Report 0004 — M4.7 full routed-expert residency

**Status: Completed for bounded M4.7 acceptance.** Tested `6e295857e2781cd79a59ebccd11b4680f80153eb` on `feature/heterogeneous-cuda-window-admission`; no production source changes were needed. Raw logs, configurations, outputs and telemetry: `/home/chris/tmp/strata-heterogeneous/m47-0004-Xi15bUiZ/` on Oberon.

## Build and setup — PASS

Checkout-local `.venv/bin/python` (Python 3.12; `PYTHONHOME`/`PYTHONPATH` unset) passed pip check; all three existing CMake caches select it. Host build, **39/39 CTests**, separate CUDA worker/coordinator and HIP worker builds passed with <=28 jobs. Expanded native file-expert/config tests cover slot bounds, full/incomplete profile handling and host/device residency behavior with CPU doubles, not real GPU execution. See `environment.log`, `host-build.log`, `ctest.log`, `cuda-build.log`, `hip-build.log`.

Original stage configurations each specify **16** expert slots, 2048 cells, max_t 8 and 536870912-byte reserve. Only `max_cache_slots` was changed in new staging copies to 6144/9216/9216; original configs and unrelated stash/probe were untouched. Canonical complete 48x512 profile/model gate passed (`model-gate.json`); passive memory estimator reported no blockers but explicitly excludes session/native dense/PLE/runtime allocations (`readiness.json`); source-bound CLI preflight passed (`full-preflight.json`). Initial CUDA free VRAM was 15929 MiB by NVML; AMD telemetry showed 59912192 bytes used of 34208743424 each. Runtime device logs independently identify CUDA `0000:04:00.0` RTX 4070 Ti SUPER and HIP R9700s `0000:03:00.0`/`0000:07:00.0`. The plan is `cuda:0:0-12,hip:0:12-30,hip:1:30-48`, caller model fingerprint 2026111301; pack index SHA-256 `c344c01f59f547ed8a59003857a82eb90499e5299154241dbca787acb54b5b7c`. Model identity does not attest all GPU weight bytes.

## Full expert residency and fallback — PASS, bounded runs

Actual worker startup and shutdown logs show:

| Stage | Filled / total | Expert payload bytes | Native graph | Host callbacks / misses |
| --- | ---: | ---: | --- | ---: |
| CUDA 0 [0,12) | 6144 / 6144 | 8493465600 | 100% VRAM resident, zero-doorbell | 0 / 0 |
| HIP 0 [12,30) | 9216 / 9216 | 12740198400 | 100% VRAM resident, zero-doorbell | 0 / 0 |
| HIP 1 [30,48) | 9216 / 9216 | 12740198400 | 100% VRAM resident, zero-doorbell | 0 / 0 |

All five full resident benchmark modes checked exact counts/bytes, three graph diagnostics and three normal-shutdown zero-callback/zero-miss statistics (`benchmark-summary.log`); full HTTP server logs show the same. These counters measure host-served callbacks and do **not** count GPU route entries. By contrast hybrid long-run callbacks/misses were CUDA 2484/37185, HIP0 3726/55716, HIP1 3726/56004 (`hybrid-long.log`), confirming a materially different host-fallback path. No claim that every non-expert operation is GPU-resident.

## Matched performance — PASS, bounded workload

Same revision, devices and settings; hybrid original versus full copied configs; resident A/B/A repeated three times for socket/shared and chunk 1/auto; matched 42-token prompt, 64-token capped completion repeated three times using socket and auto chunk. All stage request-ready resets and contiguous committed positions checked (`commit-verification.log`). Raw timing, IDs, text and startup/shutdown retained in `{hybrid,full}-{socket-1,socket-auto,shared-1,shared-auto,long}.{json,log}`; full startup includes profile loading/cache copies and is not steady-state inference.

| 42-token / 64-token long workload, auto chunk | Startup ms | Warm prefill ms (runs 2,3) | Warm feedback decode tok/s (runs 2,3) | Shutdown ms |
| --- | ---: | --- | --- | ---: |
| Hybrid 16 slots/stage | 1249.4 | 915.17, 907.44 | 21.31, 21.37 | 162.26 |
| Full residency | 7271.74 | 415.03, 423.51 | 36.82, 36.85 | 242.79 |

For this matched long sample, warm feedback decode is about 1.73x hybrid; this is not a general throughput guarantee. Startup differed (first full socket run 10883.93 ms; later full runs ~7.3 s), so cold/loading effects are separate. Short modes and all individual observations are in `benchmark-summary.log`; no values are extrapolated to bulk prefill or longer contexts.

Fresh A twice and resident A agreed in both modes: `[11751,13]` (` Paris.`). Fresh B twice and resident B agreed within each mode but **differed between modes**: hybrid `[44745,11,864,369,1048,279,7526,3177]` (` Vienna, which is also the largest city`); full `[44745,11,864,369,1048,279,3046,579]` (` Vienna, which is also the country's`). Long completions also differed between modes but were repeatable within each. The existing hybrid and all-resident execution paths differ; the observed differences alone do not identify a transfer/state defect or establish their numerical cause. No kernel or precision changes were made to force equality.

## Telemetry and release — PASS, bounded observation

Independent NVML/sysfs monitor-to-PCI mapping: `telemetry-map.log`. Existing passive sampler collected 180 timestamped 1-second samples during sequential hybrid and full long sessions (`telemetry.jsonl`, `telemetry.log`); observed peak used VRAM CUDA 11879317504, HIP0 17327644672, HIP1 17411874816 bytes, with max sampled utilization 87%/59%/73%. Baseline-to-peak used values are in `additional-summary.log`; sampling is not a proof of instantaneous maxima. After normal full-server cleanup, CUDA used 468779008 bytes and each AMD 59912192 bytes, matching idle baseline (`post-cleanup-memory.log`); owned coordinator and workers exited.

## Full-mode server and cancellation — PASS, bounded

Generated staging config and existing loopback server: buffered and SSE responses both `The capital of France is **Paris**.`; SSE deltas arrived before native completion (`http.log`). Client disconnected after a streamed token, request 3 acknowledged `finish=cancelled`; next A matched the first (`http-after-cancel.log`, `server.log`). Three worker PIDs/start times were identical before and after (`workers-before.log`, `workers-after.log`); a single SIGTERM to the owner led to `HETERO_CLOSED` and all owned processes exiting (`shutdown.log`).

No GPU reset, reboot, forced worker kill, driver changes, budget/reserve/precision change, Actions, new kernels or independent numerical fixtures. Native fault recovery, abrupt parent death, broad numerical parity, long contexts, optimized bulk prefill, packaging and production readiness remain unverified.
