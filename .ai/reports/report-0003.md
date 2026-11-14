# Report 0003 — bounded M4 validation on Oberon

**Status: Completed for bounded acceptance; native fault recovery, full packaging and wider contexts remain unverified.** Tested `e619892c823c98a03ec7ec090a737b042b6f544c` on `feature/heterogeneous-cuda-window-admission`; roadmap baseline `7266b69f823d571d54b77436c96c69a6a09f9952`. Evidence: `/home/chris/tmp/strata-heterogeneous/m4-0003-It5pX6Qj/`. Explicit user authorization for all tests was provided after the initial CPU-only report commit `ce54edd0`. No source fix was required.

## Build and audit

Checkout-local `.venv/bin/python`, Python 3.12 with `PYTHONHOME`/`PYTHONPATH` unset: `pip check` PASS. All three CMake Python3 caches use that interpreter (`environment.log`). Host build PASS, **39/39 CTests PASS**; separate CUDA worker/coordinator and HIP worker builds PASS, <=28 jobs. Existing CPU frontend/server compatibility: **402 tests PASS** (`server-tests.log`); resident helper/wire tests: 9/9 PASS (`resident-tests.log`). Logs: `host-build.log`, `ctest.log`, `cuda-build.log`, `hip-build.log`.

Source audit covered progress sequencing, final/stream cross-check, final capped-token withholding, generator-close cancellation, context/chunk/serialization limits, request serialization and clean-restart process-group refusal. CPU doubles cover buffering, streaming, cancellation, mismatch poisoning and restart refusal; these are not native fault tests.

## M4.1 — PASS, bounded performance and telemetry

Resident socket/shared, chunk 1/auto, A/B/A repeated twice per mode. All matching prompts produced matching IDs: A `[11751,13]` (` Paris.`, stop), B `[44745,11,864,369,1048,279,7526,3177]` (` Vienna, which is also the largest city`, length). Reset and committed positions for all stages and requests verified in `validation-summary.log`; raw timings and traces in `{socket,shared}-{1,auto}.{json,log}`.

| Mode | Startup ms | Warm A prefill ms | Warm A prompt tok/s | Warm A decode feedback tok/s |
| --- | ---: | ---: | ---: | ---: |
| socket chunk 1 | 1255.4 | 241.5 | 20.70 | 21.37 |
| socket auto (8) | 1237.6 | 127.1 | 39.33 | 21.33 |
| shared chunk 1 | 1280.4 | 233.9 | 21.38 | 21.32 |
| shared auto (8) | 1242.7 | 120.1 | 41.64 | 21.16 |

Warm A is request 3 of each session; first committed prediction latency equals the prefill figure. These short individual samples are not a general speedup claim. JSON contains all request timings and shutdown times. Passive sampler (`telemetry.jsonl`, 1 s interval, 30 samples/device) covered baseline, 583-token native run and cleanup. Monitor-to-PCI mapping independently recorded in `telemetry-map.log`: NVIDIA index 0 `0000:04:00.0`; AMD indices 0/1 `0000:03:00.0`/`0000:07:00.0`. Observed max utilization 90%/36%/40%; min–max used bytes 468779008–3406823424, 59912192–4610342912, 59912192–4694577152 respectively. Samples do not establish sustained utilization or a capacity benchmark.

## M4.2 — PASS, bounded context checks

CPU validation includes over-cap and serialized 64 KiB refusal. With unchanged stage configurations, requesting 8192 resolved to **2048 total** and auto chunk 8 (`preflight.json`). Fresh versus two resident runs matched for prompts of **583 and 1159 tokens** with four output tokens `[271,248068,198,760]` (`\n\n<think>\nThe`); all stage reset and commit position checks passed (`validation-summary.log`). A larger 256-repeat prompt was refused at CPU preflight with `prompt/generation budget exceeds validated context or vocabulary` (`context-256-preflight.json`); no GPU execution for it. 1159 is the longest actual prompt tested; no claim for 2048 or 8192-token inference.

## M4.3 — PASS, bounded chunk comparison

Auto resolved to 8; chunk 1 and auto produced the same recorded A/B/A IDs in socket and shared runs, with monotonic stage commits and timing results above. This uses ordinary verifier windows, not an optimized bulk prefill engine.

## M4.4 — PASS, bounded streaming/cancellation

Generated dedicated configuration started existing loopback server. Buffered and opt-in SSE on the same chat prompt both returned `The capital of France is **Paris**.`; eight SSE text deltas were observed **before** native request completion (`sse-client.log`, `stream-events.json`, `buffered.json`). A client disconnected after streamed text `The`; server logged `HETERO_REQUEST_DONE request=3 finish=cancelled next=27`; subsequent A returned the original response (`sse-cancel.log`, `after-cancel.log`, `server.log`). Streaming exposes a committed partial response before final validation and is not an automatic replay guarantee.

## M4.5 — PASS, bounded clean lifecycle; native recovery UNVERIFIED

CPU fake-peer tests verify refusal to restart while an old process group exists, and allow clean-group restart. A live server's three worker PID/start-time records were identical before and after A/B/A (`workers-before.log`, `workers-after.log`); the streaming/cancellation server retained its worker group through post-cancel generation (`processes.log`, `server.log`). Normal owner SIGTERM produced `HETERO_CLOSED` and all owned processes exited (`shutdown.log`, `server-repeat.log`). No native fault or forced termination was injected; real crash recovery, hanging workers and abrupt parent death remain unverified.

## M4.6 — PASS, bounded source deployment

Existing CLI wrote and CPU-preflighted a new exclusive loopback config (`server-generated.json`, `config-generation.log`); existing `serve.server` ran the HTTP tests above. Binary packaging, install/profile launcher integration and concurrent serving remain deferred.

Plan `cuda:0:0-12,hip:0:12-30,hip:1:30-48`, Q2_0 source and pack as in JSON, pack-index SHA-256 `c344c01f59f547ed8a59003857a82eb90499e5299154241dbca787acb54b5b7c`; caller model fingerprint 2026111301 is not a cryptographic GPU-weight attestation. No resets, driver changes, Actions, budget or precision changes, numerical fixtures or new kernels. Existing stash, untracked pool probe and configs preserved. Historical cross-backend KL parity remains a separate non-gating failed diagnostic. No general production readiness claim.
