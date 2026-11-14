# Report 0002 — bounded M3.5 resident validation

**Status: Completed for the bounded runbook; broader failure recovery and production readiness remain unverified.** Tested source revision `992fc289634e228f57cd062eb8a35fcc01fbc2cb`, branch `feature/heterogeneous-cuda-window-admission`. Logs and raw outputs: `/home/chris/tmp/strata-heterogeneous/m35-0002-PF0NDL1z/` on Oberon. This report is the only change in the reporting commit.

## Build and environment

Checkout-local `.venv/bin/python` (Python 3.12), with inherited `PYTHONPATH` and `PYTHONHOME` unset, passed `pip check`. All three CMake caches resolve Python3 to that interpreter. Standalone host build PASS, **38/38 CTests PASS**, CUDA worker/coordinator and separate HIP worker builds PASS, at <=28 jobs. Logs: `environment.log`, `host-build.log`, `ctest.log`, `cuda-build.log`, `hip-build.log`. The prior `d8a43d5` host compile failure was corrected in the tested starting revision by synchronized native/expert verifier doubles; both native-stage CPU contract tests passed. No new fix was required here.

## M3.5a — PASS, bounded source audit and CPU coverage

`FinalPipelineSession::begin_request` sends stage-specific resets, checks matching `request_ready` ACKs and poisons on failure; stage workers gate increasing IDs and idle state, execute native reset before ACK, enforce request/window identity and poison on protocol or execution errors. Native verifier bindings use `session_zero`, preserving the owning resources; coordinator begins each independent request at position zero. The server adapter serializes through a lock, checks committed progress, cooperatively cancels at a checkpoint, does not automatically reuse a failed set, and shuts down through the owning coordinator. CPU tests cover reset/replay/poison paths. This is static/CPU evidence, not proof of GPU fault recovery.

## M3.5b/e — PASS, bounded native resident operation

Fresh A/B and resident A/B/A, followed by a fresh resident shared-memory A/B/A, ran on Q2_0 with plan `cuda:0:0-12,hip:0:12-30,hip:1:30-48`. Fresh and resident tokens and termination matched:

- A `The capital of France is`: ` Paris.`, IDs `[11751,13]`, stop.
- B `The capital of Austria is`: ` Vienna, which is also the largest city`, IDs `[44745,11,864,369,1048,279,7526,3177]`, length.

`resident-verification.log` checks request IDs 1/2/3, per-stage request-ready resets at position zero and admission/provisional/commit sequences with monotonic committed positions: 7/12/7 windows per stage in both transports. `resident*.json` and `resident*.log` contain raw outputs and traces. Server coordinator PID 234130 and native children 234132/234133/234134 retained identical PID, parent and start time across all HTTP requests including cancellation; only one startup/device announcement per stage was observed and no reload is logged. This is process/lifecycle evidence of residency, not an independent measurement of every weight allocation. Physical identities: CUDA RTX 4070 Ti SUPER `0000:04:00.0`; HIP R9700 `0000:03:00.0`, `0000:07:00.0`. Pack index SHA-256 `c344c01f59f547ed8a59003857a82eb90499e5299154241dbca787acb54b5b7c`; model fingerprint 2026111301 is caller-supplied, not a weight-byte attestation. Model source and plan fingerprints are retained in JSON outputs.

## M3.5c — PASS, bounded cooperative cancellation and shutdown

A streaming client disconnected after observing request 4's first committed window. `server.log` recorded `HETERO_REQUEST_DONE request=4 finish=cancelled next=2`. A subsequent A request produced the same response as the original; worker identities remained unchanged (`workers-before.log`, `workers-after.log`, `workers-after-cancel.log`). A single SIGTERM to the owning server produced `HETERO_CLOSED`, and the coordinator and all three owned worker PIDs exited (`shutdown.log`). No GPU worker was intentionally killed. Client disconnect was observed in an active prompt, not an injected device fault. Abrupt parent death, hung workers and native failure recovery remain unverified.

## M3.5d — PASS, bounded existing loopback HTTP API

The existing `serve.server` heterogeneous engine served sequential `/v1/chat/completions` A/B/A and post-cancellation A requests. A responses: `The capital of France is **Paris**.`; B: `The capital of Austria is **Vienna`. Each finished `length` at the requested eight tokens. Responses, ready record and logs: `http-*.json`, `http-client.log`, `ready.json`, `server.log`, `after-cancel.log`. SSE cancellation is cooperative; this adapter buffers tokens, so it is not incremental token streaming.

## Scope and safety

No GitHub Actions, reset/reboot/driver unload, intentionally killed GPU workers, new kernels, numerical fixtures or speculative numerical changes. Untracked pool probe and stash were preserved. No long-context, concurrent/multiturn, sampling beyond greedy, optimized prefill, historical cross-backend KL parity, native fault injection or general production-readiness claim. Repeatability here is limited to the recorded prompts and hardware.
