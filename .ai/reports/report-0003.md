# Report 0003 — M4 CPU/build qualification, live gates blocked

**Status: Partial.** Tested `e619892c823c98a03ec7ec090a737b042b6f544c` on `feature/heterogeneous-cuda-window-admission`; roadmap baseline `7266b69f823d571d54b77436c96c69a6a09f9952`. Evidence: `/home/chris/tmp/strata-heterogeneous/m4-0003-It5pX6Qj/`. This report is the only new tracked change.

The handoff expressly says no new authorization is implied and requires existing explicit authorization for the **exact** live GPU/model operations. Previous user authorization and report 0002 covered M3.5 resident/server validation, not the new M4 benchmark, progressively longer contexts, telemetry, or token-streaming GPU sessions. Therefore no M4 native workers, live server, passive GPU sampler or GPU measurements were started. Explicit M4 live authorization is required to finish; these gates are not represented as passed. Existing prior M3.5 results remain at `.ai/reports/report-0002.md` and `/home/chris/tmp/strata-heterogeneous/m35-0002-PF0NDL1z/`.

## Host, builds and audit

- Checkout-local `.venv/bin/python`, Python 3.12 with `PYTHONHOME`/`PYTHONPATH` unset: `pip check` PASS. Existing CMake Python3 caches all point to that venv (`environment.log`).
- Host build PASS, 39/39 CTests PASS, including new resident contract test. Separate CUDA worker/coordinator build and HIP worker build PASS; <=28 jobs. Logs: `host-build.log`, `ctest.log`, `cuda-build.log`, `hip-build.log`.
- Existing CPU frontend/server compatibility: **402 tests PASS** (`server-tests.log`); new resident helper/wire tests: 9/9 PASS (`resident-tests.log`). These use CPU doubles, not actual inference.
- Inspected progress sequencing, prediction/finish cross-check, final capped-token withholding, generator-close cancellation, context/chunk and serialization limits, serialized request lock and clean-restart process-group refusal. Existing CPU tests exercise buffering, streaming prefix, cancellation, mismatch poisoning, withholding and restart refusal. No source defect was established or changed. Native fault recovery remains unverified.
- Source-bound CPU-only CLI preflight with requested context 8192 resolved to configured minimum **2048** and auto chunk **8**, using real model metadata (`preflight.json`, `preflight.log`). These are admission limits, not actual tested context lengths. CPU-only CLI config generation with stream opt-in succeeded (`server-generated.json`, `config-generation.log`); no server was launched.

## M4 milestone status

| Milestone | Status | Evidence and outstanding work |
| --- | --- | --- |
| M4.1 performance | **UNVERIFIED** | No authorized M4 native benchmark or telemetry; startup, steady-state, throughput and device utilization/VRAM measurements unavailable. No speedup claim. |
| M4.2 context | **UNVERIFIED** | CPU limit/64 KiB refusal tests PASS; configured max 2048 observed in preflight. No progressive native prompt lengths measured. |
| M4.3 prefill | **UNVERIFIED** | CPU auto-chunk calculation gives 8, but chunk 1 vs auto GPU behavior/timing not run. No bulk prefill claim. |
| M4.4 streaming | **UNVERIFIED** | CPU mock contract tests pass; no actual M4 HTTP SSE token-before-completion, decode cancellation or post-cancel isolation observed. |
| M4.5 recovery | **UNVERIFIED** | CPU fake-peer restart refusal and clean-group tests pass; live lifecycle for this revision and native fault recovery not qualified. No forced worker kills. |
| M4.6 deployment | **UNVERIFIED** | Dedicated loopback source config generated and preflighted; actual M4 server startup/API/cleanup not run. Binary packaging/profile launcher integration deferred. |

No GPU resets, driver changes, Actions, budget or precision changes, numerical fixtures or new inference kernels. Existing stash, untracked pool probe and configs preserved. Next: with explicit authorization for the M4 runbook's native benchmark, context, streaming, telemetry and loopback-server operations, execute the remaining ordered gates and retain raw logs. No full-batch PASS is claimed.
