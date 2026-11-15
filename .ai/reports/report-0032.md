# 0032 — real workload qualification (partial)

**Status: NOT QUALIFIED.** Source tested: `02893b678f6419da4185151126e384ce9e90ef54` on `feature/heterogeneous-real-workloads`. This is a partial execution, not completion of the handoff. Ordinary 2K-capacity serial and native-bulk HTTP inference worked on short controls, but coding quality failed and tool requests were refused. Long context, agentic tasks, utilization, peak memory and production readiness are not established. The unchanged numerical diagnostics in reports 0027.15 and 0030 remain FAIL; no thresholds were changed.

## Milestones

| Milestone | Status and evidence |
| --- | --- |
| 0032.1 build and CPU gates | PARTIAL: three existing host/CUDA/HIP caches built with `-j28`; host CTest 99/99; `unittest discover -s tools -p 'test_heterogeneous*.py'` 318/318, including the workload benchmark tests; resident suite 14/14; pip check clean. `pytest` is not installed; unittest was used instead. Isolated Python 3.12.3 checkout `.venv`, CMake Python set to that interpreter in all caches. Full specified negative preflight matrix, explicit extended wire/16 MiB audit and new raw asset audit NOT_RUN. |
| 0032.2 ordinary controls and invariance | PARTIAL: serial CLI actual chat returned `4`, natural stop, 23 prompt tokens, one visible completion token, clean close, all three stages full resident and zero host misses. Original bounded MTP, fresh/resident A/B/A, default capture invariance and bridge same-mode tests NOT_RUN. |
| 0032.3 serial API | PARTIAL: loopback streaming/nonstream short requests succeeded; explicit tools request returned HTTP 400 unsupported. Disconnect was observed as native cancelled, but immediate subsequent requests got 429 while the active request was still draining; post-cancel reset/next-request proof NOT ESTABLISHED. |
| 0032.4 native bulk API | PARTIAL: chunk 128 short controls completed with `HETERO_NATIVE_PREFILL_COMMITTED`, clean shutdown, full residency and zero host misses. Separate bulk CLI, fresh/resident comparison and cancellation NOT_RUN. |
| 0032.5 longer contexts | NOT_RUN: capacity 2048 configured, but only 23–49 prompt tokens exercised. No 2K near-cap, 8K, 16K or 32K qualification or resource audit. |
| 0032.6 real coding agents and quality | NOT_RUN: no autonomous tasks or scoped fixture evaluations. Frozen replay is not executed. The short synthetic coding control fails. Tool round-trip unavailable through the current engine. No perplexity or vanilla reference comparison. |
| 0032.7 performance and telemetry | PARTIAL: short HTTP controls and host timing below; sustained utilization, sampled peak GPU memory, meaningful generation throughput, balanced independent launches and profiling NOT_RUN. No optimization performed. |
| 0032.8 integrity and safety | PARTIAL: no fallback reported, full residency and zero misses for observed requests, orderly close. Comprehensive bridge/state/finiteness, original raw audit and all failure paths not reverified. |
| 0032.9 / 0033 | NOT_RUN: extended qualification, real-task performance target and production admission. |

## Setup and identity

No downloads, repack, GPU setting changes, reserve reductions, precision overrides or GitHub Actions. Default weight/activation environment left unset. Existing forward `cuda:0:0-12,hip:0:12-30,hip:1:30-48` configs from `/home/chris/tmp/strata-heterogeneous/m6-0008-XQLzkX2c/original-configs/` retain 2048 cells, max_t 8, full slots and 512 MiB reserve. Config SHA-256 in stage order: `c339ebd67e3aff8e520922666b0837dc1653d71ef62640e8ab7795e8859f5dd3`, `84afa07f8948ac11503325a82871339e545b95c89004f75d5ced2bd0170d51bb`, `6dd18ad9ba076fb3e5bdbe82b2ed98612a9dbaedf2447100382628cd81602ab6`. Configured asset index identity is `c344c01f59f547ed8a59003857a82eb90499e5299154241dbca787acb54b5b7c`; this is not a new raw audit (see report 0027-9).

Rebuilt binary SHA-256: coordinator `6c455bde4334b9e647edd5eefff8ad006bbb169df29b22e4cd55383f7c92cda4`; CUDA worker `725d62b69d744856b6d855a6a8e81f3673c153265efbf39619672c0d5919f3d5`; HIP worker `76d25eaddb46af45619efbab893d160be38dd8eabbb29d5266af76d8f0f83fab`. Both server configs were generated via source-bound no-run CLI, shared transport, grouped HIP, streaming, workload mode, max context 2048, max new 512, prompt chunk 1; bulk additionally uses native prefill chunk 128. Logs show CUDA 04:00 and HIP devices, full stage residency, zero misses and `HETERO_CLOSED`; no owned inference process remained after shutdown. CLI startup 7.821 s, separate from request timing; its short request prefill 639.5 ms. Those are single observations, not benchmark aggregates.

## Actual short HTTP controls

Two separate loopback server launches, serial and bulk; each below is the median of three sequential repetitions of the same four-case subset of smoke, with no warmup exclusion, not balanced launch orders. All completed naturally with `stop`. The original full smoke was first attempted with three repeats: after the first disconnect it immediately hit HTTP 429 for subsequent cases, so its 3-repeat summary is not a valid comparison. To allow non-cancellation controls to complete, the separate subset excludes tool, cancel and post-cancel; it does not replace or hide the full run. Client TTFT is first visible stream fragment; nonstream TTFT is unavailable. Prompt rate is the service's `heterogeneous_host_commit_progress` proxy including reset/IPC, not GPU kernel throughput. These are tiny synthetic controls, not coding-agent performance.

| Case (prompt tokens) | Expected / actual | Pass serial / bulk | TTFT serial / bulk ms | Prompt proxy serial / bulk tok/s | Total serial / bulk ms |
| --- | --- | --- | ---: | ---: | ---: |
| arithmetic (27) | `95` / `95` | 3/3 / 3/3 | 632 / 263 | 42.8 / 103.3 | 678 / 307 |
| Python output (38) | `14` / `30` | 0/3 / 0/3 | 890 / 298 | 42.8 / 127.9 | 937 / 344 |
| document (40) | `rack B7` / `rack B7` | 3/3 / 3/3 | 939 / 321 | 42.7 / 125.1 | 1010 / 390 |
| multi-turn (49), nonstream | `QN-482` / `QN-482` | 3/3 / 3/3 | unavailable | 43.8 / 159.0 | 1236 / 424 |

Both subset benchmark commands returned nonzero because the coding expectation failed. The model's `30` is incorrect for `sum(i*i for i in range(4))`, which is 14. Outputs were identical across these arms and repetitions for these cases, but this does not establish general parity or quality. Completion outputs were only 2–5 non-stop tokens at the native progress layer, too short for meaningful sustained accepted generation throughput; SSE fragments and the API rate convention are not substituted. GPU utilization and peak consumption were not sampled; no 160 tok/s claim.

The full serial smoke's tool request was HTTP 400 (`tools is not supported by the experimental resident engine`), so no tool round-trip occurred. Its client disconnect was followed by immediate 429 responses because the single active request was still draining. Server shutdown then logged `HETERO_REQUEST_DONE ... finish=cancelled`, zero misses and `HETERO_CLOSED`. This proves a cancellation observation and clean shutdown, not successful post-cancel reset/ACK or recovery. No cancellation was forced during incomplete bulk.

## Evidence and next steps

Restricted local evidence: `/home/chris/tmp/strata-heterogeneous/m7-0032/` (build/test logs, generated configs, CLI output, server logs and JSONL). Original smoke SHA-256 `6eab6834a68bfc6f838aec3af430431b4726b5cff61007c05ee94034fa42be4d`; controls subset SHA-256 `93b9a5763714c802919bf5f1baa2b3fca3a343f452e2349bb30266540ce3a883`. Frozen existing replay manifest SHA-256 `9142100d8380d5f32c28e44841f0c427669eb85171a06f5ec27309ce1d7d040c`; it was not run and is not an autonomous task benchmark. The unrelated untracked `tools/heterogeneous_pool_probe.cpp` remains untouched.

Next: investigate the unsupported tool path and safe disconnect/drain synchronization, then repeat full controls including verified next-request reset; complete the missing integrity and raw audits. Qualify actual near-cap contexts with resource checks, passive phase-aligned utilization and peak-memory sampling. Execute frozen replay and real scoped agent tasks with predeclared rubrics and matched serial/bulk budgets, report all failures, then profile before optimizing. Do not interpret the short control rates as real workload qualification.
