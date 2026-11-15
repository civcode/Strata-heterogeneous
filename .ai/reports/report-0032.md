# 0032 — real workload qualification (partial)

**Status: NOT QUALIFIED.** Initial source tested: `02893b678f6419da4185151126e384ce9e90ef54` on `feature/heterogeneous-real-workloads`. The follow-up below adds tool support and longer-context measurements on source `6648eda744cf5173c1516a89b87ddca7ce4e197e`. Execution remains partial, not completion of the handoff. Ordinary 2K-capacity serial and native-bulk HTTP inference worked on short controls, but coding quality failed and tool requests were refused. Long context, agentic tasks, utilization, peak memory and production readiness are not established. The unchanged numerical diagnostics in reports 0027.15 and 0030 remain FAIL; no thresholds were changed.

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

The original next steps are partly addressed by the follow-up below. The remaining integrity audits, real autonomous tasks and qualification gates remain open.

## Follow-up: tools, cancellation and longer contexts

Tested source `6648eda744cf5173c1516a89b87ddca7ce4e197e`; the only implementation change permits tool declarations as Service-rendered prompt metadata in the heterogeneous adapter. Service continues to render and parse calls; native inference and integrity checks are unchanged. Forced tool choices and legacy `functions` remain rejected. CPU tests: resident 15/15, heterogeneous tool discovery 318/318, and `serve.test_server` 376/376. The builds and binary identities above are unchanged.

The fixed `lookup_order` client fixture completed a real two-stage call and returned `shipped` **3/3** serial and **3/3** bulk, separately in nonstream and SSE streaming modes. No model-generated tool was executed. This is one fixture, not general tool-use qualification. Serial disconnect test: arithmetic request 7 returned `95`; request 8 was logged `finish=cancelled`; after a two-second drain, request 9 returned `95`, with the same resident coordinator and normal close. This is evidence of recovery for that case, but independent reset/ACK tracing and a fresh-process post-cancel comparison were not performed. Bulk cancellation remains untested.

### Real source and replay observations

All modes use the same original model, vanilla precision and 12/18/18 full-residency plan. Serial uses prompt chunk 1; bulk uses native prefill chunk 128. Per-tier configs differ from the originals only in `max_cells`, with max_t 8, original slots and reserve unchanged. 8K/16K/32K server limits used 1024 max new tokens. Source-bound config preflights and passive readiness inventories passed; the inventory explicitly omits native/session/staging allocations, so it is not a complete prior memory proof. Runtime guards remained enabled. Logs show full residency, zero host misses, subsequent short requests and clean `HETERO_CLOSED` for each observed arm. No spill or fallback was reported. Each tier was exercised after shorter requests; no forced worker kills or device setting changes.

Rates below are host commit-progress proxies (reset and IPC included in prefill; decode includes consumer pacing), not GPU kernel timing. TTFT is client first visible text or tool fragment, not startup. Times are medians when `n=3`; 16K and 32K have only one observation per mode. The 8K replay and host-review runs have no separate warmup exclusion; the 2K review has one disclosed warmup and three measured requests. Arms ran serial then bulk, not balanced across independent launches. Raw task outputs remain local. The replay is scripted history, not an autonomous agent execution; generated tool calls were not executed.

| Capacity / workload | Prompt tokens | Mode | n | TTFT s | Prompt tok/s | Decode feedback tok/s | Result / quality |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 2K / `heterogeneous_contract.py` source review | 1,378 | serial | 3 | 31.88 | 43.2 | 43.6 | natural stop; answer covers requested checks and example but incorrectly states the 384 KiB cap as 384,024 bytes |
| 2K / same review | 1,378 | bulk | 3 | 2.04 | 674.8 | 43.9 | all hit 512-token cap before completing the example; truncated, not a quality pass |
| 8K / frozen `expert-cache.0` replay | 2,163 | serial | 3 | 49.94 | 43.4 | 43.2 | tool calls; unscored; identical call arguments within mode |
| 8K / same replay | 2,163 | bulk | 3 | 3.51 | 641.1 | 43.4 | tool calls; unscored; path differs from serial |
| 8K / full `host.cpp` review | 6,586 | serial | 3 | 152.38 | 43.2 | 43.2 | all hit 512-token cap; truncated and not a quality pass |
| 8K / same review | 6,586 | bulk | 3 | 9.44 | 697.9 | 43.3 | all hit 512-token cap; truncated and not a quality pass |
| 16K / frozen `expert-cache.1` replay | 8,163 | serial | 1 | 188.17 | 43.4 | 43.3 | tool call; unscored |
| 16K / same replay | 8,163 | bulk | 1 | 12.29 | 664.9 | 43.5 | different tool call; unscored |
| 32K / frozen `expert-cache.2` replay | 20,748 | serial | 1 | 480.69 | 43.2 | 43.2 | tool call; unscored |
| 32K / same replay | 20,748 | bulk | 1 | 32.16 | 645.6 | 42.1 | different tool call; unscored |

The 2K serial answer and bulk answer differ in text and termination; their quality cannot be called equivalent. All 8K host-review responses were truncated. The original Python arithmetic/code control still fails in both modes. The 32K-capacity run exercises the extended native prefill wire beyond 8K, but is not a near-32K test or proof of complete protocol coverage. 16K/32K were not repeated or tested near their limits. No supported same-model vanilla comparison was run.

### Phase-aligned passive telemetry

250-ms samples were aligned by per-device sample timestamp with native prefill/decode intervals. Values are per-device mean reported busy and **maximum sampled** used MiB, not occupancy, guaranteed simultaneous work, or instantaneous peak. The 8K host-review rows cover three repeated requests; 16K and 32K each cover one replay. Device order is CUDA0 / HIP0 / HIP1. Memory includes other device allocations, and prefill and decode maxima may occur at different instants.

| Workload / mode | Prefill samples per device | Prefill mean busy % | Prefill max used MiB | Decode samples per device | Decode mean busy % |
| --- | ---: | --- | --- | ---: | --- |
| 8K host review / serial | 1,826 | 10.2 / 26.7 / 34.4 | 11,357 / 16,578 / 16,671 | 142 | 10.2 / 26.8 / 34.4 |
| 8K host review / bulk | 117 | 44.2 / 86.6 / 93.1 | 11,653 / 16,917 / 17,014 | 141 | 10.0 / 27.2 / 34.9 |
| 16K replay / serial | 753 | 10.0 / 27.0 / 34.5 | 11,411 / 16,648 / 16,757 | 6 | 9.0 / 27.2 / 34.8 |
| 16K replay / bulk | 49 | 40.2 / 88.0 / 95.9 | 11,707 / 16,991 / 17,102 | 7 | 5.1 / 24.4 / 40.9 |
| 32K replay / serial | 1,922 | 10.2 / 26.9 / 34.5 | 11,517 / 16,788 / 16,931 | 6 | 10.0 / 27.0 / 34.7 |
| 32K replay / bulk | 129 | 52.6 / 89.9 / 97.5 | 11,817 / 17,135 / 17,280 | 8 | 5.0 / 26.4 / 35.5 |

These indicate short decode intervals and low sampled CUDA activity in these runs; they do not by themselves identify the performance bottleneck. The 160 accepted tok/s real-agent target remains unproven. No performance optimization was made.

### Follow-up evidence and remaining work

Private evidence is under `/home/chris/tmp/strata-heterogeneous/m7-0032-followup/`, including generated configs, readiness inventories, server logs, benchmark JSONL and telemetry. The 2K review manifest SHA-256 is `2db6ea7bbb3bc1d44da5a3ffd23d25ab73fb06a52567b61f2b19d25f3ed556c0`; 8K host review `bbb83b883d8f0a20fd8057d2dab9c158511150ed67af2ed820b2a8d17fcb2c79`. Frozen-replay subset hashes are `b5e178a7b90eed94d191d3100be2ddbee408d4a0a42efcd2aeb42a65a9190fe9` (8K), `243d63b631cf8d9eadd07ac79570594530609827cf10f8a21e236a35e8b7a1bf` (16K), and `8b54ddd122f4e88f4bf78f3a470d8fbd8256a757d24dddb8fcf809b0c786c938` (32K). They are unchanged selected cases from the previously frozen manifest, not a new autonomous benchmark. Config hashes and complete samples are in the private evidence. No inference worker was left running after these tests; the unrelated untracked probe remains untouched.

Still NOT_RUN or unverified: full no-run negative matrix and explicit 16 MiB/wire CPU audit, original raw asset audit, MTP controls, default capture invariance, bridge same-mode and fresh/resident A/B/A, reset ACK proof after cancellation, bulk cancellation, repeated near-cap 16K/32K tests, full replay, 12-task autonomous coding qualification, representative quality/perplexity and matched vanilla reference. No production admission; previous numerical failures remain unchanged. Next work should complete these checks and investigate incorrect/truncated task responses before claiming quality or optimizing decode throughput.
