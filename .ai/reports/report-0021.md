# Report 0021 — stream-ordered MTP residual qualification

**Status: bounded single-relay trace-free proposal and separately traced committed-state gates PASS; multi-round admission remains blocked.** Starting commit `19dc42d1ee55c0510c54b2e4fdfa863d1e9dd73c`, branch `feature/heterogeneous-cuda-window-admission`. This iteration tested the report 0020 stream-ordered residual staging correction, without changing inference arithmetic. The code carried into this iteration computes a final diagnostic second draft for an explicitly requested one-relay run even without tracing; it is never submitted to the target. No GitHub Actions, downloads/repacking, fault injection, forced GPU kills, precision, context, reserve, residency, clock, power or driver changes. No performance, resident reset, bulk-prefill, rollback or production claim.

## Build, tests and asset admission

Using the original vanilla CMake caches/toolchain and checkout `.venv`, `cmake --build build-hetero-smoke-{host,cuda,hip} -j28` passed. Host `ctest --test-dir build-hetero-smoke-host --output-on-failure -j28`: **97/97**. `PYTHONPATH=tools .venv/bin/python -m unittest test_heterogeneous_mtp_reuse test_heterogeneous_mtp_front test_heterogeneous_mtp_state test_heterogeneous_mtp_admission test_heterogeneous_prefill_state test_heterogeneous_prefill_parity`: **45/45**. `PYTHONPATH=tools .venv/bin/python -m unittest serve.test_heterogeneous_resident`: **12/12**. `env -u PYTHONPATH .venv/bin/python -m pip check`: passed. Read-only two-shard asset admission with `--verify-raw` and existing CMake `gguf-py`: **PASS_ASSET_LAYOUT**. Approved Q2_0 target index SHA-256 `c344c01f59f547ed8a59003857a82eb90499e5299154241dbca787acb54b5b7c`; four runtime file hashes match report 0020 (`runtime-sha256.txt`). Exact target checkpoint revision is still unresolved.

All successful live runs used original configs and fingerprint, 2048 cells, max_t 8, reserve 536870912, full expert residency, CUDA0 [0,12) → HIP0 [12,30) → HIP1 [30,48), grouped HIP split overlap. No inherited STRATA/HIP/CUDA trace or performance flags, receiver arm, mapped egress, resident or bulk prefill. All 32 successful runs published completion markers and reported three fully resident expert route records with zero host misses. One-shot fresh runs explicitly reported `state_trace=0 front_trace=0 repeat_round=0 relay_rounds=0`; normal trace-free relay reported the same except `relay_rounds=1`. A normal supervised two-round launch refused worker preparation with `MTP relay draft-state reuse failed fresh-prefix qualification`, before native probe allocation; no completion marker. No stalls.

## Trace-free fresh-versus-reused gate

Independent fresh prefix was derived from the actual accepted relay manifests, never assumed or truncated. It was `[248045,846,198,2523,513,32767,264]` in every case. For each completed pair, the existing `tools/heterogeneous_mtp_reuse.py` validated actual target-selected relay anchor, greedy keep and accepted prefix; compared the final native draft cell/anchor/ordered token IDs; and required the existing 0.001 KL/top-1 gate plus **zero max absolute logit difference** for the seven accepted rows. All **12/12 pairs passed**:

| Prompt shape | Shared | Socket | Follow-up draft at cell 6, anchor 2724 |
|---|---:|---:|---|
| six one-row windows + one relay | 3/3 | 3/3 | `[579,23513,1830,364]` |
| original three keep2 four-row windows + one relay | 3/3 | 3/3 | `[579,23513,1830,364]` |

The live offer in every case was `[264,14075,8818,364,264]`; target kept anchor 264 only, rejected all four proposed draft tokens, and selected 2724 next. The follow-up candidate 579 was logged diagnostically and **not submitted**, so it is not evidence of actual draft acceptance. No native K/V, front or residual capture occurred in these trace-free trials; exact trace-free K/V identity is **unmeasured**.

All evaluated logits, including discarded candidate rows, were also compared across same-shape repetitions and transports using `read_run(allow_partial=True)` and `compare(...,0.001,prefill=True)`: each of six one-row relay runs matched its shared reference on **11/11 rows**, each of six three-window runs matched on **17/17 rows**. Every comparison passed with exact predictions and top-1, KL 0 and max absolute logit difference 0. Separate full-logit STFR and combined stdout/stderr exist for every process.

## Separate traced state/front gates

One independently launched paired run per shape per transport, with both `--mtp-state-trace --mtp-front-trace`, **4/4 PASS**. Fresh and relay had identical cell-6 input pair, residual R and committed K/V cells 0–5 before the follow-up, and identical R and committed K/V cells 0–6 after it. All 22 native front fields and source-before/source-after, token, step, position, rest records and selected-row staging checks matched exactly. Final diagnostic offers in all four pairs were `[579,23513,1830,364]`; seven accepted target rows per pair passed full-logit exact comparison. Traced checks are independent corroboration, not a claim that the trace-free runs' K/V was captured or that instrumentation cannot affect scheduling. No remaining divergence was observed, so no `STRATA_MTP_MULTI_COPY=0` ablation or additional broad search was justified.

## Admission decision and remaining seam

The requested gates demonstrate the correction on this fixed short rejection case, not arbitrary prefixes or multiple rounds. Considered a separate bounded three-round admission, but no actual accepted draft token has been observed in these qualified one-relay runs, and exact reused state after actual draft acceptance has not been established. A diagnostic relaxation would expose unqualified accepted-prefix/rollback behavior, so the native `>1` guard is retained rather than restored on the strength of this case. Before any relaxation, define a separately bounded, target-authoritative acceptance/rejection qualification on both transports with fresh-prefix proposal and committed-state checks; do not force predictions or treat causal overwrite as proof of rollback. No resident reset or bulk prefill was enabled.

## Artifacts

`/home/chris/tmp/strata-heterogeneous/m7-0021/` contains exact command JSON and separate process logs, STFR/manifest/completion files, 12 `*-reuse-gate.json` reports, four `trace-*-state-gate.json` reports with R/K/V and all front fields, `all-evaluated-parity.json`, test/build/admission logs and drivers. The driver ran pairs sequentially and stopped on any failed gate. The pre-existing untracked `tools/heterogeneous_pool_probe.cpp` was preserved.
