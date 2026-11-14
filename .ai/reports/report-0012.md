# Report 0012 — native prefill parity localization

**Status: PARTIAL / parity FAIL.** Handoff `2d607ad86f3c5a39114638e92567765613d74f1d`; diagnostic integration tree `28063cb3d5c374b32efede1e6951c3b7d9aa8758`. No interoperability correctness fix or independent vanilla defect was established. Native prefill remains opt-in, unqualified and off by default. Root cause and full batch completion remain **UNVERIFIED**. Raw logs, token sequences, capture files and comparison reports: `/home/chris/tmp/strata-heterogeneous/m7-0012/`.

## Build and diagnostic fixes

- Fixed missing `argc` in `--prefill-parity-at` argument parsing, which initially failed the integrated host compile. For this diagnostic only, also capture independent verifier CUDA stage-0 output and physical HIP stage-1 input under a separate `.verifier` prefix. Normal runs still clear inherited capture settings; no inference math, precision, budgets or thresholds changed.
- Host, CUDA worker/coordinator and HIP worker targets built with existing cache/toolchain and `-j28`. Host CTest **94/94 PASS**; required `unittest discover -s tools -p 'test_heterogeneous_prefill_parity.py'` **9/9 PASS**; targeted resident and parity Python suite **36/36 PASS**. Checkout `.venv` pip check passed with inherited ROS `PYTHONPATH` removed. Initial diagnostic attempt was correctly rejected because its log file shared the fresh capture prefix; subsequent captures used a dedicated directory.
- Full resident original Q2_0 12/18/18 plan, CUDA `04:00.0`, HIP0 `03:00.0`, HIP1 `07:00.0`, original context 2048/max_t 8, pack-index SHA-256 `c344c01f59f547ed8a59003857a82eb90499e5299154241dbca787acb54b5b7c`, fingerprint 2026111301. Each A/B/A was a fresh request/reset in two grouped worker processes. All runs finished normally; route logs report 6144/9216/9216 slots and zero host misses. No GPU reset, altered precision, clock, power, reserve or arena.

## Teacher-forced results

All rows use the fixed serial input IDs, not native predictions. Serial A versus serial B was **bit-identical logits/top-1** in every listed run. Existing threshold KL <=0.001 and exact prediction/top-1 were not relaxed. Each native prompt piece had independent CUDA output and physical HIP0 input device readback; exact byte comparison passed as shown. That excludes corruption for these pieces only, not incorrect source values or device state.

| Prompt / cut / chunk | physical R pieces equal | serial vs native result |
| --- | --- | --- |
| A02 / 759 / 32 | 24/24 | position 759: 8160 vs 760, KL 0.023444, FAIL |
| A02 / 759 / 1 | 759/759 | position 759: 8160 vs 760, KL 0.032818, FAIL |
| A03 / 429 / 32 | 14/14 | eight forced positions 429–436: max KL 0.029264; position 435 top-1 494 vs 11, FAIL |
| A03 / 429 / 1 | 429/429 | eight forced positions 429–436: max KL 0.036182; position 435 top-1 494 vs 11, FAIL |

For A03 chunk 32, per-position KL at 429–436 was 0.029264, 0.000518, 0.000479, 0.003611, 0.006802, 0.002250, 0.003244, 0.000455. For chunk 1 it was 0.036182, 0.000170, 0.000824, 0.002518, 0.003406, 0.001476, 0.001743, 0.000251. Refer to `*-report.json` for exact unrounded per-row metrics.

## Earliest bounded localization

A02 cuts 1, 2, 4, 8, 16, 64, 256 and 512 were tested (chunk 1 at 2/4/8/16; chunk 32 at 1/16/64/256/512). Cut 1 at position 1 **PASS** (same top-1 198 and KL 0). Cut 2 at position 2 **FAIL** (same top-1 96220 but KL 0.001896 >0.001); this is the earliest failing threshold among possible positive cuts, not necessarily the first non-bit-identical intermediate. KL is not monotone with position: cut 4 passed 0.000210, cut 8 failed 0.001062, cut 16 failed 0.033692 (chunk 1). Long cut 256 failed 0.001266 and cut 512 failed 0.002338 (chunk 32).

A separate A03 cut-2/chunk-1 capture uses the same first tokens and confirms KL 0.001896. **Before HIP executes the target position**, the actual CUDA verifier's stage-0 output for teacher-forced position 2 differs between serial-prefix and native-prefix modes, while the two serial repeats are byte-identical. On its identically formatted `R | bo | inject` output, native-versus-serial max absolute differences are R 0.000809, bo 0.000522 and inject 0.039551. All seven observed verifier CUDA-output/HIP0-input pairs, including that position, match exactly (51,216 bytes each). Native prefix physical R-only pieces also match CUDA-to-HIP, 2/2. This localizes an observed effect to CUDA stage execution/state *before the cross-vendor handoff*; it does not distinguish a native Prefill arithmetic difference from incorrectly initialized or adopted CUDA QSA/GDN/PLE state, nor prove that vanilla has the same discrepancy. Raw native-prefix R-only and verifier `R | bo | inject` are different representations and must not be treated as directly equivalent tensors.

The exact corrupt state/layer and root cause are **not yet identified**. A same-runtime vanilla Prefill-versus-Verifier reproducer was not run, and the physical readback cannot prove semantic correctness of the source R. Further work must compare CUDA stage-owned PLE previous-token/history and QSA/GDN snapshots immediately after the same short prefix, then compare the first differing same-runtime layer using native taps; isolate any external-boundary/ownership behavior before attributing a defect to vanilla. Do not change native arithmetic outside the authorized interoperability scope.

## Remaining qualification

Because parity still fails, no corrected-path replay, post-fix cancellation or default enablement is claimed. Report 0011's earlier bounded drained-prefix cancellation is historical only. Independent native-vanilla parity, KL/state localization, a safe fix and post-fix A/B/A plus reset/cancel remain outstanding. MTP remains disabled and outside this batch. No upstream writes or Actions. The pre-existing untracked `tools/heterogeneous_pool_probe.cpp` was left untouched.
