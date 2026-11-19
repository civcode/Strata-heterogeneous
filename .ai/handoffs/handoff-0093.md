# Handoff 0093

## Tier
1

Why Tier 1: the mixed split's pipelined decode collapses on the Human User's IQ3_XXS setup (23 tok/s, against about
101 tok/s serial with the same build and model, and 120 tok/s vanilla HIP-only pipelined), while it gained 7 to 17%
on Q2_0 in report 0077. Where the time goes is only visible on the cards, and the fix depends on it.

## Objective
Find why `--pipeline-windows 2` makes mixed decode about 4x slower than the serial loop on Qwen3.8-Flash-Next IQ3_XXS,
and fix it inside the mixed split's code, so that mixed pipelined decode is at least as fast as mixed serial, and
ideally above vanilla HIP-only pipelined (120 tok/s in the Human User's agent work).

## Context
- Read `.ai/PROJECT.md`, `.ai/DIVERGENCE.md` (pipelined windows, report 0077: the front stage mirrors stage 0's two
  verifiers, GDN snapshots, undo and replay; the asynchronous front window; prompt chunks across the link), report
  0077 and report 0092.
- **The configs** (in this checkout, written by `make_mixed_config.py` from `~/tmp/Strata/strata-iq3_xxs.json`,
  build `build-0092-*`): `strata-iq3_xxs-mixed-maxperf.json` (`--pipeline-windows 2`, `STRATA_MTP_STAGE0=0`,
  `STRATA_PIPELINE_AUTO=0`, 256K context, `--kv-resident 0`, `--mmap-experts`, IQ3_XXS native pack). The Human
  User's serial config is the same with `--pipeline-windows 0` (about 101-106 tok/s; report 0092: 114 short, 101 at
  9K). The vanilla reference: `~/tmp/Strata/strata-iq3_xxs.json` on its own engine (120 tok/s median, agent work).
- **From the Human User's log (22:39):** pipelining is on (two verifiers per stage on the host, two on the front
  stage, GDN snapshots 2 x 37.4 MiB); the engine's note that the draft layer is on the last card; split K=31 (HIP0
  15-30, HIP1 31-47 + head + draft layer); front expert cache 7510 slots (10.47 GiB), HIP0 8192, HIP1 8704; draft
  acceptance about 58% (217 of 377), so most windows' next-window guess misses; the front stage turned
  `STRATA_ROUTE_TAIL_SKIP=7` on (its 12-16 GB card default) while the host turned it off.
- GPU usage in the serial loop (Human User): CUDA 23%, HIP0 51%, HIP1 96%.

## Tasks
1. Safety checks; the first rule; the start/stop rule (`.ai/PROJECT.md`).
2. **Measure** with `STRATA_SPLIT_TIMING=1`, on a fixed request set (a short and a long agent-like prompt, greedy,
   enough decode tokens for a stable median), each configuration started fresh:
   mixed serial, mixed pipelined (`STRATA_MTP_STAGE0=0`), mixed pipelined with the draft layer on the first card
   (`STRATA_MTP_STAGE0=1`), and the same three on Q2_0 (the report 0077 mirrors) as the known-good comparison.
   Per configuration: decode tok/s, draft acceptance, pipelined-guess hit and miss counts, and per window the time of
   each stage, the front stage's window, the link calls (PlRun, PlCommit, PlUndo), and the host's waits.
3. **Explain** the collapse from those numbers: the cost of a miss in mixed against vanilla's, and where it is spent.
   Also check `STRATA_ROUTE_TAIL_SKIP`: whether the front stage and the host should agree (the host's setting
   passed to the front stage), and its effect on acceptance.
4. **Fix** inside `mixed_split/` and the existing `#if defined(STRATA_MIXED_SPLIT)` blocks (the implementation
   contract: no new scheduling or speculation logic in vanilla's code; the mixed split mirrors vanilla's pipeline).
   Add tests where the fix has CPU-testable logic. `check_off_is_vanilla.py` stays 6 of 6.
5. **Verify:** pipelined against serial exactness on the code prompt with `STRATA_IQ_MT_MIN=1` (as in report 0077),
   and the decode table of task 2 again for the fixed build, IQ3_XXS and Q2_0.

## Hardware Test Budget
**About 1.5 hours of GPU run time, over the 30-minute allowance** (announced to the Human User; allowed unless refused).
Runs under 10 minutes each; contexts up to 32K.

## Constraints
- **First rule: `HSA_ENABLE_SDMA=0` in every run.** All rules of `.ai/PROJECT.md`.
- The Human User's run files in this checkout (`strata-iq3_xxs-mixed*.json`, `run-*.sh`) are theirs: copy them for
  experiments, do not edit them.
- Stop and report instead of proceeding if the fix needs a change to vanilla's pipeline logic outside the seam, or a
  link protocol change beyond adding messages.

## Completion Criteria
- The cause explained with measurements; a fix committed, or a reasoned report why a fix inside the boundaries is not
  possible.
- Exactness check passes; the before/after decode table for IQ3_XXS and Q2_0.
- `.ai/reports/report-0093.md` and all changes committed and pushed to `feature/mixed-native`.

## Reporting
- The task 2 table (before) and task 5 table (after); the explanation; the change and its tests; design decisions
  that matter later; safety lines, swap, GPU time.
