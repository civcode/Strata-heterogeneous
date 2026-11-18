# Handoff 0073

## Tier
2

A measurement turn, with no engine change. Scope the agent may change: run configs, the drivers in `hetero/tools/`,
and stderr logging in the front stage's backend if a number cannot be read otherwise.

## Objective
Measure whether pipelined windows are worth building for the mixed split. Get vanilla's gain from
`--pipeline-windows 2` on this rig, and the per-window time of each stage and of the CUDA/HIP link in a mixed run.

## Context
- The base is stable: the re-check of the rebuilt branch is complete (reports 0070 to 0072). With
  `STRATA_MIXED_SPLIT` off, the engine sources are vanilla's line for line (`hetero/tools/check_off_is_vanilla.py`,
  `.ai/DIVERGENCE.md`).
- **Since report 0072,** Chat AI moved every unguarded mixed-split line into `#if defined(STRATA_MIXED_SPLIT)`
  (`7f4f8ca6`). With the option on, the only behaviour change is that a refused park is now reported to the front
  stage by the cache's caller. That is unverified on the cards, so rebuild first. The builds of report 0071 are stale
  for `check_launch.py`.
- **Pipelined windows,** vanilla `--pipeline-windows 2` (`docs/MULTI_GPU.md`):
  - The first card starts the next window as a guess while the last card checks the current one.
  - Vanilla measured +21% on 2 R9700 and +19% on 3 R9700.
  - On gfx12 it is on by default for a layer split when the draft layer can sit on the first card; the log says
    "is on by default".
  - Mixed mode refuses it: the front link carries one window at a time.
- **The Human User's observation:** `--pipeline-windows 2` raised decode tok/s on the dual R9700.
- **Timing traces** in vanilla:
  - `STRATA_SPLIT_TIMING=1` prints, per stage and window: GPU wait, pool and plan, host staging, commit.
  - `STRATA_DECODE_TIMING=1` prints one line per request.
  - The front stage's process inherits the environment.
  - `.ai/PROJECT.md`: never compare a traced run with an untraced one. Take tok/s from untraced runs and the
    breakdown from traced runs.

## Tasks
1. **Kernel log and core dumps:** record them before the first GPU run and after the last.
2. **Rebuild** the HIP and CUDA trees at the current commit, with `-DSTRATA_MIXED_SPLIT=ON`, and stamp them. Run
   `mixed_split_tests`, `mixed_split_cache_drop_test` and `check_off_is_vanilla.py`.
3. **HIP-only A/B, untraced.**
   - Setup: the committed HIP-only config, vision on, as committed. Two prompts: a short coding prompt with about 300
     tokens of reply, and the 4,924-token long prompt with 200 tokens of reply.
   - Arms: `--pipeline-windows 0` and `--pipeline-windows 2`, three samples each, alternating.
   - Record decode tok/s, drafts accepted, and which value the engine picks with the option absent (its log line).
4. **Mixed baseline, untraced:** the same two prompts, three samples each. Record decode tok/s and drafts accepted.
5. **Mixed breakdown, traced:** one run of each prompt with `STRATA_SPLIT_TIMING=1 STRATA_DECODE_TIMING=1`, on the
   host and the front stage. Then derive per decode window:
   - the front stage's own GPU and host time;
   - the host's time waiting for the front stage (its first local stage, from the front link call until the rows are
     there);
   - the link overhead, which is the host's wait minus the front stage's own time;
   - each HIP stage's time, and the drafter's time.

   If one of these cannot be read from the existing traces, add a stderr line in the front backend or `HeteroHost`
   behind `STRATA_SPLIT_TIMING` (logging only), and say so.

## Hardware Test Budget
15 minutes of GPU run time (within the project's 30-minute allowance). Every single run under 2 minutes.

## Constraints
- All rules of `.ai/PROJECT.md`. Run evidence goes under `.ai/runs/0073-*`. Never kill a GPU process by force. No two
  GPU jobs at once, except the two processes of one mixed run.
- A new GPU fault line, a crash outside the run or a reboot ends the handoff at once.

## Completion Criteria
- Builds, the CPU tests and `check_off_is_vanilla.py` pass.
- The HIP-only A/B has three samples per arm and prompt.
- The mixed baseline and both traced runs are done.
- `.ai/reports/report-0073.md` and all changes are committed and pushed to `feature/mixed-native`.

## Reporting
- **Tables:** the HIP-only A/B (median and spread per arm and prompt); the mixed baseline beside it; the per-window
  breakdown of the mixed run, in ms.
- **One line:** what share of a mixed decode window the front stage plus the link take, and what the longest single
  stage takes. That bounds what overlapping could save.
- Kernel log and core-dump lines, or "none". GPU run time used, counted per run. What remains open.
