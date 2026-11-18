# Status

Turn holder: **Coding Agent**, handoff `0073` (Tier 2) open.

## Current milestone

M5 reached (Chat AI, `7f4f8ca6`): with `STRATA_MIXED_SPLIT` off, the engine sources are vanilla 0.2.15 line for line.
`hetero/tools/check_off_is_vanilla.py` checks this (6 of 6 changed files), and `.ai/DIVERGENCE.md` lists the seam, the
options mixed mode refuses and the vanilla findings. The branch adds 713 lines to 7 vanilla files and deletes none.

Next: pipelined windows for the mixed split (Human User, 2026-11-18: after the base is stable). Handoff 0073 measures
first: vanilla's `--pipeline-windows 2` gain on this rig, and the per-window time of each stage and of the link in a
mixed run.

The re-check of the rebuilt branch is complete (report 0072, `114affa0`). Chat AI's review:
- **HIP-only serving:** through the launch check, with vision helpers from this checkout. Reuse of 140 tokens; the
  image request works.
- **Prefill A/B, current against hardened line, three samples each:** median 2,260.8 against 2,251.5 tok/s, decode
  113.7 against 113.8. The hardening removal costs nothing; the 1,787 sample of report 0070 was an outlier.
- **Open test items, both vanilla:** `ple_parity` needs captured fixtures this repository does not have, and
  `cuda_spill_guard` has no sm_89 baseline. Our branch changes no GPU kernel source. Both are in `.ai/DIVERGENCE.md`
  as upstream candidates; no Tier 1 needed.
- **Since report 0072,** by Chat AI: every mixed-split line in a vanilla file is guarded (a refused park is now
  reported by the cache's caller). Syntax-checked and CPU-tested only; handoff 0073 rebuilds first.

## History in short

- Report 0072: HIP-only serving with vision; the hardening removal has no prefill or decode cost (A/B, 3 samples each).

- Report 0071: GPU test suites on the rebuilt branch (HIP 272/289, CUDA 170/172 run), no mixed-split defect.

- 2026-11-18: hardening removed, branch rebuilt; front-stage Drop added; report 0070 re-check Partial (one vanilla HIP test).

- M4 reached on the hardened line (report 0069): caches across the boundary, server checks, configs committed.

- M2 reached (report 0068): mixed generation on vanilla's loop; long prompt 2,342 / 116 tok/s.

- Hardening gate passed (report 0067): all builds, HIP GPU tests 95/95, CUDA 17 passed and 3 skipped,
  reproducer passes, HIP-only serves with cache reuse.
- M1 reached (report 0065): HIP-only is vanilla 0.2.15. Baseline (short samples): decode 85 to 96 tok/s, prefill
  2283 tok/s at 5.4K tokens without MMQ; with MMQ 2641 tok/s at 5.0K (report 0066).
- Shared hand-off probe: one shared mapping works for CUDA and HIP (reports 0063, 0064).
- Handoff 0066 Part B: the snapshot readback reproducer passed on unpatched 0.2.15 in isolation (not proof the bug is
  gone). Part A: unfinished; the front stage failed in session graph setup (`fused_gr_read: invalid arguments`).
  That work is parked on branch `wip/mixed-front-0066` and reverted on this branch.

## Decisions taken by the Human User

- 2026-11-16: reference vanilla 0.2.15; fresh branch; old branch frozen; probes not carried; HIP hosts the loop;
  agent tiers; 30 to 60 minutes GPU time for the Tier 1 bring-up.
- 2026-11-17: integrate the private hardening before continuing the mixed engine.
- 2026-11-18: remove the hardening (rebuild, force-push); defer the host's arena trim; accept the live-session resume
  difference as vanilla behaviour.
- 2026-11-18: rename the build option to `STRATA_MIXED_SPLIT`, with related defines, tests and targets.
- 2026-11-18: a standing GPU allowance of 30 minutes per handoff; a larger budget is announced and allowed unless
  the Human User refuses it.
- 2026-11-18: pipelined windows for the mixed split, once the base functionality is stable.

## Open decisions for the Human User

- None.

## Next step

After report 0073: if the front stage plus the link leave enough to hide, Chat AI writes the Tier 1 handoff for
pipelined windows. In it the link carries a second, guessed window; the front stage uses vanilla's second verifier and
state snapshot; and the draft layer's place is decided.

Roadmap: https://claude.ai/artifact/NXCzvkbBJ2Z9fKts6ADRNn
