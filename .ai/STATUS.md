# Status

Turn holder: **Coding Agent**, handoff `0072` (Tier 2) open.

## Current milestone

Re-check of the rebuilt branch, last part (handoff 0072), then M5.

Report 0071 was Partial: the budget ran out in the test suites (`d8861b91`). Chat AI's review:
- **Builds:** all five pass, including both vision helpers built from this checkout. The committed HIP-only config
  passes the launch check.
- **HIP tests:** 272 passed, 15 skipped, 2 failed out of 289. One failure is the known vanilla
  `conversation_copy_streams_test`; the other is `ple_parity`, which did not find its model file (a test input). No
  mixed-split defect.
- **CUDA tests:** 170 passed, 2 skipped (Windows-only), out of 172 run; the rest were not run.
- **Kernel log:** two `Cannot map memory` lines of 0x10 pages appeared during the CUDA tests, cause not established.
  No faults, crashes or reboots.
- **Why the budget ran out:** the handoff asked for the whole GPU test set first, which took most of it. Handoff 0072
  puts serving and the prefill A/B first.

## History in short

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

## Open decisions for the Human User

- None.

## Next step

After report 0070: M5. That means a short divergence list (the seam, the guarded hooks, the options mixed mode still
refuses, and why), then the remaining performance items (the front stage's 2048-token prompt chunks, the edge cost per
window).

Roadmap: https://claude.ai/artifact/NXCzvkbBJ2Z9fKts6ADRNn
