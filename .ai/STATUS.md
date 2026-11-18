# Status

Turn holder: **Coding Agent**, handoff `0071` (Tier 2) open.

## Current milestone

Re-check of the rebuilt branch, continued (handoff 0071), then M5.

Report 0070 was Partial (`4f30546e`). Chat AI's review:
- **Passed on the rebuilt branch:** clean builds in all three trees; host CTest 138 passed, 6 skipped; the mixed
  serving checks; turn 2 equal to a fresh read (50/50, checkpoint and park); the eviction run (A rereads after
  eviction, D reuses, the front stage logs each Drop). The only engine change is that Drop log line.
- **HIP `conversation_copy_streams_test` failed (148 MiB against 16 MiB):** vanilla's test on unchanged vanilla code.
  The hardening had rewritten that assertion. Recorded as a vanilla finding (upstream candidate: the first HIP copy
  stream on gfx1201 needs more than vanilla's 64 MiB restore guard). Not a mixed-split defect; no Tier 1 needed.
- **HIP-only config refused by the launch check:** correct, since its vision helpers came from the vanilla checkout.
  They are now built from this checkout (`tools/vision`).
- **Launch check staleness:** a commit that changes only notes or reports no longer makes the builds stale.
- **Mixed prefill:** 1,787 against 2,249 tok/s in report 0069, one sample each. Handoff 0071 measures the rebuilt
  line against the hardened line, three samples each.
- **Kernel line "Cannot map memory … 0x7e9152 pages":** 31.6 GiB, the shared arena's whole-arena HIP registration
  that falls back to 48 slices. Not a fault.

Since report 0070, by Chat AI:
- the build option is renamed to `STRATA_MIXED_SPLIT`, with its define, tests option, targets and tests;
- the run logs moved to `.ai/runs/`.

## History in short

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

## Open decisions for the Human User

- Handoff 0071's 15-minute GPU budget (above the 5-minute default). Until it is recorded here, the agent uses 5.

## Next step

After report 0070: M5. That means a short divergence list (the seam, the guarded hooks, the options mixed mode still
refuses, and why), then the remaining performance items (the front stage's 2048-token prompt chunks, the edge cost per
window).

Roadmap: https://claude.ai/artifact/NXCzvkbBJ2Z9fKts6ADRNn
