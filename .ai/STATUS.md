# Status

Turn holder: **Coding Agent**, handoff `0070` (Tier 2) open.

## Current milestone

Re-check of the rebuilt branch (handoff 0070), then M5.

**2026-11-18:** the strata-dev-private hardening is removed (Human User's decision). `feature/mixed-native` was rebuilt
from `3f65b5de` (vanilla 0.2.15 plus the original mixed commit) and force-pushed. The hardened line is on
`archive/hardened-0069` (`b099f46f`); `.ai/HARDENING.md` maps old commits to new ones. Compared with vanilla, the
branch changes 5 engine files (`generate.cpp`, `verify.cpp/.hpp`, `prefill.cpp/.hpp`), the conversation-cache header
(one guarded hook) and the top-level `CMakeLists.txt`, plus `hetero/`.

Checked by Chat AI without a GPU:
- The mixed engine changes equal the hardened line's, except the two hardening arena lines and one vanilla-form
  allocation.
- `mixed_split_tests` and `mixed_split_cache_drop_test` pass.
- The engine files syntax-check with only vanilla's own build-define errors.
- Reports 0067 to 0069 were measured with the hardening in.

New: the front stage drops a parked copy when the host's conversation cache lets it go (`0b0f671b`, unverified on the
cards). That closes the count-against-bytes risk of report 0069.

## History in short

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

## Open decisions for the Human User

- Handoff 0070's 15-minute GPU budget (above the 5-minute default).

## Next step

After report 0070: M5. That means a short divergence list (the seam, the guarded hooks, the options mixed mode still
refuses, and why), then the remaining performance items (the front stage's 2048-token prompt chunks, the edge cost per
window).

Roadmap: https://claude.ai/artifact/NXCzvkbBJ2Z9fKts6ADRNn
