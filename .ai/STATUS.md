# Status

Turn holder: **Human User**: decisions below (hardening removal, RAM, the live-resume criterion).

## Current milestone

M4 reached with one open point (report 0069, `4785ba1b`):
- **Caches:** both work across the boundary. The front stage repeats checkpoints and parked conversations through
  vanilla's own save and restore calls, after checking that it holds the same tokens.
- **Turn 2 with and without reuse:** checkpoint and park paths match a fresh read (50/50, and 33/33 where the reply
  stops early). The live-session resume differs at token 7; vanilla HIP-only also differs on that path, at token 29.
- **Reuse:** the same counts as HIP-only.
- **Server checks:** all pass; the slow reader is within 0.3% of the fast one.
- **Long prompt:** prefill 2,249 and decode 113 tok/s against HIP-only 2,236 and 111.
- **RAM and VRAM:** one shared expert arena (memfd), but no extra free RAM: about 19 GiB available with the engine
  idle. The front stage no longer loads the head.
- **Review:** Chat AI reviewed the engine diff: vanilla calls only, behind `STRATA_HETERO`.
- **Open risk (from reading the code, unconfirmed):** the front stage's store of checkpoints and parked
  conversations is bounded by count, while the host's conversation cache evicts by bytes. The front stage can keep
  copies the host has dropped, costing RAM, and a mismatch in eviction order would turn into a failed restore, which
  vanilla treats as fatal for the engine.

## History in short

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

## Open decisions for the Human User

- **Remove the ported hardening?** The Human User now attributes the 17 November crash to a launch config that mixed
  components from different builds. Chat AI checked in a scratch copy:
  - All mixed commits, 0069 included (`33e5782c`, `0bb4225b`, `4455e9a2`, `ec8bb9c6`, `9655e74d`), replay without
    conflicts onto `3f65b5de` (vanilla plus the original mixed commit, before the hardening).
  - Two lines of `ec8bb9c6` (the hardening arena record) must be dropped.
  - The session-save calls 0069 uses have the same signatures in vanilla.
  - Options: A, rebuild and archive today's head as `archive/hardened-0069`; or B, twelve revert commits on top.
- **RAM:** whether to limit the host's expert arena to its own layers (new code, about 10 GiB of RAM).
- **Live-session resume:** whether to accept its token difference as vanilla behaviour, since HIP-only also differs.

## Next step

After the decisions: if the hardening goes, Chat AI rebuilds the branch and writes a short Tier 2 re-check handoff
(builds, hetero_tests, HIP-only and mixed serving, same-build check of the launch config). Then the front stage's
store is bounded and kept in step with the host's evictions, then M5.

Roadmap: https://claude.ai/artifact/NXCzvkbBJ2Z9fKts6ADRNn
