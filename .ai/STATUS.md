# Status

Turn holder: **Coding Agent**, handoff `0067` (Tier 2) open.

## Current milestone

Hardening gate: the engine hardening from `civcode/strata-dev-private` is ported onto vanilla 0.2.15
(`.ai/HARDENING.md`) and must build, pass its GPU tests and serve HIP-only before mixed-engine work resumes.

Why: on 2026-11-17 Oberon had a host crash (native fault in `libhipblaslt`, PID 1 fault, reboot) on a vanilla 0.2.15
build. Cause unknown. The Human User ran the patched private branch for days without such a failure and asked for the
hardening first.

## History in short

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

- None blocking. Recommended in parallel, outside this project: the host-integrity checks the incident report lists
  (memory test, EDAC/MCE review, storage health), because two PID 1 crashes (6 and 17 November) need an explanation
  of their own.

## Next step

After report 0067: Chat AI re-applies the parked mixed-engine work on top of the hardening, reviews it against the
contract, and writes the next Tier 1 handoff for the front stage bring-up.

Roadmap: https://claude.ai/artifact/NXCzvkbBJ2Z9fKts6ADRNn
