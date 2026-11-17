# Status

Turn holder: **Coding Agent**, handoff `0068` (Tier 1) open.

## Current milestone

M2: mixed generation on vanilla's loop, CUDA [0,15) + HIP [15,33) + HIP [33,48) (handoff 0068).

Hardening gate passed (report 0067): the ported hardening builds in all three trees with no source fixes; host CTest
225 passed, 6 skipped; HIP GPU tests 95 passed; CUDA 17 passed, 3 skipped (need two CUDA cards); readback reproducer
passes; HIP-only serves with conversation-cache reuse and shuts down cleanly; no new kernel GPU faults or crashes.
HIP-only with MMQ (different prompts from 0065): decode 115 to 122 tok/s, prefill 2831 tok/s at 6.4K tokens.

The mixed-engine work is re-applied (`584e8e62`). The front stage failure of report 0066 is traced and fixed in source
(`33e5782c`, unverified): two whole-model graph captures ignored the mixed split's layer range.

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

After report 0068: Chat AI reviews the engine diff against the contract and plans the next step (caches across the
boundary, or the CUDA/HIP edge cost, depending on what 0068 reaches).

Roadmap: https://claude.ai/artifact/NXCzvkbBJ2Z9fKts6ADRNn
