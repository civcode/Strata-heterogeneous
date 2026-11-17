# Status

Turn holder: **Coding Agent**, handoff `0069` (Tier 1) open once the Human User confirms its 30-minute budget.

## Current milestone

M4: the mixed profile serves like vanilla. That means caches across the boundary, one copy of the experts in RAM, and
the run configs and drivers committed (handoff 0069).

M2 reached (report 0068, `c002991d`): mixed requests run through vanilla's loop with CUDA [0,15) and HIP [15,33) +
[33,48). All five bring-up steps passed, including `serve/server.py` streaming, non-streaming, cancel and SIGTERM
shutdown. Long prompt (4,924 tokens), mixed against HIP-only: prefill 2,342 against 2,229 tok/s; decode 116 against
111 tok/s. One sample each. Greedy output differs from HIP-only at token 0; the front stage's layer-14 rows match a
CUDA-only run (cosine >= 0.9997), so this is a near-tie and a low-priority numerical finding. The engine changes use
vanilla options (`--no-prefill-borrow`, `--mmap-experts`, lookup windows 0), leave the PLE to the front stage, and fix
a stale arena record in the hardening port (all builds). Chat AI reviewed them and accepts them under the contract.
Kernel log: 17 amdgpu `hogged CPU` warnings, mostly during a RAM-starved run; no GPU faults, crashes or reboots.

## History in short

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

- None blocking. Recommended in parallel, outside this project: the host-integrity checks the incident report lists
  (memory test, EDAC/MCE review, storage health), because two PID 1 crashes (6 and 17 November) need an explanation
  of their own.

## Next step

After report 0069: Chat AI reviews the session-op binding against the contract. It then plans M5 (divergence list,
open refusals) and the remaining performance items: the front stage's 2048-token prompt chunks, and the CUDA/HIP
edge cost per window.

Roadmap: https://claude.ai/artifact/NXCzvkbBJ2Z9fKts6ADRNn
