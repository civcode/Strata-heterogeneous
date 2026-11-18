# Status

Turn holder: **Human User**: decide how to continue with pipelined windows (see Open decisions).

## Current milestone

M5 reached. Pipelined windows for the mixed split are under evaluation.

Report 0073 was Partial (`a15fe83d`). Chat AI's review:
- **HIP-only, `--pipeline-windows` 0 against 2,** untraced, median of three: coding 92.6 against 97.9 tok/s (+5.7%);
  long 101.2 against 105.3 (+4.1%). Arm order 0, 2, 2, 0, 0, 2. The option is on by default for this split.
- **Mixed, untraced:** coding 102.1 tok/s, long 108.8. The HIP-only config has vision and a control vector and the
  mixed one has neither, so this is not a controlled comparison. Direct engine runs without those (report 0072):
  mixed 113.7 against HIP-only 111.3.
- **Traced mixed decode window:** 17.5 ms (coding) and 20.4 ms (long), of which verify is 15.2 and 17.6 ms. The
  per-stage and link split is missing: vanilla's stage counters stay at zero on this path, and the front link call
  returns before the rows arrive. Getting the split needs new timing lines at the hand-off wait and in the front
  backend. That is logging only; Chat AI can write it and a Tier 2 turn can run it.
- **What it suggests:** on this rig and model, vanilla's pipelining gains only 4 to 6% on HIP-only, not the 19 to
  21% in `docs/MULTI_GPU.md` (which compared against an older draft-layer placement). A third stage gives pipelining
  more to overlap, but the link and failed guesses take some of it back. A modest gain is likely; how much is open.

## History in short

- M5 reached (Chat AI): option off = vanilla source; `.ai/DIVERGENCE.md`.
- Report 0073: HIP-only pipelined windows +4 to 6%; the mixed per-stage split not measurable with the existing traces.

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

- **Pipelined windows,** one of:
  - (a) Chat AI adds the timing lines, and a short Tier 2 run measures the split, before deciding;
  - (b) go straight to the Tier 1 implementation;
  - (c) put pipelining aside for now.

## Next step

Depends on the decision above.

Roadmap: https://claude.ai/artifact/NXCzvkbBJ2Z9fKts6ADRNn
