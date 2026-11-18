# Status

Turn holder: **Coding Agent**, handoff `0074` (Tier 1) open.

## Current milestone

M6: the mixed config runs what the HIP-only config runs, vision and the control vector (handoff 0074).

- **Control vectors,** implemented by Chat AI (`e7d6d857`, CPU-tested and syntax-checked, not run on the cards):
  - the front stage loads the same vector, since the options were already forwarded to it;
  - `SessionOp::Steer` repeats each request's switch;
  - the front stage's keys carry the switch.
- **Vision** (Tier 1 in handoff 0074): the host sends the front stage each request's position table, image rows and
  image keys. The front stage's keys gain the image keys.

Pipelined windows are set aside for now (Human User, 2026-11-18). They stay an important goal and are revisited later.
Report 0073 measured +4 to 6% for HIP-only; the mixed per-stage split still needs timing lines.

## History in short

- Report 0073: HIP-only pipelined windows +4 to 6% on this rig; mixed per-stage split not measurable yet.

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
- 2026-11-18: set pipelined windows aside for now (an important goal, revisited later); do vision and control vectors
  in mixed mode first.

## Open decisions for the Human User

- None.

## Next step

After report 0074: Chat AI reviews the vision change against the contract and updates `.ai/DIVERGENCE.md`. Then one
of: the remaining refusals (`--batch`, DFlash2, `--layer-split auto`), or pipelined windows, as the Human User
decides.

Roadmap: https://claude.ai/artifact/NXCzvkbBJ2Z9fKts6ADRNn
