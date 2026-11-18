# Status

Turn holder: **Coding Agent**, handoff `0076` (Tier 2), once the Human User hands it over.

## Current milestone

Production as the reference: bring up both production mirrors and measure mixed against production (handoff 0076).

Report 0075 was Completed (`96a18261`). Chat AI's review:
- **Prompt pipeline, HIP-only:** best R = 700. That gives +3.1% prefill on the long prompt and −0.4% at 16K (the
  default already reads 16K in two 8,192-token chunks). Not worth turning on here. In mixed mode R = 700 costs 14% at
  16K: each extra chunk is another serial CUDA pass.
- **Vision and control vector:** no measurable cost on text requests. The vector even raises draft acceptance on the
  coding prompt (+2.7% decode).
- **The mixed decode window is fully serial.** Vanilla's `Verifier::run` synchronises before the front stage answers.
  Per window (coding prompt): CUDA 3.3 ms (19%), first R9700 5.45 (31%), second R9700 with the head 5.68 (32%),
  drafter 1.6 (9%), commit 0.7 (4%), gaps about 0.9. The link adds 0.03 ms. The CUDA card is faster per layer than an
  R9700 (0.22 against 0.30 ms).
- **`--layer-split auto` in mixed mode:** implemented by adding a starting layer to vanilla's planner, all guarded. It
  picks K = 31; no A/B yet. `--split-skip-if-fits` is now refused in mixed mode.
- **Design notes** (in the report): `--batch` takes 2 Tier 1 turns for serial groups, plus 2 for pipelined groups
  (shared with pipelined windows); DFlash2 takes 2 Tier 1 turns plus a short Tier 2 turn.
- **Contract:** the timing hook in `Verifier` (`set_launch_mark`) and the planner hook are guarded, and timing is
  logging only. `check_off_is_vanilla.py` reports 6 of 6.
- **After report 0075,** Chat AI re-applied the production-mirror revision (configs, launcher, `PROJECT.md`), set the
  mixed mirror to `layer_split: auto`, and updated `.ai/DIVERGENCE.md`.
- **The numbers of reports 0067 to 0075** come from the older example settings, not production.

## History in short

- Report 0075: mixed decode window measured (serial, CUDA 19%, link 0.03 ms); prompt pipeline not worth it; vision/cvec free; `--layer-split auto` in mixed mode.

- M6 reached (report 0074): vision and control vectors in mixed mode.

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
- 2026-11-18: next handoff (0075) combines four things:
  - calibrating vanilla's prompt pipeline (`--prefill-pipe`) on HIP-only (the production config has none);
  - the A/B on vision and the control vector;
  - timing lines for the mixed decode window;
  - the remaining refusals: `--layer-split auto` implemented, design notes for `--batch` and DFlash2.
- 2026-11-18: the production setup is the reference for performance comparisons (`.ai/PROJECT.md`). The example
  configs mirror it; reports 0067 to 0074 ran other settings.

## Open decisions for the Human User

- None.

## Next step

After report 0076: the Human User chooses the next performance work, with mixed-against-production numbers in hand:
- **Pipelined decode windows:** the target is the 19% CUDA share; the link is cheap.
- **Pipelined prompt chunks across the edge.**
- **Moving layers onto the faster CUDA card:** about 0.08 ms per layer, VRAM permitting.
- **`--batch` or DFlash2,** from the design notes.

Roadmap: https://claude.ai/artifact/NXCzvkbBJ2Z9fKts6ADRNn
