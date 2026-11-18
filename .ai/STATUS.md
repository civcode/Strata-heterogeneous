# Status

Turn holder: **Coding Agent**, handoff `0075` (Tier 1) open.

## Current milestone

M6 reached (report 0074, `9a16e06f`): the mixed config runs what the HIP-only config runs, vision and the control
vector. `mixed.example.json` is the counterpart of `hiponly.example.json`.

Chat AI's review:
- **Engine change:** only `generate.cpp` changed, all inside the guards; `check_off_is_vanilla.py` reports 6 of 6.
  The host sends each request's image input as vanilla built it (link message Images, protocol 2), and the front
  stage's keys gain the pictures. The front stage's table upload copies vanilla's eight-line `upload_mrope` lambda:
  acceptable duplication in guarded code.
- **Control vectors:** on and off differ, as in HIP-only. A conversation parked with `cvec=1` is not reused at `cvec=0`
  and is reused again at `cvec=1` (52 tokens). Against HIP-only, a near-tie at token 0, as in report 0068.
- **Vision:** two different images with the same question each get their own description, including two same-size
  images with swapped colours (same image tokens, no reuse across them). A text request after an image is unaffected.
  Reuse is 318 tokens from an image's own checkpoint, and 410 in a two-turn chat about one image.
- **Long prompt,** like-for-like configs, untraced, median of three: mixed 2,257 / 107.9 tok/s (prefill / decode),
  HIP-only 2,308 / 105.4.
- **Finding:** with the vector loaded but off, mixed is not bit-identical to mixed without it (layer 14, the front
  stage's last layer). Likely rounding; recorded in `.ai/DIVERGENCE.md`.
- **Since report 0074,** by Chat AI: `.ai/DIVERGENCE.md` updated. `check_launch.py` no longer counts the scripts in
  `hetero/tools/` or the configs as built sources, so drivers can be committed there without staleness (report 0074
  had to keep its driver in the run evidence).

## History in short

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

After report 0075: Chat AI reviews it and proposes the next performance work from its numbers:
- pipelined decode windows;
- pipelined prompt chunks across the CUDA/HIP edge (the front stage reading chunk c+1 while the first R9700 reads c);
- or `--batch` / DFlash2, from the design notes.

Roadmap: https://claude.ai/artifact/NXCzvkbBJ2Z9fKts6ADRNn
