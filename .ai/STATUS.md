# Status

Turn holder: **Human User**: the native-pack decision and the next direction (see Open decisions).

## Current milestone

The native-pack question is answered (report 0082, `9de3c43a`; Partial only because outputs differ between packs).

Chat AI's review (greedy, six measured requests per pack, prompt and mode, warm servers):
- **Prefill:** the native pack is faster everywhere.
  - HIP-only: +10.4% / +19.7% / +13.6% (code / long / 16K).
  - Mixed: +39.6% / +16.3% / +3.0%.
  - Report 0080's −11.5% on code was a cold-start artefact.
- **Decode:** the native pack is slower.
  - Medians HIP-only: −2.6% / −7.4% / −0.6%. Mixed: −8.6% / −7.4% / −11.2%; in mixed mode the code ranges do not
    overlap.
  - Where both packs produced identical output, the cost is still there: HIP-only long, first measured request,
    121.9 against 114.6 tok/s (−6%); mixed 16K, about 162 against 141 to 157 (−3 to −13%).
  - So it is not just different text. In mixed mode it is larger, which points at the CUDA front stage's decode kernels
    for the GGUF-form experts (not measured).
- **Per turn the two cancel out:** for a turn with 5,000 new prompt tokens and 500 generated tokens, the production pack
  takes 6.08 s and the native 6.12 s in HIP-only, and 5.70 against 5.71 s in mixed mode. Turns heavier in new prompt
  favour native; turns heavier in output favour production.
- **Report gaps:** drafts accepted per window are not given, but accepted/offered per request is, and acceptance is
  similar between packs.
- **Safety:** boot ID unchanged; no faults or core dumps; 547 s of GPU time. No engine change.

## History in short

- Report 0082: native pack prefill +3 to +40%, decode −1 to −11% (confirmed on identical outputs); per typical turn even.

- Report 0081: `--batch` no-go (vanilla drops pipelined windows with slots); native pack in mixed mode prefill +24 / +16 / +3%, decode unsettled; an Xid 79 on the CUDA card outside the project.

- Report 0080: DFlash2 no-go on this rig; the native pack prefills 13 to 16.5% faster on HIP-only.

- Report 0079: no desync after a cancel (vanilla routes it to Reset / Restore); checkpoint overlap without gain, reverted.

- M8 reached (report 0078): prompt chunks across the link (+12.7% / +16.2% at 16K / 32K); mixed decodes 7 to 17% faster than production.

- M7 reached (report 0077): pipelined decode windows in mixed mode, exact; +11% decode at 16K.

- Report 0076: production mirrors at 262,144 tokens; mixed without pipelining on par with production decode; prefill −3% long, −18% on a short prompt.

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
- 2026-11-18: the next performance step is pipelined decode windows in mixed mode.
- 2026-11-18: keep `--pipeline-windows 2` in the mixed config. Next, in one handoff: tell the front stage when its
  window is judged wrong, cut the short-prompt fixed cost, overlap prompt chunks across the link.
- 2026-11-19: the shutdown at 23:37 on 18 November was the Human User's; report 0078's results stand.
- 2026-11-19: do `--batch` and DFlash2 in mixed mode now; the re-pin to vanilla 0.2.16 waits. Chat AI orders DFlash2
  first (production runs one conversation at a time), each with a go/no-go on HIP-only before implementing.
- 2026-11-19: the NVIDIA Xid 79 is unrelated to this project. Next: the native-pack decode comparison.

## Open decisions for the Human User

- **Native pack:** Chat AI recommends keeping the production pack. Per turn it is even; a switch costs a cold start and
  changes outputs. Switch only if turns carry much more new prompt than generated text.
- **The next direction:** the re-pin to vanilla 0.2.16 (Chat AI's recommendation), or a release-candidate
  qualification.

## Next step

After report 0082: Chat AI lays out the native-pack decision. Then the re-pin to vanilla 0.2.16 (Chat AI's
recommendation), or a release-candidate qualification.

Roadmap: https://claude.ai/artifact/NXCzvkbBJ2Z9fKts6ADRNn
