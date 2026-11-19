# Status

Turn holder: **Coding Agent**, handoff `0084` (Tier 2), once the Human User hands it over.

## Current milestone

M10: re-pinned to vanilla 0.2.16 (report 0083, `a4d3f434`; Partial). Handoff 0084 finishes it and checks the rename on
the cards.

Report 0083, Chat AI's review:
- **The merge** (`50450647`, no force-push; the pre-merge head is archived as `archive/mixed-native-0.2.15`) had one
  conflict: the `<functional>` include in `conversation_cache.hpp`, which vanilla 0.2.16 now includes itself. Git
  carried every guarded block into `engine/`. With the option off, the six changed vanilla files are vanilla 0.2.16
  (`check_off_is_vanilla.py`, also run by Chat AI). The branch adds 1,534 lines to 7 vanilla files and deletes none.
- **Checks passed:**
  - `mixed_split_tests` 2/2 in every tree;
  - both mirrors serve;
  - pipelined against serial exact (300/300 and 200/200), also with forced misses;
  - the server suites, with a 16K conversation parked and restored, vision and `cvec=0`.
- **Mixed against production on 0.2.16** (first requests, fresh servers, median of three):
  - prefill −7.7% / +3.4% / +16.0% (code / long / 16K);
  - decode +7.4% / +22.8% / +13.2%.
- **Open:**
  - all-target builds;
  - a full audit of the production options on 0.2.16;
  - prompt reading looks slower on 0.2.16 than in reports 0078 / 0082 (HIP-only long 1,975 against 2,372 tok/s), but
    under other conditions. Handoff 0084 measures it controlled.
- **Since report 0083, by Chat AI:**
  - the rename to `mixed_split` (Human User's request, `740e2bea`). Checked without a GPU: CPU tests, off = vanilla
    0.2.16, syntax with the option on, the Python tools;
  - `.ai/PROJECT.md` updated for the 0.2.16 base and layout.

## History in short

- Report 0083: merged vanilla 0.2.16 (one small conflict); re-checks pass; then renamed `hetero` to `mixed_split`.

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
- 2026-11-19: re-pin to vanilla 0.2.16 next.
- 2026-11-19: rename every `hetero` to `mixed_split` / `MIXED_SPLIT` in the project's own code.

## Open decisions for the Human User

- **Native pack:** Chat AI recommends keeping the production pack (report 0082: even per turn). No action needed unless
  the Human User wants to switch.

## Next step

After report 0084: Chat AI reviews it, and lays out what 0.2.16 means for the production setup. Then a release-candidate
qualification (`.ai/WORKFLOW.md` Section 12: long-duration runs), or the next performance step.

Roadmap: https://claude.ai/artifact/NXCzvkbBJ2Z9fKts6ADRNn
