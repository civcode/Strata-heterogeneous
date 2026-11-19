# Status

Turn holder: **Human User**: the production version and the next direction (see Open decisions).

## Current milestone

M10 closed: the branch is vanilla 0.2.16 plus the mixed split (report 0084, `1e6390da`; Partial only for the full
option audit).

Report 0084, Chat AI's review:
- **The rename works on the cards:**
  - every default target built in all five trees;
  - host CTest 147/147; `mixed_split_tests` and `mixed_split_cache_drop_test` 2/2 on HIP and CUDA;
  - the mixed mirror served the features suite (16K conversation restored, vision, `cvec=0`, clean shutdown);
  - pipelined against serial exact.
- **Vanilla 0.2.16 removed llama.cpp's MMQ prompt kernels** (`engine/CMakeLists.txt`, `docs/AMD_HIP_PERFORMANCE.md`):
  the MMQ build options are now ignored. HIP-only, the same production arguments, warm blocks, six samples each:

  | Prompt | Prefill, 0.2.15 / 0.2.16 | Decode, 0.2.15 / 0.2.16 |
  | --- | --- | --- |
  | Long | 2,744 / 2,202 (−19.8%) | 106.4 / 117.3 (+10%) |
  | 16K | 3,786 / 3,210 (−15.2%) | 117.7 / 116.2 (−1%) |

  Beyond the spread. MMQ's removal is the likely cause (it was worth about 22% in report 0066), but not isolated. Per
  turn:
  - 5,000 new tokens and 500 generated: 6.52 s on 0.2.15, 6.53 s on 0.2.16;
  - 16,000 and 500: 8.48 s against 9.28 s (+9%).
- **The option audit** names every production argument's meaning on 0.2.16, but does not compare defaults across
  versions. Found: `STRATA_SPLIT_OVERLAP=1` has no effect next to `--pipeline-windows 2`.
- **Safety:** boot ID unchanged, no faults, no core dumps.
- **Since report 0084, by Chat AI:** `.ai/PROJECT.md` notes the MMQ removal and the inert overlap variable.

## History in short

- Report 0084: rename confirmed on the cards; vanilla 0.2.16 prefill 15 to 20% slower than 0.2.15 with MMQ (HIP-only), decode equal or better.

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

- **Production on 0.2.15 or 0.2.16?** Vanilla's own change; about even for 5K-token turns, about 9% slower on 0.2.16
  for 16K-token turns.
- **Report the prefill drop upstream?** Chat AI can draft an issue, following vanilla's `CONTRIBUTING.md`
  (`[Benchmark]`, a same-day comparison); the Human User posts it.
- **The next direction:** a release-candidate qualification (long-duration runs), or more performance work.

## Next step

Depends on the decisions above.

Roadmap: https://claude.ai/artifact/NXCzvkbBJ2Z9fKts6ADRNn
