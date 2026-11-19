# Status

Turn holder: **Coding Agent**, handoff `0079` (Tier 1), once the Human User hands it over and confirms the 18 November shutdown.

## Current milestone

M8 reached (report 0078, `5b82ad6d`), pending the Human User's confirmation of the shutdown during the turn.

Chat AI's review:
- **Reboot:** the machine was shut down cleanly at 23:37 on 18 November, while the turn was paused, and booted at 08:01.
  The agent missed it at resume and ran about 14 more GPU minutes before noticing at the final check: no faults, no
  core dumps. `.ai/PROJECT.md` now checks the boot ID after any pause. If the shutdown was the Human User's, the
  results stand.
- **Wrong-guess notice:** measured and correctly not added. The front window is always launched before the verdict;
  82 to 89% of wrong guesses find it already ended; the host's undo never waited for it.
- **Short prompts:** the gap is the CUDA card's own prompt compute, 8.5 ms per layer against about 3.6 ms on an R9700.
  - Report 0076's 80 ms was a first-request effect: every server measurement starts a fresh server.
  - Pipelined against serial costs only on the first request (+33 ms); warm requests are equal.
  - An MMQ build of the front stage helps short prompts (+13% warm) but costs 6 to 14% on long ones: not adopted.
- **Prompt chunks across the link** (`2fb7f403`): exact. Prefill +12.7% at 16K and +16.2% at 32K.
  `--prefill-pipe 700` now gives +27.6% on 5K prompts but −5.4% at 32K: it stays off.
- **Mixed against production, final:**
  - prefill −22.5% on the code prompt, −2.4% on long, +11.4% at 16K;
  - decode +12.7%, +7.3%, +16.5%.
- **Contract:** guarded hooks; `Prefill::set_remote_in`'s callback now gets the next chunk's size and returns its
  buffer. `check_off_is_vanilla.py` reports 6 of 6.
- **Found by Chat AI in review (from the code, not reproduced):** a cancel between prompt chunks leaves the front stage
  a chunk ahead of the host, and nothing marks its session invalid. A continuation would then be refused. Handoff 0079
  reproduces and fixes it.
- **Since report 0078,** by Chat AI: `.ai/DIVERGENCE.md` updated; the standing safety rule added to `.ai/PROJECT.md`.

## History in short

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

## Open decisions for the Human User

- Confirm that the clean shutdown at 23:37 on 18 November was the Human User's.

## Next step

After report 0079: Chat AI reviews it. Then the next direction: `--batch`, DFlash2, or a release-candidate
qualification with a long-duration run (`.ai/WORKFLOW.md` Section 12).

Roadmap: https://claude.ai/artifact/NXCzvkbBJ2Z9fKts6ADRNn
