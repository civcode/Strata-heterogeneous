# Status

Turn holder: **Coding Agent**, handoff `0080` (Tier 1), once the Human User hands it over.

## Current milestone

M9: DFlash2 and `--batch` in mixed mode, DFlash2 first (handoff 0080, turn 1 of 2, starting with a go/no-go on
HIP-only).

M8 closed (report 0079, `ad6e06e7`). No engine change in the end; `check_off_is_vanilla.py` reports 6 of 6.

Chat AI's review:
- **The read-ahead desync Chat AI suspected does not happen,** reproduced on the cards with cancels at 8,192 and 24,576
  tokens. Vanilla already refuses to continue a session after a prompt read stops partway (`pp_exact` / `live_ok`), so
  the next request starts from a Reset or a Restore, which the front stage takes without checking its history. Chat
  AI's reading of the code was wrong. `serve_session.py` keeps the cancel suite.
- **Checkpoint overlap:** built, exact, and no gain. Running at the same time, the two copies slow each other down
  (host 17 → 24.5 ms, front 11.5 → 20.5 ms). Reverted, which is the right call; history keeps it as `d172f54e`.
- **Safety:** boot ID unchanged throughout; no faults or core dumps; 214 s of GPU time.
- **Process note:** the agent's first push was rejected because of Chat AI's status commit after the shutdown
  confirmation, pushed as the handoff was being handed over. Rebased without conflicts.

**Vanilla 0.2.16 is out** (tag of 18 November, 864 commits after 0.2.15):
- The repository is restructured: the engine moved under `engine/` (and `app/`, `terminal/`, `tools/`).
- Our seam files changed moderately in content: `generate.cpp` +1,337 / −141, `verify.cpp` +174 / −37,
  `conversation_cache.hpp` +33 / −1, `verify.hpp` +12, `prefill.cpp` +28 / −1, `prefill.hpp` unchanged.

## History in short

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

## Open decisions for the Human User

- None.

## Next step

After report 0080:
- after a go: DFlash2 turn 2 (checkpoints, parks, the drafter's context across them);
- after a no-go: `--batch`, starting with its own go/no-go: HIP-only aggregate throughput with 2 slots against 1, and
  whether the CUDA card fits 2 slots at 262,144 tokens.

The re-pin to vanilla 0.2.16 stays open.

Roadmap: https://claude.ai/artifact/NXCzvkbBJ2Z9fKts6ADRNn
