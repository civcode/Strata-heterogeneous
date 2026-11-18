# Status

Turn holder: **Coding Agent**, handoff `0078` (Tier 1), once the Human User hands it over.

## Current milestone

M8: mixed-mode performance (handoff 0078): the wrong-guess notice, the short-prompt cost, prompt chunks across the link.

M7 reached (report 0077, `f5eec160`): `--pipeline-windows 2` works in mixed mode and emits exactly the serial loop's
token IDs.

Chat AI's review:
- **Asynchronous front window** (`9d1546f5`): the link round trip fell from 3.36 to 0.10 ms per window, and the first
  R9700 now waits about 3.7 ms on the ready word on its GPU. Token IDs are identical to before; decode +0.4 to +0.5%,
  within noise.
- **Pipelined windows** (`7a5fab5f`):
  - token IDs identical to serial on code and long, also with `STRATA_PIPELINE_FORCE_MISS=3` (81 and 52 windows
    undone) and for a seeded sampled request;
  - a repetition penalty falls back to serial with vanilla's message;
  - the server checks pass (streaming, cancel, a 16K two-turn conversation parked and restored, vision, `cvec=0`,
    shutdown).
- **Contract:** two hooks in `Verifier` (`pl_launch`, `pl_commit_async`) and one line in the decode loop's undo, all
  guarded; vanilla's pipelined logic is unchanged. `check_off_is_vanilla.py` reports 6 of 6.
- **Decode, median of three** (mixed serial / mixed pipelined / production): code 121.3 / 118.7 / 107.9; long
  121.7 / 119.2 / 114.5; 16K 132.0 / 146.6 / 125.9.
  - Pipelining gains +11% at 16K. Its −2% on code and long is within noise: production's own medians moved by up to 7%
    between reports 0076 and 0077.
  - Guesses are right 37 to 54% of the time (production 39 to 60%).
  - Prefill is unchanged on long prompts and −6% on the short one.
  - Mixed, serial or pipelined, now decodes faster than production on all three prompts.
- **The mixed mirror config** now has `--pipeline-windows 2` with the two production variables (agent, task 6). The
  agent leaves keeping it to the Human User.
- **Open:**
  - the front stage computes a window already judged wrong to its end (it is not told);
  - `STRATA_SPLIT_TIMING` does not cover pipelined windows;
  - `--pipeline-windows 1` is untested;
  - the short-prompt fixed cost.
- **Since report 0077,** by Chat AI: `.ai/DIVERGENCE.md` updated.

## History in short

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

- None. (Handoff 0078's 40-minute budget is above the allowance; the Human User is told, and it stands unless refused.)

## Next step

After report 0078: Chat AI reviews the hooks, updates `.ai/DIVERGENCE.md`, and proposes the next step:
- `--batch`;
- DFlash2;
- or a long-duration qualification run once the project nears a release candidate.

Roadmap: https://claude.ai/artifact/NXCzvkbBJ2Z9fKts6ADRNn
