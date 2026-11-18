# Status

Turn holder: **Human User**: choose the next performance step (see Open decisions).

## Current milestone

Production measured (report 0076, `2f5255e3`). Both production mirrors serve at 262,144 tokens without fallback.

Chat AI's review:
- **Memory:** the CUDA card holds 14.3 of 16 GB, the R9700s 19.2 and 20.6 GB in mixed mode (25.8 / 26.1 GB in
  HIP-only). With `--mmap-experts`, about 50 GiB of RAM is available (page cache included), against about 19 GiB with
  the RAM arena of reports 0069 to 0075. The RAM headroom concern is gone.
- **Overlap:** `STRATA_SPLIT_OVERLAP=1` takes effect on the host's split in mixed mode (log: "overlapped").
- **Caches:** a 16,093-token conversation was parked and restored from a checkpoint under the production cache
  settings (`cache_n=16088`). A `cvec=0` request and a vision request both work.
- **Mixed (no pipelined windows) against production (HIP-only with `--pipeline-windows 2`),** median of three:

  | Prompt | Prefill, mixed / production | Decode, mixed / production |
  | --- | --- | --- |
  | Code (118 tokens) | 265 / 324 tok/s (−18%) | 117.8 / 107.9 (+9%) |
  | Long (5,000) | 2,287 / 2,351 (−3%) | 118.7 / 122.7 (−3%) |
  | 16K | 3,388 / 3,478 (−3%) | 131.2 / 132.2 (−1%) |

  Decode is on par: the samples are small and the generated sequences differ. Prefill is about 3% behind on long
  prompts and 18% on the short one, where the absolute gap is about 80 ms of first-token time. That looks like a fixed
  per-request cost in mixed mode (session ops, Images, Steer, the front stage's synchronised prompt pass);
  unconfirmed.
- **`--layer-split auto` (K = 31) against the fixed 33:** no difference.
- **No engine change; no faults, core dumps or reboots;** about 510 s of GPU time.

## History in short

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

## Open decisions for the Human User

- **The next performance step.** Chat AI recommends pipelined decode windows (Tier 1), starting with the front stage
  answering right after it launches its window (which pipelining needs anyway). Alternatives:
  - the short-prompt fixed cost (about 80 ms);
  - pipelined prompt chunks across the edge;
  - more layers on the CUDA card;
  - `--batch` or DFlash2.

## Next step

Depends on the decision above.

Roadmap: https://claude.ai/artifact/NXCzvkbBJ2Z9fKts6ADRNn
