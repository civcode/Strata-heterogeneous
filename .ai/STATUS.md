# Status

Turn holder: **Chat AI**, waiting for the Human User's decision on how to resume long runs.

## The 19 November host crash (handoff 0086)

- 18:01 local, 17 minutes into the full mixed session, at a 214K-token context: right after the probe, the restore of
  the main conversation gave non-finite logits; the Python server crashed on an invalid pointer and the kernel found
  overwritten page tables in the HIP engine, then oopsed; the boot ended uncleanly. Evidence and analysis stay with the
  Human User (private).
- Chat AI's reading: memory of two processes overwritten from outside them (a GPU or driver, or hardware); the
  overwritten page-table entries decode as int8 KV-like data; under swap, with the AMD driver repeatedly restoring
  userptr memory and the GPU paused up to 23 s for it. HSA_ENABLE_SDMA=0 was set. Not proven.
- Response (Human User, 2026-11-19): carry private commit `05134f54` (HIP pinned snapshot readback) as the one
  labelled divergence (`.ai/HARDENING.md`); long runs on hold; `HSA_ENABLE_SDMA=0` for every GPU process.
- Handoff 0086's other results: none reported (the report was not written).

## Current milestone

M11: a release candidate on vanilla 0.2.15, qualified on the Human User's real workload: the pi coding agent with
256K-token context, several compactions per long session, and a tool-call pruning / compaction extension. The
qualification uses a synthetic workload shaped like it; model, measurements and pass criteria are in `.ai/RC.md`.

**Back on vanilla 0.2.15** (2026-11-19, the Human User's decision):
- Production stays on 0.2.15 with MMQ, and the project with it.
- `feature/mixed-native` was rebuilt from the 0.2.15 head `7d573bee` and force-pushed. The 0.2.16 line (reports 0083
  and 0084, the merge and the rename on it) is archived as `archive/mixed-native-0.2.16`.
- On the rebuilt line, Chat AI made the `hetero` → `mixed_split` rename afresh and carried the `.ai/` notes and reports
  over. `PROJECT.md` and `DIVERGENCE.md` describe the 0.2.15 base.
- **Checked without a GPU:** CPU tests; `check_off_is_vanilla.py` 6 of 6 against `c775728`; a syntax check with the
  option on (nothing beyond vanilla 0.2.15's own); the Python tools. `mixed_split/` differs from the 0.2.16 line's only
  in its layout lines and build paths.
- **Not yet on the cards:** the build directories must be configured anew (paths moved with the rename). The release
  candidate's first handoff starts with that.
- **Kept from the 0.2.16 work:** the finding that 0.2.16 prefill is 15 to 20% slower on HIP-only (MMQ removed, the
  FP16 path on every layer); already reported upstream by others, so no report from this project.

## History in short

- 2026-11-19: back on vanilla 0.2.15 (0.2.16 line archived); `mixed_split` rename on 0.2.15.

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
- 2026-11-19: production stays on 0.2.15 with MMQ, and the project returns to 0.2.15. The 0.2.16 prefill drop has
  already been reported upstream. Next: a release candidate, with testing and qualification covering the pi coding-agent
  workload: 256K context, compactions, tool-call pruning.

## Open decisions for the Human User

- **The workload numbers in `.ai/RC.md` are estimates** (tool-output sizes, pruning every 8 turns of outputs over
  1,000 tokens, a summary of about 8,000 tokens). The Human User may correct them; each is a flag of the driver.

## Next step

Report 0088: the live-versus-fresh probe differences occur on HIP-only too (vanilla plus the carried patch); with
caches off the tested probe was equal; two fresh mixed servers agreed. So "equal" was too strict a criterion. Chat AI
changed `.ai/RC.md` criterion 2 to "equal or a near-tie" (live token in the fresh top 5, at most 1.0 nat below), plus
no non-finite-logits line; `agent_session.py replay` now asks for top-5 logprobs and judges (selftest 18/18). The
driver now turns thinking off for `--reasoning-budget 0` (report 0088).

Open decision for the Human User: how to resume long runs after the 19 November crash (swap setting, order of
HIP-only and mixed long sessions, `HSA_USERPTR_FOR_PAGED_MEM`). The setup integration waits on
`chat/setup-mixed-split`.

Roadmap: https://claude.ai/artifact/NXCzvkbBJ2Z9fKts6ADRNn
