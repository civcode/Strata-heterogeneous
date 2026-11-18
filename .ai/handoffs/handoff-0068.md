# Handoff 0068

## Tier
1

The remaining work is engine bring-up across two GPU runtimes inside vanilla's 21,000-line `generate.cpp`. Each step is
judged only by a CUDA build, a HIP build and a run on the cards, and the next failure is not predictable from the source
(report 0066 stopped on one that took a source audit to find). Writing it blind in the Chat AI environment would produce
a large body of unverified engine code.

## Objective
Milestone M2: a mixed request completes through vanilla's own generation loop with layers [0,15) on the NVIDIA card and
[15,33) + [33,48) on the two R9700.

## Context
- Read `.ai/PROJECT.md` and `.ai/SEAM.md` first. `.ai/HARDENING.md` lists the hardening now in the engine; report 0067
  showed it builds, passes its GPU tests and serves HIP-only (decode 115 to 122 tok/s, prefill 2831 tok/s at 6.4K
  tokens with MMQ; prompts differ from report 0065, so not a controlled comparison).
- The mixed-engine code from handoff 0066 is re-applied on top of the hardening (`584e8e62`). It adds `--front-exe`,
  `--front-layers K`, repeatable `--front-arg` on the host and `--front-stage K` on the front stage; hooks in `Verifier`
  (`set_front`) and `Prefill` (`set_remote_in`, `set_remote_out`), all behind `STRATA_HETERO`. Mixed mode turns the
  prompt and conversation caches off and refuses vision, control vectors, `--batch`, DFlash2, `--pipeline-windows`,
  peer devices, helper caches and `--layer-split auto`.
- Report 0066's front stage failed with `fused_gr_read: invalid arguments`. Chat AI traced it: two whole-model graph
  captures (`session_capture`, `session_capture_token`) are limited to the first stage's range only when `multi_gpu`
  is set. The front stage runs on one card (`multi_gpu` false), so they captured all 48 layers while it had loaded the
  dense weights of layers 0-14 only. Commit `33e5782c` limits both to the process's own layers. **Unverified: only
  syntax-checked.**
- The same pattern may recur: `generate.cpp` has 35 `!multi_gpu` conditions. In the front stage's process, a path that
  treats "not a layer split" as "this process holds every layer" is wrong. On the host, a path that assumes the first
  local stage starts at layer 0 is wrong.
- Builds: the HIP tree uses both MMQ options (`.ai/PROJECT.md`); the CUDA tree uses CUDA 13.0, sm_89; both with
  `-DSTRATA_HETERO=ON`.

## Design (decided; see `.ai/SEAM.md`)
- Both processes are the ordinary `strata` program. The HIP build hosts the serve loop: its first local stage starts at
  layer 15 and reads its residual from the stage hand-off, as a later stage of a vanilla split does. Head, sampler,
  MTP drafter and the serve protocol stay there.
- The CUDA build, started by the host through `WorkerProcess`, runs layers [0,15) with vanilla's single-card objects
  and serves the front link on descriptor 3 instead of the serve protocol.
- The edge is vanilla's own hand-off: buffer and ready word in one `SharedRegion` that each process registers as mapped
  pinned memory (the probe in reports 0063 and 0064 showed both runtimes accept this). Only residual rows cross.

## Boundaries you decide within
- A change to a vanilla file is a behaviour-preserving extension of how vanilla already handles a layer split's range,
  or the smallest hook the seam needs. No new scheduling, speculation, prefill, cache, sampling or synchronisation logic.
- Everything stays behind `STRATA_HETERO` and the mixed-split options. Without them both builds behave as before.
- Do not weaken or bypass the ported hardening (ownership, fences, drain-on-failure, terminal failure checks).
- The `hetero/` interfaces are the design; extend them with tests if needed (`hetero_tests` must pass). If the design
  cannot work, stop and report why.
- Keep the safety gates: identity check before the first window, a window is published only after the front stage
  acknowledged its commit, vanilla's bounded hand-off wait, cooperative shutdown of both processes.

## Bring-up order
1. Front stage alone on the CUDA card with `hetero/tools/front_client` (handoff 0066): loads layers [0,15), takes the
   hand-off memory, serves the link, answers a window and a commit.
2. Host plus front stage, greedy decode, one row per window, MTP off.
3. MTP windows with native `--spec 4 --spec-min-p 0.5`.
4. A prompt of a few thousand tokens through the native prompt path.
5. Through `serve/server.py --engine strata` with a run config whose `exe` is the HIP build and whose arguments carry
   the mixed-split options: one streaming and one non-streaming request, a cancelled request followed by a normal one,
   then a clean shutdown of both processes.

## Constraints
- All rules of `.ai/PROJECT.md`. Never kill a GPU process by force. Never two GPU jobs at once, except the two processes
  of one mixed run.
- Before the first GPU run and after the last, record the kernel log lines matching
  `amdgpu|gfxhub|GCVM|MES|page fault|segfault|general protection` and `coredumpctl list` since boot, as in handoff 0067.
  A new GPU fault, a crash of any process outside the run, or an unexpected reboot ends the handoff at once.
- Do not touch the frozen branch, the vanilla checkout or the private repository. `wip/mixed-front-0066` is history;
  work on `feature/mixed-native`.
- Commit in small steps.

## Hardware Test Budget
Up to 60 minutes of GPU run time in total (agreed by the Human User for this bring-up). Keep every single run under 2
minutes and use short prompts except in step 4. When the budget is used, stop starting runs and report what was not
reached.

## Completion Criteria
- Bring-up steps 1 to 5 work; both processes end cleanly; `hetero_tests` passes; with the mixed-split options absent,
  the HIP-only request of report 0067 still runs from the same HIP build.
- If the budget or a genuine blocker stops it early: **Partial**, with the last step that works. A redesign is not an
  acceptable outcome.
- `.ai/reports/report-0068.md` and all changes are committed and pushed to `feature/mixed-native`.

## Reporting
- Every vanilla file changed, with the function and one line on why; design decisions made inside the boundaries.
- Which of the `!multi_gpu` paths needed a change, and which were checked and left alone.
- Mixed numbers beside report 0067's HIP-only numbers on the same prompts: prefill and decode tok/s, drafts accepted,
  memory in use per card. For greedy decode with MTP off, whether the first 50 token IDs equal HIP-only's, and if not,
  the first position that differs (a finding, not a failure).
- Kernel log and core-dump lines that appeared, or "none". GPU run time used. What remains open.
