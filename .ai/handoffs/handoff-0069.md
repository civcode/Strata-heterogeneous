# Handoff 0069

## Tier
1

The main item, caches across the boundary, changes how two processes keep their sessions in step inside vanilla's
serve loop. Whether it works is judged only on the cards, and a wrong step shows up as a silently wrong continuation,
not a crash.

## Objective
Milestone M4: the mixed profile serves like vanilla. That means prompt checkpoints and the conversation cache work
across the boundary, RAM has headroom, and the run configs and test drivers are in the repository.

## Context
- Read `.ai/PROJECT.md`, `.ai/SEAM.md` and `.ai/reports/report-0068.md` first. Report 0068 reached M2: mixed requests run
  through vanilla's loop, and through `serve/server.py` they start, stream, cancel and shut down cleanly. Chat AI
  reviewed its engine changes (vanilla options for the front stage, PLE left to the front stage, arena-record fix) and
  accepts them.
- **RAM.** Both processes are built for a whole-model RAM expert arena. In report 0068 the front stage moved to
  `--mmap-experts`, but the host still holds a 31.6 GiB arena, including about 9.9 GiB for layers 0 to 14 that it never
  reads, on a 60 GiB machine. Under RAM pressure the amdgpu `svm_range_restore_work ... hogged CPU` warnings appear, and
  one run stalled. Vanilla has `--shared-expert-arena FILE` ("back the resident arena with one MAP_SHARED file", for
  arenas shared by processes). If both processes can use it, there is one copy of the experts in RAM, and the front
  stage fills its cache from RAM instead of the GGUF file, with no new code.
- **Caches.** `hetero::SessionOp` already defines Rollback, Snapshot, Restore and Truncate. The front stage answers
  only Reset and Abandon (`generate.cpp`, the front backend's session handler: "does not save or restore its session
  yet"), so mixed mode turns prompt checkpoints and the conversation cache off (the two `o.prompt_cache = 0;
  o.conversation_cache_mib = 0;` blocks). Every request reads its prompt from token 0. With HIP-only, report 0067
  reused 58 tokens from the conversation cache.
- **M4 leftovers.** The run configs (`runs68/mixed.json`, `runs68/hiponly.json`) and the drivers (`/tmp/s68`) are not
  committed. The roadmap's M4 also asks that the mixed decode rate does not depend on how fast the client reads.

## Tasks, in this order
1. **One copy of the experts in RAM.** Try vanilla `--shared-expert-arena` on both processes, with the file on tmpfs or
   wherever vanilla documents it. The front stage stops forcing `--mmap-experts` when a shared arena is given. If that
   option cannot serve this case, the fallback is the smallest change that keeps the host's arena to its own layers,
   following how vanilla skips foreign layers' weights (`add_foreign`). Report available RAM with the server idle,
   before and after.
2. **Session operations across the boundary.** Bind Rollback, Snapshot, Restore and Truncate on the front stage to the
   same vanilla session calls the host makes for its own layers, with the same arguments. On the host, send each op
   where vanilla makes the call on its first stage. Then let mixed mode keep vanilla's default prompt checkpoints and
   conversation cache. The front stage's checkpoint store is its own, sized and stored as vanilla does for one stage
   of a split. An op the front stage cannot complete fails the request; it is never skipped.
3. **Unused head on the front stage.** The front stage loads the native output head (437 MB of VRAM) and never uses it.
   Skip it if vanilla already has a way to skip it; otherwise make the smallest guard behind the front-stage option.
4. **Commit the M4 material.** Commit the two run configs as examples under `hetero/configs/`, with machine paths as
   placeholders or comments, and the drivers under `hetero/tools/`. They must work without editing the vanilla
   checkout.

## Boundaries you decide within
- Same as handoff 0068. Vanilla options before new code; any change to a vanilla file is the smallest hook, behind
  `STRATA_HETERO` and the mixed-split options. No new cache, checkpoint, scheduling or synchronisation logic: the front
  stage repeats vanilla's operations on its own layers.
- Do not weaken the ported hardening or the 0068 safety gates (identity check, publish after the front stage's
  commit acknowledgement, bounded hand-off wait, cooperative shutdown).
- `hetero_tests` must pass. Add a test there for each newly bound session op, using the stub backend.
- If the shared arena is not usable and the fallback needs more than a hook, stop task 1 and report. Continue with
  tasks 2 to 4 only if the RAM measured with the server idle leaves room for the front stage's checkpoints. Otherwise
  report Partial.

## Checks
- **Cache correctness**, mixed only, greedy, MTP off:
  - a two-turn conversation where turn 2 reuses turn 1 from the conversation cache;
  - the same turn 2 sent to a freshly started server, so nothing is reused.
  The first 50 token IDs of turn 2 must be equal. This compares mixed with mixed, so a difference is a bug, not a
  near-tie.
- **Reuse:** the number of reused tokens in mixed mode, beside HIP-only on the same conversation.
- **Through `serve/server.py` with the committed mixed config:**
  - health and model listing;
  - one streaming, one non-streaming and one sampled request;
  - a cancelled request followed by a normal one;
  - a streaming request whose client reads slowly (for example, a 50 ms pause per chunk), with its decode rate from
    the engine log against a fast reader;
  - a clean shutdown with SIGTERM.
- HIP-only from the same HIP build still serves, with conversation-cache reuse, as in report 0067.

## Constraints
- All rules of `.ai/PROJECT.md`. Never kill a GPU process by force. No two GPU jobs at once, except the two processes
  of one mixed run.
- Record the kernel log and `coredumpctl` before the first run and after the last, as in handoff 0068. A GPU fault
  line, a crash outside the run or a reboot ends the handoff at once. If the `hogged CPU` warnings recur with the
  server idle after task 1, report it.
- Do not touch the frozen branch, the vanilla checkout or the private repository. Work on `feature/mixed-native`, and
  commit in small steps.

## Hardware Test Budget
30 minutes of GPU run time in total; every single run under 2 minutes. Report 0068 used 13 of its 60 minutes. Stop
starting runs when the budget is used, and report what was not reached.

## Completion Criteria
- RAM: one copy of the experts, or the host's arena limited to its own layers.
- The session ops are bound and tested, and both caches are on in mixed mode. Turn 2 with and without reuse gives the
  same 50 token IDs.
- The server checks pass, and the slow reader's decode rate is within 5% of the fast reader's.
- The configs and drivers are committed; `hetero_tests` passes; HIP-only is unchanged.
- `.ai/reports/report-0069.md` and all changes are committed and pushed.
- If stopped early: **Partial**, with the tasks reached.

## Reporting
- Every vanilla file changed, with the function and one line on why; design decisions made inside the boundaries.
- Available RAM with the server idle, and VRAM per card, before and after tasks 1 and 3.
- Reused tokens in mixed mode and in HIP-only. Turn-2 TTFT with and without reuse.
- The token-ID comparison. Slow-reader against fast-reader decode rate. Prefill and decode tok/s beside report 0068.
- Kernel log and core-dump lines, or "none". GPU run time used. What remains open.
