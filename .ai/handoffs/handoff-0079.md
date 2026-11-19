# Handoff 0079

## Tier
1

A suspected session desync after a cancelled long prompt, found by Chat AI from reading the code, and one small
overlap. Both change how the host and the front stage stay in step, and both are judged on the cards.

## Objective
1. Reproduce and fix the front stage running ahead of the host after a prompt read stops between chunks.
2. Overlap the front stage's part of a checkpoint with the host's own checkpoint save.

## Context
- Read `.ai/PROJECT.md`, including the new standing safety rule (boot ID after any pause), `.ai/STATUS.md` and report
  0078. With `STRATA_MIXED_SPLIT` off, the engine sources must stay vanilla's line for line: run
  `hetero/tools/check_off_is_vanilla.py` before every commit that touches `src/` or `include/`.
- **The suspected desync (Chat AI, from the code, not reproduced):**
  - `HeteroHost::prefill` starts the next chunk on the front stage (PfStart) as soon as the current chunk's rows are
    there. The front stage's `f_pf_start` records the chunk in its book (`book.on_prefill`) at once, and its run moves
    the front stage's GDN state past it.
  - Vanilla's chunk loop (`Prefill::run_impl`) checks `should_stop` at the start of every chunk. A cancel (`stop_req`)
    therefore ends the read with "cancelled" after the front stage has started the next chunk.
  - The host drops the unread chunk (`pf_buf = -1`, `HeteroHost`, "any other call ends a prompt chunk read ahead")
    but nothing marks the front stage's live session as ahead of the host's.
  - The next request that continues the host's live session would send a prompt chunk at the host's position.
    The front stage would refuse it as out of step: an error for that request, or the engine's end, whichever vanilla
    does with a failed prompt read.
  - Report 0078 noted the book "refuses every later mismatch, so it cannot corrupt silently". It is the refusal itself
    that is the problem.
  - The same applies to any other way a read ends between chunks with a chunk read ahead: a failure in a host stage,
    or `stop_at_chunk`.
- **The checkpoint overlap** (report 0078, task 4): after the prompt, the front stage saves its part of the checkpoint
  (11.5 ms) in series with the host's own save (17 ms). Report 0078 did not do it, because vanilla skips a checkpoint
  that could not be saved (#3350), so an asynchronous Snapshot needs a wait that carries its result.

## Tasks
1. **Safety checks** (`.ai/PROJECT.md`): the boot ID, kernel log and core dumps, before the first GPU run, after the
   last, and after any pause.
2. **Rebuild** the HIP and CUDA trees, and stamp them.
3. **Reproduce the desync** through the server, with the mixed mirror:
   - a 32K-token request, cancelled by the client after its first chunk is read (about 2 to 3 s in; the engine log
     shows the chunk);
   - then a request that continues the same conversation (same prompt, one more turn);
   - then an unrelated request.

   Report what each returns and whether the engine stays up. If it does not reproduce, say why (for example, the host
   already starts the next request from a Reset or a Restore) and skip the fix.
4. **Fix it,** if it reproduces. When a chunk read ahead is dropped unread, the front stage's session is ahead of the
   host's. Its GDN state cannot be rolled back, so the host must not continue its live session on the front stage.
   - At the next use, take vanilla's existing path for a live session that cannot be used: restore the best checkpoint,
     or reread from the start. The front stage then gets a Restore or a Reset.
   - Do it with the smallest guarded hook. A flag in `HeteroHost`, set wherever an unread chunk is dropped, and read
     where vanilla decides whether the live session can be reused, is one way.
   - Do not snapshot the front stage before each read-ahead: that costs every long prompt for a rare case.
   - **Check:** task 3 again (no error; the continuation reuses a checkpoint where one exists). Plus token IDs of a
     normal 16K two-turn conversation unchanged against the previous commit, with `STRATA_IQ_MT_MIN=1`.
5. **The checkpoint overlap:** the front stage's checkpoint part starts at once, and the host saves its own while it
   runs. The host then waits for the front stage's result before it counts the checkpoint as saved, so vanilla's
   "skip a checkpoint that could not be saved" still holds on both sides.
   - **Check:** token IDs unchanged, and time to first token on the warm code prompt (second and third request of a
     server, three servers), against the previous commit.

## Boundaries you decide within
- As in handoff 0078. The smallest hooks in vanilla files, inside `#if defined(STRATA_MIXED_SPLIT)`. No new caching
  or session logic: the fix routes the case into vanilla's own "live session not usable" path.
- Every safety gate stays:
  - the identity check;
  - publish after the front stage's commit acknowledgement;
  - the bounded waits, with `release_gpu_waits` on an error;
  - cooperative shutdown.

## Hardware Test Budget
20 minutes of GPU run time (within the project's 30-minute allowance). Every single run under 2 minutes. Priority
order: tasks 3, 4, 5.

## Constraints
- All rules of `.ai/PROJECT.md`. Run evidence goes under `.ai/runs/0079-*`; reusable drivers go into `hetero/tools/`.
  Never kill a GPU process by force. No two GPU jobs at once, except the two processes of one mixed run.
- Commit in small steps.

## Completion Criteria
- Task 3 reproduced or explained; if reproduced, task 4's checks pass.
- Task 5 done with its checks, or reported with the reason.
- `mixed_split_tests` (with a test for the desync case if the fix touches `hetero/`) and `check_off_is_vanilla.py`
  pass.
- `.ai/reports/report-0079.md` and all changes are committed and pushed to `feature/mixed-native`.

## Reporting
- What task 3 showed, before and after.
- Every vanilla file changed, with the function and one line on why.
- The time-to-first-token table for task 5.
- Safety lines (boot ID, kernel log, core dumps), or "none". GPU run time used. What remains open.
