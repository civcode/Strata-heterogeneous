# Handoff 0087

## Tier
2

Build, test and check on code that is complete. Scope the agent may change: build directories, run configs (build
paths only), and `.ai/` notes it is asked to write. A failure in the engine is reported, with Tier 1 escalation
recommended, not fixed here.

## Objective
Bring the newly carried patch onto the cards: private commit `05134f54`, the HIP pinned snapshot readback and
conversation-restore preparation (`.ai/HARDENING.md`). Short runs only.

## Context
- Read `.ai/PROJECT.md` (new: point 6, the carried divergence; `HSA_ENABLE_SDMA=0` for every GPU process; long runs on
  hold; more kernel-log patterns), `.ai/HARDENING.md` and `.ai/STATUS.md` (the 19 November crash).
- **Handoff 0086 ended in a host crash** at 18:01 on 19 November, during the full mixed session. Its builds
  (`build-0085-*`) are at a commit without the patch: build anew. Do not resume handoff 0086. No report 0086 is
  expected.
- Chat AI checked without a GPU: `check_off_is_vanilla.py` 6 of 6 against `base/vanilla-0.2.15-pinned-readback`; the
  CPU build with the option on, 145 of 145 tests (6 skipped as before); a syntax check against CUDA headers (nothing
  new). Not checked: the HIP and CUDA builds and the patch's GPU tests.

## Tasks
1. **Safety checks** before the first GPU run, after the last and after any pause (`.ai/PROJECT.md`, with the new
   patterns). Also record `free -m` and `swapon --show` at the start and the end.
2. **Builds** at the current commit, in new directories `build-0087-*`, `-DSTRATA_MIXED_SPLIT=ON -DSTRATA_BUILD_TESTS=ON`,
   28 jobs: CUDA (13.0, sm_89) and HIP (gfx1201, `-DSTRATA_PREFILL_MMQ=ON -DSTRATA_MMQ=ON`), both vision helpers. Stamp
   each. Point the mirrors at them.
3. **The patch's GPU tests**, with `HSA_ENABLE_SDMA=0` exported, one at a time: `conversation_readback_test`,
   `conversation_prepare_test`, `conversation_copy_streams_test`, `conv_cache_test`,
   `conversation_restore_orchestration_test`, in the HIP tree and the CUDA tree where built. Then
   `mixed_split_tests` and `mixed_split_cache_drop_test` in both trees.
4. **HIP-only serving with the production mirror** (`hiponly.example.json`, through `check_launch.py`):
   - `serve_session.py ... features` (a two-turn chat over 16K with another request between the turns: the first
     conversation is parked and restored; control vector; vision);
   - in the server log: each park and restore, and no `skip restore` line other than the new
     "resource preparation unavailable" form if one occurs (report its text).
5. **The same with the mixed mirror** (`mixed.example.json`): `features`, then `checks` (includes a cancelled request).
6. **Park/restore at a medium context, mixed mirror:** `agent_session.py run OUT --scaled` (32K context, one
   compaction, the three probes), then replay the probes on a fresh mixed server. All requests succeed; report the
   probe results.

## Hardware Test Budget
30 minutes of GPU run time (the allowance). Every single run under 10 minutes. No run past a 64K-token context.
Priority: tasks 3, 4, 5, then 6.

## Constraints
- All rules of `.ai/PROJECT.md`. Evidence under `.ai/runs/0087-*`. Never kill a GPU process by force. No two GPU jobs
  at once, except the two processes of one mixed run.
- Stop at once and report if a kernel log line matches the safety patterns, a process crashes, or swap use grows by
  more than 1 GiB during a run.
- No changes to engine code.

## Completion Criteria
- Builds pass; the patch's GPU tests pass (or each failure is listed with its output).
- Tasks 4 to 6 ran, or the stop rule was taken and reported.
- `.ai/reports/report-0087.md` and all changes are committed and pushed to `feature/mixed-native`.

## Reporting
- Builds and tests per tree (passed, failed, skipped), with `HSA_ENABLE_SDMA=0` confirmed.
- The serving checks: parks, restores, any skipped restore and its text, probe results.
- Safety lines, swap before and after, or "none". GPU run time used. What remains open.
