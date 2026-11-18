# Handoff 0070

## Tier
2

A run-and-measure turn on code that is complete. Scope the agent may change: build directories, run configs, the new
launch check script, the test drivers in `hetero/tools/`. A failure inside the engine or the front stage is reported,
with Tier 1 escalation recommended, not fixed here.

## Objective
Re-check the mixed split on the cards now that the strata-dev-private hardening is gone. Verify the new front-stage
`Drop` on real evictions. Add the launch-config check of `.ai/PROJECT.md` rule 7.

## Context
- **The branch was rebuilt and force-pushed on 2026-11-18.** `feature/mixed-native` is now vanilla 0.2.15 plus the
  mixed split only. The hardened line, with reports 0067 to 0069, is on `archive/hardened-0069`; `.ai/HARDENING.md`
  maps the old commits to the new ones. In the worktree `/home/chris/workspace/Strata-mixed-native`:
  - Check that nothing uncommitted is worth keeping. The untracked `runs68/` and `runs69/` directories stay.
  - Then `git fetch origin && git reset --hard origin/feature/mixed-native`.
  - Do not touch the vanilla checkout or the archive branch.
- **Why the hardening went.** The Human User attributes the 17 November crash to a launch config that mixed components
  from different builds. Rule 7 now says every process runs from one build, and both builds of a mixed config come from
  the same commit.
- **The mixed engine code** is line for line what report 0069 measured, except two hardening-only lines and one
  vanilla-form allocation (`.ai/HARDENING.md`). The live-session resume difference of report 0069 is accepted as
  vanilla behaviour.
- **New since report 0069** (`0b0f671b`, checked by Chat AI without a GPU only):
  - The front stage drops a parked copy when the host's conversation cache lets it go (eviction, superseded entry,
    clear, refused put).
  - It is a `STRATA_HETERO`-guarded `on_drop` hook in vanilla's `ConversationCache` and a new `SessionOp::Drop`.
  - `hetero_tests` and `hetero_cache_drop_test` pass. Vanilla's `conversation_cache_test` passes with and without the
    define.
- **Deferred by the Human User:** limiting the host's expert arena to its own layers.

## Tasks
1. **Before any GPU run:** record the kernel log and core dumps, as in handoff 0069.
2. **Clean builds.** Use new, empty build directories for the host, CUDA and HIP trees (`.ai/PROJECT.md`: CUDA 13.0
   sm_89; gfx1201 with both MMQ options), with `-DSTRATA_BUILD_TESTS=ON -DSTRATA_HETERO=ON` and 28 jobs. Old
   directories may hold objects of the hardened line. Run host CTest, and in the GPU trees the hetero tests plus
   vanilla's own GPU tests.
3. **Launch check.** Add `hetero/tools/check_launch.py CONFIG`, which refuses a config unless all of these hold:
   - every executable it names exists;
   - each executable lies inside a build directory whose `CMakeCache.txt` points at this checkout;
   - a HIP-only config names one build; a mixed config names one HIP build (host) and one CUDA build (front stage);
   - each executable's libraries, from `ldd`, resolve inside its own build directory or a system or toolkit path
     (`/opt/rocm*`, `/usr/local/cuda*`, `/usr/lib*`, `/lib*`), never in another Strata build;
   - both builds were made from the checkout's current `HEAD`: record it at build time, for example in a
     `BUILD_COMMIT` file in each build directory, and compare.

   Wire it into `serve_mixed.py` before the server starts. Show it refuse a config that mixes in an old build
   directory; that's a check without any GPU run.
4. **HIP-only:** the committed `hiponly.example.json` serves, with conversation-cache reuse, as in report 0069.
5. **Mixed:** `hetero/tools/run_server_checks.sh` with the committed mixed config, and the turn-2 token comparison of
   report 0069 (checkpoint and park paths, 50 token IDs).
6. **Drop on real evictions.** Run a mixed server with `--conversation-cache-slots 2`. Run four two-turn conversations
   A, B, C, D, then a turn 3 for each of B and A.
   - A was evicted on the host. Its turn 3 must reread its prompt, with no error, and the engine stays up.
   - D is still parked. A follow-up turn for D must reuse it.
   - Report the front stage's log lines for the drops. The front stage's parked-copy count never exceeds the host's.
     Add one stderr line per Drop if that's needed to see it; that's a change to `generate.cpp` logging only.

## Constraints
- All rules of `.ai/PROJECT.md`. Never kill a GPU process by force. No two GPU jobs at once, except the two processes
  of one mixed run.
- A new GPU fault line, a crash outside the run or a reboot ends the handoff at once.
- Commit in small steps.

## Hardware Test Budget
15 minutes of GPU run time in total; every single run under 2 minutes.

## Completion Criteria
- Clean builds and the tests in task 2 pass.
- `check_launch.py` accepts the committed configs and refuses a mixed-in build.
- HIP-only and mixed serving checks pass, and the turn-2 comparison gives 50 of 50 on the checkpoint and park paths.
- The eviction run in task 6 behaves as described.
- `.ai/reports/report-0070.md` and all changes are committed and pushed to `feature/mixed-native`.
- If anything inside the engine fails: **Partial**, with Tier 1 escalation recommended.

## Reporting
- Build and test results per tree, compared with report 0067 where comparable.
- Long-prompt prefill and decode tok/s for HIP-only and mixed, beside report 0069 (one sample each is fine), and
  available RAM with the engine idle.
- The eviction run: per turn, reused tokens and any front-stage drop line.
- Kernel log and core-dump lines, or "none". GPU run time used. What remains open.
