# Handoff 0084

## Tier
2

Build, check and measure on code that is complete. Scope the agent may change: build directories, run configs, the
drivers in `mixed_split/tools/`, and `.ai/` notes it is asked to write. A failure inside the engine or the front stage
is reported, with Tier 1 escalation recommended, not fixed here.

## Objective
1. Confirm the rename to `mixed_split` on the cards.
2. Finish what report 0083 left open: all-target builds and the audit of the production options on 0.2.16.
3. Find out whether vanilla 0.2.16 reads prompts slower than 0.2.15 on HIP-only, which matters for the Human User's
   production upgrade.

## Context
- Read `.ai/PROJECT.md` (new: base `v0.2.16`, the 0.2.16 layout, the new names), `.ai/STATUS.md` and report 0083.
- **Since report 0083, by Chat AI** (`740e2bea`):
  - **The rename:** `hetero/` became `mixed_split/`, `strata::hetero` became `strata::mixed_split`, and the headers
    moved to `strata/mixed_split/`. In the guarded engine code: the host object `mixed_host`, `MixedSplit*` types,
    `mixed_*` variables. Shared-memory names: `strata-mixed-split`.
  - **Checked by Chat AI without a GPU:** the library and CPU tests; `check_off_is_vanilla.py` 6 of 6 against
    `v0.2.16`; a syntax check of the engine with the option on (nothing beyond vanilla 0.2.16's own warnings); the
    Python tools.
  - **The builds of report 0083 are stale:** their source paths moved. Configure new ones.
- **Report 0083 left open:**
  - builds of every target in the HIP and CUDA trees (only selected targets were built);
  - a complete check that every argument and environment variable of the two mirrors exists and means the same in
    0.2.16.
- **Report 0083's orientation numbers:** HIP-only on 0.2.16 read prompts at 1,975 tok/s (long) and 3,019 (16K),
  against 2,372 and about 3,500 in report 0078 on 0.2.15. Those runs differ in conditions, so nothing is established.
  The archive branch `archive/mixed-native-0.2.15` is vanilla 0.2.15 plus the mixed split; with the option off, its
  HIP build is vanilla 0.2.15.

## Tasks
1. **Safety checks** (`.ai/PROJECT.md`): the boot ID, kernel log and core dumps, before the first GPU run, after the
   last, and after any pause.
2. **Builds** at the current commit, in new directories, with `-DSTRATA_MIXED_SPLIT=ON -DSTRATA_BUILD_TESTS=ON` and 28
   jobs, every target: host, CUDA (13.0, sm_89) and HIP (gfx1201, both MMQ options). Also both vision helpers. Stamp
   each.
   - Run host CTest in full, and `mixed_split_tests` / `mixed_split_cache_drop_test` in every tree.
   - In the GPU trees, run vanilla's GPU tests for at most 8 minutes of GPU time. List the tests run, failed and
     skipped. Known vanilla failures (`.ai/DIVERGENCE.md`) are named, not chased.
3. **The rename on the cards:**
   - the mixed mirror serves through the launch check: one request, a two-turn conversation over 8,192 tokens (parked
     and reused), a vision request, and a clean shutdown;
   - pipelined against serial exactness on the code prompt with `STRATA_IQ_MT_MIN=1`.
4. **The option audit.** For every argument and environment variable in `hiponly.example.json` and
   `mixed.example.json`, check 0.2.16's sources and docs: it exists, it means what the mirror assumes, and its default
   did not change in a way that matters. Write the table into the report. Adjust a mirror only where 0.2.16 requires
   it, and say so.
5. **Vanilla 0.2.15 against 0.2.16, HIP-only, prompt reading:**
   - Build the HIP tree of `archive/mixed-native-0.2.15` in a separate worktree, with the option **off** (that build is
     vanilla 0.2.15), and the current tree with the option off too. Building costs no GPU time.
   - Run each with the production mirror's arguments, from its own build. Paths follow each version's layout: 0.2.15's
     server is `serve/server.py`; use `run_engine.py` / `engine_drive.py` directly if the server path makes it awkward.
   - Untraced, warm requests in blocks as in report 0082 (order A B B A A B; A = 0.2.15, B = 0.2.16), the long and the
     16K prompt: prefill and decode tok/s, six measured samples per version and prompt.

## Hardware Test Budget
30 minutes of GPU run time (the project's allowance). Every single run under 2 minutes. Building costs no GPU time.
Priority order: tasks 3, 5, then the GPU tests of task 2.

## Constraints
- All rules of `.ai/PROJECT.md`. Run evidence goes under `.ai/runs/0084-*`. Never kill a GPU process by force. No two
  GPU jobs at once, except the two processes of one mixed run.
- Do not commit in the archive worktree.

## Completion Criteria
- All builds pass, with the tests as listed.
- The rename checks of task 3 pass.
- The option table is complete.
- The 0.2.15 against 0.2.16 table has six samples per version and prompt.
- `.ai/reports/report-0084.md` and all changes are committed and pushed to `feature/mixed-native`.

## Reporting
- Build and test results per tree.
- The rename checks.
- The option audit table.
- The version table, with one line: does 0.2.16 read prompts slower than 0.2.15 on HIP-only, beyond the spread?
- Safety lines (boot ID, kernel log, core dumps), or "none". GPU run time used. What remains open.
