# Handoff 0083

## Tier
1

Re-pin the mixed split from vanilla 0.2.15 to vanilla 0.2.16. Vanilla restructured its repository and changed the
files our seam lives in. Every guarded addition has to land in the right place of the new code, and the result is
judged only by builds and runs on the cards.

## Objective
`feature/mixed-native` is vanilla 0.2.16 plus the mixed split. With `STRATA_MIXED_SPLIT` off, it is vanilla 0.2.16
line for line; with it on, everything reports 0068 to 0082 established still holds.

## Context
- Read `.ai/PROJECT.md`, `.ai/STATUS.md` and `.ai/DIVERGENCE.md`.
- **New base:** vanilla tag `v0.2.16`, commit `64524fb9` (18 November 2026, 864 commits after 0.2.15's `c775728`). Fetch
  it from `https://github.com/Niko1221/Strata.git`. The Human User decided the re-pin on 2026-11-19.
- **The layout moved** (vanilla commit `4058aa78`):
  - the engine is under `engine/`: `engine/src`, `engine/include`, `engine/CMakeLists.txt`;
  - builds are `cmake -S engine -B …` (`docs/AMD_HIP.md`);
  - the Terminal is under `terminal/`: `terminal/serve/server.py`, `terminal/setup.py`, `terminal/tools/` (with
    `calibrate.py`, `iq_pack.py`, `strata_pack.py`, `vision/`);
  - repository tools are under `tools/`.
- **Our seam files changed in vanilla:** `generate.cpp` +1,337 / −141, `verify.cpp` +174 / −37 (among others a fork
  that runs after the doorbell is published: `STRATA_SH_FORK_LATE`), `conversation_cache.hpp` +33 / −1, `verify.hpp`
  +12, `prefill.cpp` +28 / −1. `prefill.hpp` is unchanged.
- **Not related:** `engine/src/split` (`STRATA_BUILD_SPLIT`, off by default) is sFNX's separate split-every-layer
  engine. It only moved in 0.2.16 and has nothing to do with our seam.
- **Our changes:**
  - only insertions in 7 vanilla files, all inside `#if defined(STRATA_MIXED_SPLIT)` except the option itself in
    `CMakeLists.txt`;
  - the `hetero/` subtree (library, tests, tools, configs);
  - `.ai/`.

  `hetero/tools/check_off_is_vanilla.py` proves the off build is vanilla; its paths and default reference must follow
  the move.

## How to do it (decided)
- **Archive first:** push the current head as branch `archive/mixed-native-0.2.15`.
- **Merge `v0.2.16` into `feature/mixed-native`** (no force-push), letting git's rename detection carry our insertions
  to the new paths. Resolve every conflict so that:
  - vanilla 0.2.16's text stays as it is;
  - each guarded block sits where it does the same job in the new code.

  If vanilla moved or rewrote the code a hook belongs to, find where that job happens now. Never restore 0.2.15 code
  to make a hook fit.
- **`hetero/`** stays at the top level as our own subtree. `engine/CMakeLists.txt` takes it in behind the option (for
  example `add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../hetero ${CMAKE_BINARY_DIR}/hetero)`). Move it under
  `engine/` only if that does not work, and say why.
- **Follow the move in our tools:**
  - `check_off_is_vanilla.py`: compare `engine/src` and `engine/include` against `v0.2.16` by default;
  - `check_launch.py`: the build's source directory is the checkout's `engine/`; vision helpers come from
    `terminal/tools/vision`;
  - `serve_mixed.py` and the other drivers: `terminal/serve/server.py`;
  - `hetero/configs`: the paths.
- **`.ai/PROJECT.md`:** the reference commit, the build commands (`-S engine`), and anything else the move changes.
  Keep the history in `.ai/` as it is.

## Tasks
1. **Safety checks** (`.ai/PROJECT.md`): the boot ID, kernel log and core dumps, before the first GPU run, after the
   last, and after any pause.
2. **The merge** as above, then:
   - `check_off_is_vanilla.py` against `v0.2.16`: every changed vanilla file is vanilla with the option off;
   - `.ai/DIVERGENCE.md`: rewritten for the new paths and line counts. List any hook that changed place or shape, and
     why.
3. **Builds,** in new directories, all with `-DSTRATA_MIXED_SPLIT=ON` and `STRATA_BUILD_TESTS`: host, CUDA (13.0,
   sm_89), HIP (gfx1201, both MMQ options), and both vision helpers. Write `BUILD_COMMIT` after each. Run
   `mixed_split_tests` and `mixed_split_cache_drop_test` in every tree.
4. **The production mirrors on 0.2.16:**
   - check that every argument and environment variable in `hiponly.example.json` and `mixed.example.json` still
     exists and means the same in 0.2.16 (the engine refuses unknown options). Report renamed, removed or changed
     defaults, and adjust the mirrors only where 0.2.16 requires it;
   - both serve through the launch check: one request each, with the engines' start lines.
5. **Exactness,** mixed, `STRATA_IQ_MT_MIN=1`, greedy, the code and the long prompt:
   - pipelined against serial: identical token IDs, also with `STRATA_PIPELINE_FORCE_MISS=3`;
   - the prompt-chunk read-ahead on against off (`STRATA_MIXED_PF_AHEAD=0`) on the 16K prompt: identical token IDs.
6. **Server suites** (`serve_session.py`) with the mixed mirror:
   - streaming, cancel, and the cancel suite of report 0079;
   - a two-turn conversation over 8,192 tokens, parked and reused;
   - a vision request, `cvec=0`, and a clean shutdown.
7. **Performance,** untraced, three samples each, alternating: mixed against production (HIP-only mirror), on the code,
   the long and the 16K prompt, on 0.2.16. Prefill and decode tok/s. Put reports 0078 and 0082 beside it for
   orientation; they used other vanilla code.

## Boundaries you decide within
- The contract holds: vanilla 0.2.16 wherever possible, and our code only for the CUDA/HIP seam, guarded.
- No new feature in this turn. A hook that cannot be placed in 0.2.16 without new logic is reported, with the feature
  it serves refused in mixed mode for now, not forced.
- Every safety gate stays:
  - the identity check;
  - publish after the front stage's commit acknowledgement;
  - the bounded waits, with `release_gpu_waits` on an error;
  - cooperative shutdown.

## Hardware Test Budget
30 minutes of GPU run time (the project's allowance). Every single run under 2 minutes. Building costs no GPU time.
Priority order: tasks 4, 5, 6, 7.

## Constraints
- All rules of `.ai/PROJECT.md`. Run evidence goes under `.ai/runs/0083-*`. Never kill a GPU process by force. No two
  GPU jobs at once, except the two processes of one mixed run.
- Do not touch the vanilla checkout or the archive branches.
- Commit in steps: the merge, the tool updates, the config changes, the report.

## Completion Criteria
- The merge is pushed; `archive/mixed-native-0.2.15` exists.
- `check_off_is_vanilla.py` passes against `v0.2.16`.
- All builds and the CPU tests pass. Both mirrors serve.
- The exactness checks give identical token IDs. The server suites pass. The performance table is complete.
- `.ai/DIVERGENCE.md` is current; `.ai/reports/report-0083.md` and all changes are committed and pushed to
  `feature/mixed-native`.
- If stopped early: **Partial**, with the last step that works. A hook that cannot be placed is reported, not forced.

## Reporting
- The merge: conflicts per file and how each was resolved; hooks that moved or changed shape.
- Options or variables of the mirrors that 0.2.16 changed.
- Build and test results; the exactness results; the server-suite results; the performance table.
- Safety lines (boot ID, kernel log, core dumps), or "none". GPU run time used. What remains open.
