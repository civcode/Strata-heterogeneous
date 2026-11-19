# Handoff 0092

## Tier
2

Housekeeping, builds and a short check. Scope the agent may change: git worktrees on Oberon (as task 2 says), build
directories, the example configs' build paths, and `.ai/` notes. No engine or tool code changes.

## Objective
1. One worktree for this project: `/home/chris/workspace/Strata-heterogeneous`, the only one on `feature/mixed-native`.
2. Fresh builds there with the front-stage fix (`8819f09b`).
3. A mixed-split run config and start script for Qwen3.8-Flash-Next **IQ3_XXS**, in vanilla's own form, for the Human
   User's real agent run, checked once with a short start.

Handoff 0091 is deferred (Human User, 2026-11-19); do not run it.

**Resumed after report 0092 (Human User's decisions, 21:22):**
- **Work outside `Strata-heterogeneous` is authorized for task 2:** removing the three other worktrees as described
  there, and writing `/home/chris/workspace/strata-local-archive/`. (The earlier paste prompt's "work only in
  Strata-heterogeneous" meant: build, generate and run there. Chat AI's wording conflicted with task 2.)
- **`tools/heterogeneous_pool_probe.cpp`** (untracked, in `Strata-heterogeneous`): move it to
  `/home/chris/workspace/strata-local-archive/Strata-heterogeneous/tools/heterogeneous_pool_probe.cpp`, then continue
  with task 3. The untracked run artifacts under `.ai/runs/` stay where they are (local evidence, not sources).
- Note: `Strata-mixed-native` now shows HEAD `0976fe41` (the pull in the kept worktree moved the shared branch), so its
  files lag its HEAD and `git status` there shows changes that are only that lag. Do not commit, reset or check out
  anything in it: archive its untracked non-build files as task 2 says, then `git worktree remove --force` it.

## Context
- Read `.ai/PROJECT.md` (the first rule; starting and stopping GPU processes) and `.ai/STATUS.md`.
- `git worktree list` on Oberon (Human User, 21:07) shows four worktrees, two of them on `feature/mixed-native` (git
  allows that only with `--force`; a pull in one leaves the other looking modified):

  | Path | HEAD |
  | --- | --- |
  | `/home/chris/workspace/Strata-heterogeneous` | `7eda91f4` `[feature/mixed-native]` (keep) |
  | `/home/chris/workspace/Strata-0084-archive` | `7d573bee` (detached; the 0.2.15 archive head) |
  | `/home/chris/workspace/Strata-hardened-0072` | `b099f46f` `[archive/hardened-0069]` |
  | `/home/chris/workspace/Strata-mixed-native` | `7eda91f4` `[feature/mixed-native]` |

  Both archive lines are on GitHub as branches (`archive/mixed-native-0.2.15` = `7d573bee`, `archive/hardened-0069`).
- **`mixed_split/tools/make_mixed_config.py`** (Chat AI, tested on a stand-in config; read its docstring): from a
  vanilla `strata-<model>.json` it writes `strata-<name>.json` with exactly the vanilla config's keys (only setup's
  `args_written` / `user_args` bookkeeping left out) and `run-<name>.sh` in vanilla `setup.py`'s `write_run_script`
  form. The mixed split comes in through vanilla's own keys: `exe`, `args` (`--front-exe`, `--front-layers`), `env`
  (`HSA_ENABLE_SDMA=0`), `vision`. It runs `check_launch.py` on the result and writes nothing if that fails. Both files
  are git-ignored, as vanilla's are.

## Tasks
1. **Before anything:** the boot ID, kernel log and core dumps (`.ai/PROJECT.md`); `free -m`, `swapon --show`.
2. **Worktrees** (no GPU). For each worktree other than `/home/chris/workspace/Strata-heterogeneous`:
   - record its HEAD and `git status --porcelain --ignored`;
   - if it has **modified tracked files**: stop this task for that worktree and report them (do not remove it).
     Exception: `Strata-mixed-native` (see above);
   - copy every untracked or ignored file that is **not** a build product (`build*/`, `.venv/`, `__pycache__/`,
     `*.o`) into `/home/chris/workspace/strata-local-archive/<worktree name>/`, keeping the paths (local run logs
     under `.ai/runs/`, untracked sources such as `tools/heterogeneous_pool_probe.cpp`); list what was copied;
   - then `git worktree remove` it (`--force` only after the copy, for the ignored build directories).

   Then `git worktree prune` and `git worktree list` (one line expected). Delete no local branch.
3. **The kept worktree:** `git status` (the Human User's prompt shows a pending change: report what it is; if it is a
   source file, report it and stop before building); `git pull --ff-only` to `origin/feature/mixed-native`; check
   that `.venv/bin/python` exists (if not, create it: `python3 -m venv .venv && .venv/bin/pip install -r
   requirements.txt`, as vanilla's `setup.sh` does).
4. **Builds** at that commit, `build-0092-*`, as in handoff 0087 step 2 (CUDA 13.0 sm_89; HIP gfx1201 with
   `-DSTRATA_PREFILL_MMQ=ON -DSTRATA_MMQ=ON`; both vision helpers; `-DSTRATA_MIXED_SPLIT=ON`; 28 jobs). Stamp each
   `BUILD_COMMIT`. `mixed_split_tests` (including `test_session_root_kept`) in both trees through
   `ctest_sdma_off.sh`. Point the example configs in `mixed_split/configs/` at `build-0092-*`.
5. **The IQ3_XXS files.** The model's files on Oberon (Human User, 21:17):
   - shards: `/home/chris/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf`
     and `...-00002-of-00002.gguf` (both with `.done` marks);
   - pack: `/home/chris/workspace/Strata-data/packs/iq3_xxs` (a native pack: `dense.bin`, `index.txt`,
     `native_experts.txt`, `tokenizer/`; no `experts.bin`);
   - shared with Q2_0: the drafter `/home/chris/workspace/Strata-data/mtp/rt`, the vision projector
     `/home/chris/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/mmproj-Qwen3.8-Flash-Next-BF16.gguf`.

   **The tool's input:** vanilla's IQ3_XXS run config in `/home/chris/workspace/Strata` if there is one (read only;
   `strata-iq3_xxs*.json`; report its name and args). If there is none: copy vanilla's production Q2_0 run config
   (`strata-q2_0.json` there) to `/tmp/strata-iq3_xxs.json` and change in the copy only the model's own fields, to the
   files above: `--pack` -> `packs/iq3_xxs`, `--native` -> the first IQ3_XXS shard, `--ple-gguf` -> the second,
   `"tokenizer"` -> `packs/iq3_xxs/tokenizer`, `"model_name"` -> `qwen3.8-flash-next-iq3_xxs`, `"vision"."model"` ->
   the first IQ3_XXS shard. Nothing else; report the diff against the Q2_0 file.

   Then `.venv/bin/python mixed_split/tools/make_mixed_config.py <the input> --build-tag 0092 --name iq3_xxs-mixed`.
   Report the two files' paths, the tool's notes, and the generated config in full.
6. **A short start** (GPU, under 10 minutes): start `run-iq3_xxs-mixed.sh` from a Python parent that calls
   `mixed_split_common.reset_ignored_signals()` first; wait for the server; send one short chat request and one of
   about 8K tokens; stop it with `stop_cooperatively`. Report the server log's start lines (both stages, the layer
   split, the expert cache), the two requests' speeds, `free -m` before and after, and any record- or stop-pattern
   kernel line and `paused the GPU` line. Start nothing else afterwards: the next run is the Human User's.

## Hardware Test Budget
10 minutes of GPU run time (task 6 only).

## Constraints
- **First rule: `HSA_ENABLE_SDMA=0` in every run.** All rules of `.ai/PROJECT.md`.
- `/home/chris/workspace/Strata` (vanilla) is read only. Delete nothing outside the three worktrees being removed;
  never remove the kept one.
- Evidence under `.ai/runs/0092-*`.

## Completion Criteria
- One worktree left; the archive folder listed; builds and tests pass; the IQ3_XXS files written and the short start
  done (or the reason it stopped).
- `.ai/reports/report-0092.md` and all changes committed and pushed to `feature/mixed-native`.

## Reporting
- Per removed worktree: its HEAD, what was archived, anything not removed and why.
- Builds and tests; the IQ3_XXS config in full and the run script; the short start's results.
- Safety lines, swap before and after, GPU time.
