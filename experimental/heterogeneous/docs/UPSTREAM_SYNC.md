# Upstream synchronization — manual, selective and milestone-boundary

**Working branch:** `feature/heterogeneous-cuda-window-admission`.
**Maintainers:** `Niko1221/Strata`, branch `main`.
**Rule:** compare on every meaningful ChatGPT turn; integrate only when a
relevant change is understood and locally validated. No automatic merge,
rebase, reset, branch deletion, fork-main fast-forward or GitHub Actions.
Do not incorporate upstream `.github/workflows/*` without checking and
removing automatic triggers; existing fork workflows are manual-only.

## Current checkpoint (2026-11-13)

GitHub comparison: upstream `13eedf552417a6a791a7d13c6597e24300ee3a09`
versus our M1.18 `6b3b6be30a81d7d35d3b577e1f13b3c5c31aebfe`:
**379 upstream-only commits, 25 branch-only commits**, merge base
`146df3c468739147f520ac3816de806f71232571`.
The fork's `main` was 379 commits behind upstream and had zero unique
commits. **No upstream commits have been integrated by this checkpoint.**
The JSON snapshot at `.ai/upstream/checkpoint.json` is an *observation*
cursor, not a claim that 379 upstream commits were individually reviewed.

Initial impact assessment:

| Upstream area | Observation | Decision |
| --- | --- | --- |
| `CMakeLists.txt`, `cmake/backends/*`, `cmake/strata_gpu_engine.cmake` | Upstream split its large top-level engine build into included modules. Our M1.18 CUDA worker links through the old engine target layout. | **Review before M2 native HIP build.** Do not wholesale import the CMake refactor. |
| `include/strata/core/serve_stop_bound.hpp`, `src/program/generate.cpp` | Upstream's latest commit `13eedf55` applies accepted-output stop bounds on CUDA as well as HIP and supports `STRATA_SERVE_STOP_BOUND=0`. | **Defer backport.** This bounds the *serve/generation session* commit; our v4 heterogeneous stage admission and commit protocol are not that call site. Revisit when wiring end-to-end generation and EOS handling. |
| `.github/workflows/*` | Upstream added automatic `push` / `pull_request` checks. | **Exclude.** Never introduce event-triggered Actions without explicit authorization. |
| Other upstream runtime, app, CLI, docs and tests | Upstream is 379 commits ahead; this review was targeted, not exhaustive. | Review touched files when they become relevant to a specific heterogeneous milestone. |

## Repeatable read-only comparison

From the permitted Oberon project worktree only:

```sh
cd /home/chris/workspace/Strata-heterogeneous
git status --short
# Check whether the upstream remote exists; only add it if missing:
git remote get-url upstream
# If missing, configure this once (does not change a worktree or branch):
git remote add upstream https://github.com/Niko1221/Strata.git
# Explicit fetch only updates refs/remotes/upstream/main, never local branches.
python3 tools/heterogeneous_upstream_sync.py --fetch --all-upstream
# Subsequent calls compare against the last observed SHA in checkpoint.json:
python3 tools/heterogeneous_upstream_sync.py --fetch
# Full machine-readable path classification:
python3 tools/heterogeneous_upstream_sync.py --json --all-upstream
python3 -m unittest discover -s tools -p 'test_heterogeneous_upstream_sync.py'
```

The tool refuses an unexpected upstream remote URL, reports exact SHAs,
merge base, ahead/behind and categorized changed paths. It does not modify
the worktree, write a checkpoint or perform a merge. `--fetch` is an explicit
opt-in that updates only an upstream-tracking ref. If the observed SHA is no
longer an ancestor of upstream, it stops rather than fabricating a diff.

## Safe integration process

1. **Assess before implementing a milestone.** Capture the two heads and
   changed paths with the tool. Check directly relevant upstream code and
   APIs. Keep the observation SHA separate from the *integrated* SHA.
2. **Choose a coherent change.** For isolated fixes, manually backport the
   minimal adjustment or cherry-pick an exact upstream commit when its
   dependencies are understood. For structural integrations, use a temporary
   branch based on the last validated feature commit. Do not blindly merge
   379 commits or rebase established M1 milestones.
3. **Respect original build isolation.** Keep CUDA/HIP executables separate,
   respect experimental opt-in flags and process boundaries. Reject any
   import that silently changes default engine behavior or enables Actions.
4. **Validate in the active developer's environment.** Run targeted CPU
   tests first; batch CUDA/HIP native link and hardware checks into an
   Oberon handoff only when necessary. No GPU runtime without authorization.
5. **Promote only validated changes.** Fast-forward the feature branch to
   an isolated integration result when safe, without force push. Record
   exact upstream commits imported (or "none"), tests and remaining gaps.
   The Chat AI has exclusive write ownership until a numbered handoff.
6. **Advance the observation cursor consciously.** Update
   `.ai/upstream/checkpoint.json` after reviewing a new upstream head,
   not after merely fetching. Do not label the observation as integrated.

Oberon isolation still applies: do not access `/home/chris/workspace/Strata-dev`
or `/home/chris/tmp/strata-dev`. Use `--parallel 28` for applicable native
builds. Preserve existing stash and probe. No GitHub Actions execution.
