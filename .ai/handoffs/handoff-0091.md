# Handoff 0091

## Tier
2

Build and measure. Scope the agent may change: build directories, run configs (build paths only), a small launch
wrapper for the vanilla run (task 4), small fixes to `agent_session.py` / `agent_table.py` (say each), and `.ai/`
notes. No engine changes; a failure in the engine is reported, not fixed.

## Objective
Report 0090's three findings, one task each:
1. Confirm Chat AI's fix of the mixed error ("no checkpoint at 9956 tokens is kept here").
2. Find out whether the HIP-only non-finite logits are vanilla's (#879) or come from the carried pinned-readback patch.
3. Account for the mixed split's extra host memory (report 0090: +3.4 GiB swap, 26 s of GPU pauses, 7 workqueue
   warnings in mixed; none of that in HIP-only).

## Context
- Read `.ai/PROJECT.md`, `.ai/STATUS.md` and report 0090.
- **The fix** (`8819f09b`, Chat AI, CPU-tested): the front stage evicted checkpoint parts by plain LRU across all
  conversations; the host pins its checkpoint chain's root (the 9956-token system prompt and tools) and keeps it.
  The front stage now never evicts the live chain's root. `test_session_root_kept` fails on the old code and passes.
- **Non-finite logits in HIP-only:** vanilla has a guard for them (#879, and #3465 for the windows that read a prompt's
  last tokens): the request ends without the bad tokens, every cached state is dropped, and the prompt is read again.
  Report 0090's case was a prompt window at position 74438 in request 360. The crash on 19 November also logged one in
  a prompt window. Whether the carried patch plays a part is unknown: no unpatched run has been made.
- The driver's sampler now records the memory breakdown (`Shmem`, `Mlocked`, `AnonPages`; per process `VmSwap`,
  `VmLck`, `VmPin`, `RssAnon`, `RssFile`, `RssShmem`).
- All three runs below use the RC defaults, seed 85, the same request sequence as report 0090 (`agent_session.py run
  --max-requests 361`, `STRATA_IQ_MT_MIN=1`), with the log watcher of report 0090.

## Tasks
1. Safety checks; `free -m` and `swapon --show` at the start, between runs and at the end. First rule.
2. **Builds** at the current commit (`build-0091-*`, as in handoff 0087). `mixed_split_tests` in both trees. Point the
   mirrors at them.
3. **Mixed, requests 1 to 361**, against the mixed mirror. Expected: request 251 succeeds now. Stop on a `non-finite
   logits` line or a kernel stop-pattern line (cooperatively, as in report 0090), and report.
4. **HIP-only, unpatched vanilla, requests 1 to 361:** build plain vanilla 0.2.15 (`c775728`, a separate worktree, the
   mixed-split option off, the production HIP flags) and serve the production mirror's arguments from that build and
   that worktree's own `serve/server.py`. `check_launch.py` cannot check another commit: start it through a small
   Python wrapper that calls `reset_ignored_signals()` and `force_sdma_off()` and stops it with `stop_cooperatively`.
   Do not stop on `non-finite logits` here: record each with its request number and position, and go on to request 361.
5. **HIP-only, this branch (with the patch), requests 1 to 361:** the production mirror on `build-0091-hip`, the same
   way, recording `non-finite logits` lines without stopping. (Report 0090 had one at request 360.)
6. Table: per run, every `non-finite logits` line (request, position, window kind); swap first and max; the GPU pause
   lines (count, total ms, max); the workqueue lines; and, from the samples at matched request numbers (100, 200,
   300, 361), each process's `RssAnon`, `RssFile`, `RssShmem`, `VmLck`, `VmPin`, `VmSwap`, plus the system's `Shmem`,
   `Mlocked`, `AnonPages`. One line per run on what differs most between mixed and HIP-only.

## Hardware Test Budget
**About 1.5 hours of GPU run time, over the 30-minute allowance** (announced to the Human User; allowed unless refused).
Each run takes about 25 minutes. Order: tasks 3, 4, 5. If time runs out, stop cooperatively and report.

## Constraints
- **First rule: `HSA_ENABLE_SDMA=0` in every run**, the vanilla run included. All rules of `.ai/PROJECT.md`.
- Kernel stop-pattern lines end any run. `non-finite logits` ends the mixed run (task 3) but not the HIP-only runs
  (tasks 4, 5), which are there to reproduce it.
- Do not commit in the vanilla worktree. Evidence under `.ai/runs/0091-*`.

## Completion Criteria
- Builds and tests pass; the three runs done (or stopped and reported); the table in the report.
- `.ai/reports/report-0091.md` and all changes are committed and pushed to `feature/mixed-native`.

## Reporting
- Task 3: did request 251 pass, and how far did the run get.
- Tasks 4 and 5: the non-finite lines side by side. Does it happen without the patch, at the same request?
- The memory table and the one-line differences.
- Safety lines, swap, GPU time.
