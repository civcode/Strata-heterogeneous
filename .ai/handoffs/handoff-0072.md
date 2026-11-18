# Handoff 0072

## Tier
2

A run-and-measure turn, as handoff 0071. Scope the agent may change: build directories, run configs, the drivers in
`hetero/tools/`, and logging. A failure inside the engine or the front stage is reported, with Tier 1 escalation
recommended, not fixed here.

## Objective
Finish what report 0071 did not reach, in this order: HIP-only serving with vision, the prefill A/B against the
hardened line, and the remaining tests.

## Context
- Read `.ai/reports/report-0071.md`. Chat AI accepts its results:
  - HIP: 272 passed, 15 skipped. The two failures are the known vanilla `conversation_copy_streams_test` and
    `ple_parity`, which only missed its model file (a relative path).
  - CUDA: 170 passed, 2 skipped, out of 172 run.
- The builds of report 0071 (`build-0071-*`, `build-vision-*`) are current. No source changed since their stamp, so
  `check_launch.py` still accepts them. Do not rebuild unless it refuses.
- Budget: count GPU time per run, not per batch, and check the remaining budget before starting each run.

## Tasks, in priority order
1. **Kernel log and core dumps:** record them before the first GPU run and after the last, as in handoff 0071.
2. **HIP-only server** through `serve_mixed.py` with the committed `hiponly.example.json`:
   - a two-turn chat with conversation-cache reuse;
   - one request with an image;
   - the long-prompt request of report 0070, for prefill and decode tok/s;
   - a clean shutdown.

   Note whether a "Cannot map memory" kernel line appears, and its size.
3. **Prefill A/B**, mixed, the 4,924-token long prompt, MTP and caches as in report 0070. Three samples of each,
   alternating A, B, A, B, A, B:
   - **A:** the current branch, through `engine_drive.py` with this branch's mixed config;
   - **B:** `archive/hardened-0069`, built in a separate worktree (two build directories, one HIP and one CUDA, both
     from that worktree's `HEAD`), run through `engine_drive.py` from that worktree.

   Never commit in that worktree. Building it costs no GPU time. Report prefill and decode tok/s per sample, and the
   median for each line.
4. **`ple_parity`** in both trees, with its model path given so that it finds the PLE shard.
5. **The remaining CUDA tests**, as far as the budget allows. List the ones not run.

## Hardware Test Budget
25 minutes of GPU run time (within the project's 30-minute allowance). Every single run under 2 minutes. Tasks 2 and
3 come first and should take about 12 minutes together.

## Constraints
- All rules of `.ai/PROJECT.md`. Run evidence goes under `.ai/runs/0072-*`. Never kill a GPU process by force. No two
  GPU jobs at once, except the two processes of one mixed run.
- A new GPU fault line, a crash outside the run or a reboot ends the handoff at once.
- Do not change the archive branch.

## Completion Criteria
- HIP-only serves through the launch check, with reuse and an image request.
- The prefill A/B has three samples each.
- `ple_parity` was run with its model in both trees.
- `.ai/reports/report-0072.md` and all changes are committed and pushed to `feature/mixed-native`.
- Tests not reached in task 5 do not make the turn Partial; list them.

## Reporting
- The HIP-only results beside report 0070: reused tokens, the image request, prefill and decode tok/s.
- The prefill A/B table, and one line: whether the difference follows the hardening or is within the spread.
- `ple_parity` per tree. The CUDA tests run, and the ones not run.
- Kernel log and core-dump lines, or "none". GPU run time used, counted per run. What remains open.
