# Handoff 0071

## Tier
2

A run-and-measure turn, as handoff 0070. Scope the agent may change: build directories, run configs, the drivers and
`check_launch.py` in `hetero/tools/`, and logging. A failure inside the engine or the front stage is reported, with
Tier 1 escalation recommended, not fixed here.

## Objective
Finish the re-check that report 0070 left open: the rest of the GPU tests, HIP-only serving with vision from builds of
this checkout, and a controlled answer to the mixed prefill drop (1,787 against 2,249 tok/s in report 0069).

## Context
- Read `.ai/reports/report-0070.md` and Chat AI's review in `.ai/STATUS.md`. Pull `feature/mixed-native` (no
  force-push this time).
- **The build option is renamed:** `-DSTRATA_MIXED_SPLIT=ON` (was `STRATA_HETERO`), with targets and tests
  `strata_mixed_split`, `mixed_split_tests`, `mixed_split_cache_drop_test`, `mixed_split_front_client`. A build
  directory configured with the old option builds the engine **without** the mixed split, so use new directories.
- **The HIP `conversation_copy_streams_test` failure is vanilla's.**
  - The test and every `conversation_copy*` source are unchanged from vanilla 0.2.15. The hardening had replaced
    this assertion, which is why it passed in report 0067.
  - It measures device-wide free memory. On gfx1201 with ROCm 10.1, the first copy stream and event took 148 MiB,
    against the test's 16 MiB bound and vanilla's 64 MiB restore guard (`kRestoreFirstUseMib`).
  - Chat AI records it as a vanilla finding (an upstream candidate), not a mixed-split defect, and not a reason for
    Tier 1.
- **`check_launch.py` changed:**
  - A build is stale only when the built sources changed since its `BUILD_COMMIT`; a report-only commit no longer
    makes it stale.
  - Vision helpers may come from a build of `tools/vision` in this checkout.
  - The HIP-only example now names `build-vision-hip/strata-vision-hip` and `build-vision-cpu/strata-vision`.
- **The kernel line "Cannot map memory … size of 0x7e9152 pages"** is 31.6 GiB, the size of the shared expert arena.
  It matches the engine's own log of the whole-arena HIP host registration failing before its 48-slice fallback
  succeeds. Not a GPU fault.

## Tasks
1. **Kernel log and core dumps:** record them before the first GPU run and after the last, as in handoff 0070.
2. **Builds** in new directories `build-0071-host`, `build-0071-hip`, `build-0071-cuda`, with `-DSTRATA_BUILD_TESTS=ON
   -DSTRATA_MIXED_SPLIT=ON` and the toolchains of `.ai/PROJECT.md`. Write `BUILD_COMMIT` after each successful build.
3. **Vision helpers** from this checkout: build `tools/vision` as `setup.py` does (`build_vision_hip` for gfx1201,
   `build_vision_cpu`) into `build-vision-hip` and `build-vision-cpu`, and stamp them. Adjust the two paths in
   `hiponly.example.json` if the executables land elsewhere in those directories.
4. **The GPU tests** in both trees, the whole set this time:
   - Run `conversation_copy_streams_test` on HIP once more, alone, and report its number. Then exclude it from the HIP
     run as a known vanilla failure.
   - List every test that fails or skips, with its reason. Run `hip_prefill_mmq_parity`.
5. **HIP-only server** through `serve_mixed.py` with the committed `hiponly.example.json`, which must pass the launch
   check:
   - a two-turn chat with conversation-cache reuse;
   - one request with an image;
   - a clean shutdown.

   Note whether the "Cannot map memory" kernel line appears here too.
6. **Prefill A/B**, mixed, same prompt (the 4,924-token long prompt), MTP and caches as in report 0070. Three samples
   of each, alternating:
   - **A:** the current branch;
   - **B:** the hardened line `archive/hardened-0069`, built in a separate worktree with its own two build
     directories and stamps.

   Run B through `engine_drive.py` directly; `check_launch.py` on this branch does not know that worktree. Each
   process must still run from one build, with both builds from that worktree's `HEAD`. Report prefill tok/s per
   sample, and the median for each line.

## Order and budget
- **Budget:** 15 minutes of GPU run time (within the project's 30-minute allowance). Every single run under 2
  minutes.
- **Priority order:** tasks 4, 5, 6. If the budget runs out, stop and report what was not reached.

## Constraints
- All rules of `.ai/PROJECT.md`. Run evidence goes under `.ai/runs/0071-*`. Never kill a GPU process by force. No two
  GPU jobs at once, except the two processes of one mixed run.
- A new GPU fault line, a crash outside the run or a reboot ends the handoff at once.
- Do not change the archive branch: build it in a worktree, never commit there.

## Completion Criteria
- Builds pass. The GPU test lists are complete, and the only failure is the known vanilla one, or each new failure is
  reported.
- HIP-only serves through the launch check, with reuse and an image request.
- The prefill A/B has three samples each.
- `.ai/reports/report-0071.md` and all changes are committed and pushed to `feature/mixed-native`.
- If anything inside the engine fails: **Partial**, with Tier 1 escalation recommended.

## Reporting
- Test results per tree, with failures and skips listed.
- The HIP-only server results: reused tokens, the image request, prefill and decode tok/s beside report 0070.
- The prefill A/B table, and one line on what it says: whether the difference follows the hardening or is noise.
- Kernel log and core-dump lines, or "none". GPU run time used. What remains open.
