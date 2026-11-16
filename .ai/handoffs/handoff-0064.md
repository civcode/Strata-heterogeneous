# Handoff 0064

## Tier
2

## Objective
Finish what handoff 0063 could not: build the CUDA tree with the new CUDA 13.0 toolkit and record the vanilla
HIP-only baseline.

## Context
- Read `.ai/PROJECT.md` first; it now names the CUDA 13.0 toolkit and the known-good vanilla run config.
- Report 0063: the host and HIP trees build, the seam's CPU tests pass, and the shared hand-off probe passed in all
  three pairings. The CUDA engine did not compile with nvcc 12.0. The baseline was not run because handoff 0063
  described the vanilla config with the wrong context size.
- Continue in the worktree report 0063 used (`/home/chris/workspace/Strata-mixed-native`), with its `.venv` and its
  host and HIP build directories.
- No code has changed since report 0063.

## Tasks
1. **CUDA tree.** Configure a fresh CUDA build directory as report 0063 did, with the compiler and architecture
   that `.ai/PROJECT.md` gives, and build it with 28 jobs. Run `ctest -R hetero_tests` in it.
2. **Probe with the new build.** Run the shared hand-off probe once: CUDA writer (the CUDA 13.0 build) with HIP
   reader on device 0. Expected: both sides end with `ok, 200 windows`.
3. **Vanilla HIP-only baseline.** Copy `/home/chris/workspace/Strata/strata-q2_0.json` into the worktree as it is
   (context 196608) and change **only** `exe`, to this branch's HIP build of `strata`. Start the server the way
   `run-q2_0.sh` starts vanilla, wait until it is ready, send the same short greedy chat request twice (about 200
   new tokens; the first is the cold one), then shut down cleanly.

## Constraints
- You may change build configuration and the copied run config's `exe`. Nothing else.
- Do not change vanilla engine, kernel, server or Terminal files. If the CUDA tree still does not build with CUDA
  13.0, report the first error and stop that task with `Tier escalation recommended`.
- If the copied config names files that this branch's engine refuses (for example a pack or draft layer of an older
  format), report the refusal text; do not convert or download anything.
- GPU processes are never killed by force.

## Hardware Test Budget
5 minutes in total for tasks 2 and 3. Builds and CPU tests do not count.

## Completion Criteria
- The CUDA tree builds with CUDA 13.0 and `hetero_tests` passes in it.
- The probe pairing has been run with the new CUDA build.
- One HIP-only vanilla server start, two requests and a clean shutdown have been done.
- `.ai/reports/report-0064.md` is committed and pushed to `feature/mixed-native`.

## Reporting
- CUDA: configure options, nvcc version, build result.
- Probe: the printed lines.
- Baseline: which engine and drafting mode vanilla chose; the startup lines that show the layer split and any
  architecture defaults applied; for each of the two requests the prompt tokens, completion tokens, time to first
  token, prefill tokens per second and decode tokens per second, as the server reports them; free memory per card
  while loaded.
- The engine arguments and `env` of the copied config, so later mixed runs can use the same settings.
