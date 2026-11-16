# Handoff 0063

## Tier
2

## Objective
Build the new working branch on Oberon, answer one question about shared memory with a ready-made probe, and record
how fast vanilla Strata 0.2.15 runs on the two AMD cards alone.

## Context
- Read `.ai/PROJECT.md` first. The design is in `.ai/SEAM.md`.
- `feature/mixed-native` is vanilla Strata 0.2.15 (`c7757280`) plus:
  - `.ai/` files (no code);
  - `hetero/`: a host-only C++ library (shared hand-off memory, front link, worker process) with CPU tests. Built
    and tested by Chat AI on Linux x86-64, g++ 13: `hetero_tests` passes, also under ASan/UBSan.
  - `hetero/probe/shared_handoff_probe.cu` and its CMake wiring at the end of `hetero/CMakeLists.txt`.
    **UNVERIFIED: written without a GPU toolchain; never compiled.**
  - one new option in the top-level `CMakeLists.txt`: `STRATA_HETERO` (OFF by default).
- No engine or server file of vanilla is changed. HIP-only inference on this branch is plain vanilla.
- The earlier branch `feature/heterogeneous-hip-only` is frozen. Do not change it, its working tree or its build
  directories. Work in a separate checkout or `git worktree` of `feature/mixed-native`, with its own
  checkout-local `.venv` made the way vanilla's setup makes it.

## Tasks
1. **Archive the old vanilla pin.** In the local vanilla checkout, commit `8db7ddc4` is the earlier reference. If
   every commit between the public history and `8db7ddc4` is already public, or contains nothing private, push it
   to the fork as branch `archive/vanilla-pin-oberon` (push only; do not modify that checkout). If it holds local
   commits you cannot judge, do not push; list their one-line subjects in the report instead.
2. **Build three trees** of this branch with 28 jobs and the vanilla toolchain: host (CPU only), CUDA, and HIP
   (gfx1201). Use the same configure options as the existing vanilla builds on this machine, plus
   `-DSTRATA_HETERO=ON`. Use new build directories.
3. **CPU tests.** In each tree run `ctest -R hetero_tests`. In the host tree also run the vanilla CPU test suite
   once and note failures; do not fix vanilla test failures.
4. **Shared hand-off probe.** Make `hetero_shared_probe` (CUDA tree) and `hip_hetero_shared_probe` (HIP tree)
   build; fix compile, link and CMake errors in the probe and its wiring. Then run, as the comment at the top of
   the source describes, on one file in `/dev/shm`:
   - CUDA writer with HIP reader on HIP device 0, then on HIP device 1;
   - HIP writer (device 0) with CUDA reader.
   Expected: both sides print `register ... ok` and end with `ok, 200 windows`. Any other outcome is a result to
   report, not something to work around.
5. **Vanilla HIP-only baseline.** Take the existing known-good vanilla run config of this machine (the one that
   serves Q2_0 on the two R9700 with `--kv int8`, `--max-context 262144`, a native layer split and MTP). Copy it
   into the new checkout and change **only** `exe`, to this branch's HIP build of `strata`. Start the server the way
   vanilla is normally started here, wait until it is ready, send the same short greedy chat request twice
   (about 200 new tokens; the first run is the cold one), then shut down cleanly.

## Constraints
- You may change: `hetero/probe/shared_handoff_probe.cu`, `hetero/CMakeLists.txt`, build configuration, and the
  copied run config's `exe`. If `hetero/src` or `hetero/tests` does not compile with the local compiler, make the
  smallest fix and say so.
- Do not change vanilla engine, kernel, server or Terminal files. If a task seems to need that, stop that task and
  report `Tier escalation recommended` with the reason.
- Do not change the probe's method (register the shared mapping, GPU writes rows then the ready word, the other
  GPU waits on the word). If registration or the cross-process wait fails, report the exact error text.
- GPU processes are never killed by force.

## Hardware Test Budget
5 minutes in total for tasks 4 and 5. Builds and CPU tests do not count.

## Completion Criteria
- The three trees build; `hetero_tests` passes in each.
- The probe has been run in the three pairings and its output is in the report (or the reason it could not run).
- One HIP-only vanilla server start, two requests and a clean shutdown have been done, and the numbers are in the
  report.
- Changes and `.ai/reports/report-0063.md` are committed and pushed to `feature/mixed-native`.

## Reporting
- Task 1: pushed, or the subjects of the commits that kept you from pushing.
- Build: configure options used per tree, compiler versions, any fix made (file and one line why).
- Probe: the printed lines of each pairing.
- Baseline: which engine and drafting mode vanilla chose; the startup lines that show the layer split and any
  architecture defaults applied; for each of the two requests the prompt tokens, completion tokens, time to first
  token, prefill tokens per second and decode tokens per second, as the server reports them; free memory per card
  while loaded.
- Anything in the vanilla 0.2.15 start that differs from how the earlier vanilla checkout behaved.
