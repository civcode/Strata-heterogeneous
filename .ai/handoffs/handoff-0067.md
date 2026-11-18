# Handoff 0067

## Tier
2

## Objective
Build and GPU-test the hardening Chat AI ported from `civcode/strata-dev-private` onto vanilla 0.2.15, and show that
HIP-only still serves as before. This is the gate before any further mixed-engine work.

## Context
- Read `.ai/PROJECT.md` first (the HIP build now includes `-DSTRATA_MMQ=ON`). `.ai/HARDENING.md` lists the ported
  commits and how conflicts were resolved.
- On 17 November Oberon had a host crash (native fault in `libhipblaslt`, PID 1 fault, reboot) while running a
  vanilla 0.2.15 build. No cause is established. Days of work on the patched private branch showed no such failure.
- `feature/mixed-native` at the handoff commit = vanilla 0.2.15 + the 11 ported commits + one port fix + the
  `hetero/` seam (unchanged, off by default). The unfinished mixed-engine work of handoff 0066 is reverted here and
  kept on branch `wip/mixed-front-0066`; do not use or change it.
- Checked by Chat AI without a GPU: the 90 CPU tests that come with the patches pass; the merged engine `.cpp` files
  pass a syntax check against CUDA headers. **Not compiled yet: everything in the CUDA and HIP builds, including the
  `.cu` files the patches change (`remote_expert_opt.cu`, `cvec.cu`, `q36_engine.cu`, `moe_fused` headers).**
- Use the worktree `/home/chris/workspace/Strata-mixed-native` with new build directories.

## Tasks
1. **Before any GPU run**, record the kernel log baseline: the last boot's `journalctl -k` lines matching
   `amdgpu|gfxhub|GCVM|MES|page fault|segfault|general protection` and `coredumpctl list` since boot. Repeat after the
   last GPU run. A new GPU fault, a new crash of any process, or an unexpected reboot ends the handoff at once:
   do not start another GPU process, and report what happened.
2. **Build** host, CUDA (CUDA 13.0, sm_89) and HIP (gfx1201, with both MMQ options) trees with `-DSTRATA_BUILD_TESTS=ON
   -DSTRATA_HETERO=ON`, 28 jobs. Fix compile and link errors that come from the port (a name 0.2.15 renamed, a missing
   include, a moved helper) without changing what the code does.
3. **CPU tests** in the host tree: the whole suite.
4. **GPU tests** in the HIP tree, then the CUDA tree, one tree at a time: the tests the patches add
   (`conversation_readback_test`, `conversation_prepare_*`, `conversation_capture_formats_test`, `lifetime_native_*`,
   `pinned_ring_gpu_test`, `transfer_hip_*`, `remote_owner_*`, `expert_cache_native_*`, `peer_owner_*` or
   `peer_experts_native_*`, `remote_opt_native_*`, whatever names CTest lists for them) plus vanilla's
   `conversation_copy_streams_test`. Run with `ctest -j1`. A test that needs a model file and skips is fine; list it.
5. **Readback reproducer** from handoff 0066 Part B, once per R9700, against the patched HIP build.
6. **HIP-only smoke** with the MMQ HIP build and the copied run config from reports 0065 and 0066 (only `exe`
   changed): start, the same short greedy request twice (the second reuses the conversation cache), the ~5000-token
   prompt once, then a clean shutdown. Confirm no engine process and no GPU memory is left.

## Constraints
- You may change: the ported files, only to make them compile against 0.2.15 (each change named in the report);
  build configuration; the copied run config's `exe`. Nothing else.
- If a ported test fails at run time, do not change the production code or the test to make it pass. Report the
  test, its output and your reading of it. If a fix needs a design decision, mark it `Tier escalation recommended`.
- If the engine fails to start or a request fails, report the exact message and stop.
- All rules of `.ai/PROJECT.md`: never two GPU jobs at once, never kill a GPU process by force, do not touch the
  vanilla checkout, the frozen branch or the private repository.

## Hardware Test Budget
15 minutes of GPU run time in total (tasks 4 to 6). Builds and CPU tests do not count.

## Completion Criteria
- The three trees build; every compile fix is listed.
- CPU suite, GPU tests, reproducer and smoke have run, or the first failure that stopped them is reported.
- The kernel log and core-dump checks before and after are in the report.
- `.ai/reports/report-0067.md` and all changes are committed and pushed to `feature/mixed-native`.

## Reporting
- Compile fixes (file, one line why).
- Test results: counts per tree; every failure or skip by name with its message.
- Smoke: per request prompt tokens, prefill tok/s, decode tok/s, drafts accepted, conversation cache reuse,
  beside the report 0065 baseline; memory per card while loaded; shutdown clean or not.
- Kernel log and core-dump lines that appeared during the handoff, or "none".
- GPU run time used.
