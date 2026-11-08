# Phase 11B R3 — ExpertCache startup completion safety

## Context

Incremental source patch relative to the R1-validated Phase 11B dirty worktree.
Base HEAD `0ef75af1730c9374bbf1cb66a361f410d699b805`, with R4/R5/R6,
no-MTP disk save, R2, and R1 already applied. Do **not** reapply the full `review.patch`.

Agent starts in `/home/chris/workspace/Strata-dev` and selects the linked
`/home/chris/workspace/Strata-worktrees/phase11b-prefill-restore` worktree.
Copy this patch from `/home/chris/tmp/strata-dev/` and apply there once.

Do not inspect or modify the separate heterogeneous project paths.

## Correction

The existing synchronous startup H2D and D2H calls were checked but could
return error without guaranteeing the caller's source blob, HIP pinned stage,
or local verification vector no longer had outstanding GPU references.
The queued legacy-stream path could also report a partial submission failure
without draining earlier accepted pieces. `close()` freed storage without
establishing whole-device completion.

Use the existing transfer-terminal process boundary on an unproven drain.
On an issue or wait error, attempt legacy-stream completion; if it succeeds,
return the original failure. If completion remains unknown, exit without
unwinding data owners. During a cache close with allocations, synchronize the
whole device before freeing cache memory. This is a lifecycle operation, not a
per-token synchronization. No success-path wait is added to fill/verify.

R3 was originally classified as plausible/unproven hardware hazard: this patch
closes the missing static ownership policy, **not** a proven GPU use-after-free.

## Initial Oberon results — 2026-11-07

The initial R3 input applied cleanly once to the exact R1-validated dirty state.
The device-scope failure below was then corrected by the incremental follow-up in
[PHASE11B_R3_DEVICE_SCOPE_FIX.md](PHASE11B_R3_DEVICE_SCOPE_FIX.md). Its production-class
native diagnostic now passes in both physical GPU directions. The authoritative
worktree remains uncommitted. The results below describe the initial R3 step.

- CPU: nine new completion cases plus the existing 72 regressions passed, 81/81,
  normally and under ASan/UBSan with leak detection.
- HIP gfx1201 and CUDA 12.0 sm_89: actual expert_cache.cpp and production executable
  built. Existing ignored-return warnings remain; this was not a warning-free build.
- Added tests/core/expert_cache_gpu_test.cpp, linked to the actual production class.
  On R9700 #0, 23 new same-device cases and three existing cache tests passed 26/26.
  A small subset on R9700 #1 passed 5/5. R1/R2 native regressions passed 42/42.
- Native coverage includes blocking H2D, D2H local-vector destruction, async and
  queued/partial uploads, first/second sync errors, failed close, open memset and
  staging failures, sized/uniform reopen, null verification, nonblocking readers
  and a launched graph reader. It checks exact bytes and absence of new healthy
  fill-path waits. Unknown completion exits 1 before owner destruction.
- At this initial step, a separate two-device diagnostic failed with
  WRONG_DEVICE_FENCE and EARLY_RELEASE (exit 2), exposing the device-scope gap below.
  The follow-up makes that same diagnostic pass, without WILL_FAIL or suppression.
- All HIP work was bounded and serial with SDMA off. No kernel faults matched the
  MES/TLB/page-fault/reset stop condition. HIP testing stopped after the diagnostic.
- NVIDIA enumeration remains unavailable to NVML despite PCI visibility. No fresh
  CUDA runtime attempt was made because availability had not changed.

Raw evidence, commands, report and updated full/incremental patches are in
`/home/chris/workspace/Strata-phase11b/r3-delta/`. The in-repo CTest names are
expert_cache_native_*; device-scope needs two visible cards, the other cases one.
These are model-independent fixtures, not a full-model or throughput result.

## Device-scope finding — resolved by the follow-up

ExpertCache stores no allocation device. The new close fence synchronizes only
whatever device is current on the thread. HIP explicitly documents that scope:
[hipDeviceSynchronize](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/device_management.html).

Reproduction: allocate/fill a cache and queue a nonblocking reader on logical
device 1, switch the caller to device 0, then call the actual cache.close(). The
production fence runs on 0. The fixture observes the subsequent allocation-release
attempt without production completion proof on 1 and stops before the release.
For diagnostic safety it separately fences valid owner-device work, but does not
count that extra fixture fence as production proof. No actual use-after-free or
GPU fault was manufactured or observed.

Related caller from static inspection: generate.cpp's GpuStage has a cache and no
custom device-scoped destructor. The stage-cache startup loop uses a local
OnDevice guard and has fill/validation early returns. That guard unwinds before
the stages vector; the later default cache destructor has no guarantee that its
allocation device is current. This application path was not run with a real model.

Per the review instruction, the initial step stopped rather than weakening the
lifetime policy. The follow-up records the cache's allocation device and scopes
its entire close fence/free to that device, then restores the entry device. The
native diagnostic passes. Full-model stage early-return integration remains untested.

## Review points

- The device-scope regression milestone now passes. A launched graph reader on
  the owning device also passes; active stream-capture teardown and full layer-split
  graph/caller ordering remain unqualified.
- `fill_slot` remains asynchronous on success; the caller still owns ordering.
- Inspect any `open()` or `open_sized()` failure that might release an arena
  while allocations still have work outstanding.
- Do not move the completion fences into per-token successful hot paths.
- Keep source and results in Strata-dev project; do not commit/push automatically.

## ChatGPT local checks

- GCC and Clang: nine fake-runtime completion scenarios each passed,
  including `-DNDEBUG` optimized builds.
- GCC and Clang ASan/UBSan: nine scenarios each passed.
- Patch application and whitespace checks passed against a source-context
  reconstruction. **Production Strata compilation and native tests have not
  run here**; Oberon validation remains authoritative.
- The full R1 artifact ZIP and patch SHA were verified before developing this
  incremental change. The R1 full patch does not touch expert_cache.cpp.
