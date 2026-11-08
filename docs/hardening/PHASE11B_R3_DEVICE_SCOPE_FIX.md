# Phase 11B R3 — allocation-device teardown correction

## Context and patch base

This patch is **incremental to the R3-integrated, uncommitted** Phase 11B worktree;
its base is the complete `r3-delta/review.patch` (SHA-256
`9b922a075fb1873a9082d8af9ea50f498c54119165b50753b774daf85da36527`).
Do not reapply any earlier R1, R2 or R3 patch.

Agent starts in `/home/chris/workspace/Strata-dev`, then selects the linked
`/home/chris/workspace/Strata-worktrees/phase11b-prefill-restore` worktree.
Receive the incremental patch in `/home/chris/tmp/strata-dev/`.
Do not access any heterogeneous-project worktree or temporary directory.

## Defect and correction

The first R3 device-wide close fence waited on the calling thread's **current**
GPU. A `GpuStage` or other caller can allocate an `ExpertCache` on one GPU and
later destroy it after its temporary device scope has restored another GPU.
HIP `hipDeviceSynchronize()` waits only on the current device. The existing
`expert_cache_native_device-scope` regression correctly stopped before unsafe
release with `WRONG_DEVICE_FENCE` / `EARLY_RELEASE`.

`ExpertCache::open` now records its allocation device before allocating storage.
`ExpertCache::close` selects that device for the entire completion fence **and**
all allocation releases, and restores the previous current device afterward.
If device lookup, selection, the fence, or restoration fails, the existing
process-terminal boundary is used before any potentially unsafe caller unwind.
No GPU selection or wait is added to successful fill or inference hot paths.
No resource means no selection/fence is needed.

## Local checks (ChatGPT)

Focused CPU fake-runtime regressions cover normal owner switch and restore,
owner already current, no storage, lookup failure, selection failure, device
sync failure, restoration failure and missing owner. See handoff response for
compiler/sanitizer results. Full production CUDA/HIP builds and the native
multi-GPU diagnostic must run on Oberon.

## Oberon validation completed — 2026-11-07

- Verified the exact R3 base patch hash above; the input checked and applied once
  without adjustment. No further production correction was required.
- Actual HIP gfx1201 and CUDA 12.0 sm_89 ExpertCache/production builds passed.
- CPU regressions, including all 17 expert-cache completion cases: 89/89 passed
  normally and under ASan/UBSan with leak detection.
- Production-class device-scope diagnostic: passed in both physical R9700
  directions; its CTest entry also passed. It verifies the owner fence, owner-device
  releases, queued reader bytes, caller-device restoration and sized reopen.
- Corrected a stale fixture expectation: after restoring caller device 0, its
  later reopen correctly allocates on 0, not the original owner 1. The fixture now
  records the actual allocation device for each cache allocation. No completion
  check was removed; added release-device and entry-restoration assertions.
- Negative control: the strengthened fixture, linked to an isolated copy of the
  pre-scope production implementation, still fails at WRONG_DEVICE_FENCE /
  EARLY_RELEASE (exit 2). No current source or production build was reverted.
- Cache native tests passed 26/26 on R9700 #0, subset 5/5 on #1. Rebuilt R1/R2
  native regressions passed 42/42. All HIP workloads were bounded, serial and
  SDMA off; no kernel fault matched the MES/TLB/page-fault/reset checks.

The native device-scope blocker is cleared. Full-model GpuStage/layer-split startup
failure was not run: no existing dedicated fixture was found. The class-level
cross-device diagnostic is not a full-model reproduction. NVIDIA remains PCI-visible
but unavailable to NVML; native CUDA, CUDA VMM, SDMA-on, active stream capture and
broader backend qualification remain outstanding. No commits, pushes, driver or
separate-project changes were made.

Report, raw evidence and full/incremental patches:
`/home/chris/workspace/Strata-phase11b/r3-device-scope/`.

The required owner-device regression has passed on the actual production class
on Oberon. This closes that native blocker, not the remaining qualification limits
listed above.
