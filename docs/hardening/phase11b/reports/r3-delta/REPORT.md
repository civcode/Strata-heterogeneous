# Phase 11B R3 integration — 2026-11-07

## Status: integrated, but R3 is not closed

The supplied patch applied cleanly once. Same-device startup/error-path tests pass.
A two-device teardown diagnostic exposes a missing allocation-device scope. Per the
handoff's review instruction, stopped HIP testing and reported the specific case
instead of weakening the fence or silently rewriting caller teardown.

## Integration identity

Started in `/home/chris/workspace/Strata-dev` (clean). Authoritative implementation:
`/home/chris/workspace/Strata-worktrees/phase11b-prefill-restore` on
`fix/phase11b-prefill-restore-lifetime`, HEAD
`0ef75af1730c9374bbf1cb66a361f410d699b805`.

Exported the actual dirty state before integration. It exactly matched the R1 full
patch SHA-256: `156d6fa43cae8e5fdd81674ee172083953ecf25b81d9c2f5036ae9def431ec1d`.
Saved as `pre-integration.patch`; no previous patch was reapplied.

Input: `/home/chris/tmp/strata-dev/STRATA_PHASE11B_R3_INCREMENTAL.patch`.
SHA-256: `5d9e0d4b5213b30b0fd7b33e81119887f64f8f1e82cf0ba3936b9b3608972c2a`.
Copied here, checked against the actual worktree, then applied once without a hunk
adjustment. No production correction to the input implementation was made.

## Fresh results

| Check | Result |
|---|---|
| Nine supplied CPU cases + existing 72 relevant regressions | 81/81 passed |
| Same group under ASan/UBSan, leak detection | 81/81 passed |
| Actual expert_cache.cpp + production strata, HIP gfx1201 | Built successfully |
| Actual expert_cache.cpp + production strata, CUDA 12.0 sm_89 | Built successfully |
| R9700 #0: 23 new same-device native cases + 3 existing cache tests | 26/26 passed |
| R9700 #1: five-case native subset | 5/5 passed |
| Rebuilt R1 native regression fixture | 20/20 passed |
| Rebuilt R2 optimization + remote-owner fixtures | 22/22 passed |
| Two-device cache teardown device-scope diagnostic | FAILED: process exit 2, wrong-device fence / guarded release |
| CUDA runtime | Not run: availability unchanged, NVML still cannot obtain the PCI-visible device handle |

HIP tests ran serially with HSA_ENABLE_SDMA=0 and bounded process deadlines. No
MES/TLB/page-fault/reset matches occurred in the recorded kernel-journal windows.
No HIP tests ran after the failing device-scope diagnostic. Existing ignored-return
warnings in device_free_bytes and cache freeing remain in the HIP build log;
these are successful builds, not warning-free or all-backend qualification.

## New production-class fixture

Added `tests/core/expert_cache_gpu_test.cpp` and `expert_cache_native_*` CTest entries.
It links the real engine, uses public ExpertCache APIs and model-independent 4/8-KiB
payloads, and does not copy production methods or access private fields.

Coverage:
- Exact blocking and queued startup bytes; residency/fill accounting; uniform and
  sized reopen after every nonfatal case.
- Accepted H2D issue errors, including HIP's actual pinned blocking stage.
- D2H verification errors: link wrappers observe deletion of the actual local
  verification vector, requiring its drain before destruction.
- Async nonblocking-stream failures and queued legacy-stream failures.
- Multipart upload errors after a second piece was accepted; accepted bytes and
  untouched remaining bytes checked, no successful-fill count published.
- First sync failure followed by a successful second drain, and persistent errors.
- Cache close with an outstanding nonblocking reader and a launched graph reader.
- Failed device completion before close/free; caller/cache/vector owner markers
  must not run after unknown completion. Fatal children must exit 1.
- Open allocation refusal, accepted memset/report-error, HIP staging allocation
  refusal and sized-stage growth refusal; cleaned-up caches must reopen.
- Null verification refusal; successful async/queued fill paths must not acquire
  extra stream waits.

Fault wrappers submit or fence valid work, then report synthetic errors. They do
not cause real GPU faults. Fork/exec occurs before HIP initialization. The driver
uses a 60-second process deadline with kill grace and checks kernel logs after
each case. Source and native fixture binary identities are in `evidence/`.
The first focused success run and earlier whole-group runs overlap final coverage;
only final selections above are counted as unique coverage.

## Blocking result: wrong-device close fence

The actual cache stores no allocation device. `ExpertCache::close()` calls the new
whole-device completion helper on whichever device is current, then frees the
cache's staging/arena. HIP documents that hipDeviceSynchronize waits for the
**current device** only; hipSetDevice changes thread-local selection and does not
synchronize either device:
https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/device_management.html
Both configured web search tools agreed; the authoritative AMD page was fetched.

Reproduction (actual production class):
1. With both R9700s visible, allocate/fill the cache on logical device 1.
2. Queue a valid nonblocking cache reader on device 1.
3. Switch the caller to device 0 and call cache.close().
4. Its production device fence runs on 0. A cache allocation release is then
   attempted without production completion proof for 1.

The fixture prints WRONG_DEVICE_FENCE and EARLY_RELEASE and exits 2 before the
release. For diagnostic safety only, it separately fences valid work on the owner
device; that extra fixture action is deliberately not counted as production proof.
This demonstrates a completion-policy gap, not a physically observed UAF: the work
might finish naturally, and the test does not allow a real unsafe free or GPU fault.

Log: `evidence/hip-r3-device-scope-00-expert_cache_gpu_test-device-scope.log`.
JSON: `evidence/hip-r3-device-scope-results.json` (failed=1, kernel faults empty).
The registered `expert_cache_native_device-scope` remains a failing regression on
two visible devices. It is not hidden with WILL_FAIL or counted as a pass.

Related application path, **unconfirmed static analysis**, not a full-model repro:
- generate.cpp's GpuStage contains an ExpertCache and no custom device-scoped
  destructor (around lines 1487-1508); stages is a vector of unique_ptrs (line 5726).
- The stage-cache startup loop (around lines 8477-8590) uses a local OnDevice guard
  and returns early on profile-fill/validation failures. That guard unwinds before
  the outer stages vector. The eventual default cache destructor therefore lacks
  a guaranteed owner-device selection.
- Prefill restores its entry device during release; Verifier's destructor has no
  allocation-device scope. No full layer-split teardown qualification is claimed.

Next change should establish the correct device for fence/free, either by storing
cache ownership or by explicit caller teardown with correct ordering. Resolve this
case before calling R3 complete, then rerun the device-scope regression and stage
startup-failure integration. No recovery or ownership redesign was attempted here.

## Lifecycle review scope

Inspected uniform/sized startup failure paths, staging growth, segmented mapping
failure/cleanup and the relevant peer/remote close callers. HIP memset/staging
failures now reach the checked close boundary and passed native tests. PeerExperts
and RemoteExperts explicitly select their helper device before cache.close; their
rebuilt native regressions pass. No GPU work precedes the segmented startup mapping
failure path, but native CUDA/VMM cleanup remains untested.

A launched graph reader on the owning device passed. This is not permission to
close during stream capture (host/device synchronization is prohibited there),
and not proof of every active-graph caller's lifetime ordering. The static review
stopped at the concrete device-scope finding rather than becoming a general audit.

## Commands

All shell calls had explicit outer timeouts. Logs preserve the actual build/test
outputs. Shorthand below summarizes executed commands.

```sh
W=/home/chris/workspace/Strata-worktrees/phase11b-prefill-restore
R=/home/chris/workspace/Strata-phase11b/r3-delta
B=/home/chris/workspace/Strata-builds
cd "$W"
git apply --check "$R/STRATA_PHASE11B_R3_INCREMENTAL.patch"
git apply "$R/STRATA_PHASE11B_R3_INCREMENTAL.patch"
git diff --check
cmake --build "$B/phase11b-hip" --target strata -j 4
cmake --build "$B/phase11b-cuda" --target strata -j 4
cmake --build "$B/phase11b-hip" --target expert_cache_gpu_test hip_expert_cache_staging expert_cache_per_layer_test expert_cache_slot_guard_test -j 4
cmake --build "$B/phase11b-hip" --target peer_experts_gpu_test remote_opt_gpu_test remote_experts_gpu_test -j 4

# Separately for D=phase11b-cpu and D=phase11b-sanitizers:
cmake --build "$B/$D" --target expert_cache_completion_test peer_experts_completion_test remote_opt_completion_test transfer_failure_test conversation_lifetime_cuda_test conversation_lifetime_hip_test prefill_completion_test pinned_ring_test conv_cache_test conversation_restore_orchestration_test -j 4
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 ctest --test-dir "$B/$D" -R '^(expert_cache_completion_|peer_experts_completion_|remote_opt_completion_|transfer_|lifetime_|prefill_completion_test$|pinned_ring_test$|conv_cache_test$|conversation_restore_orchestration_test$)' --output-on-failure

python3 "$R/r1_regression.py" hip --uuid GPU-e39306ee46993aad,GPU-2f182f01261ba30b --label hip-r1-regression
python3 "$R/r2_regression.py" hip --uuid GPU-e39306ee46993aad,GPU-2f182f01261ba30b --label hip-r2-regression
python3 "$R/run_gpu_tests.py" hip --uuid GPU-e39306ee46993aad --label hip-r3-final-device0
python3 "$R/run_gpu_tests.py" hip --uuid GPU-2f182f01261ba30b --label hip-r3-final-device1-small --small
# Last HIP test; returns 1 from the driver, child fixture returns 2:
python3 "$R/run_gpu_tests.py" hip --uuid GPU-e39306ee46993aad,GPU-2f182f01261ba30b --label hip-r3-device-scope --remaining
nvidia-smi -L
lspci -nn -s 04:00.0
python3 "$R/finalize_evidence.py"
python3 "$R/package_incremental.py"
```

## Deliverables and limits

- `review.patch`: updated full uncommitted worktree patch, including new files,
  against HEAD. SHA-256:
  `9b922a075fb1873a9082d8af9ea50f498c54119165b50753b774daf85da36527`.
- `R3_REVIEW_INCREMENTAL.patch`: R3 + native fixture + handoff results, relative
  to the exact R1-validated state. **Contains the unresolved diagnostic failure.**
  SHA-256: `b2fe629c8aa0ef7138858668ec9f9c18688748d4c63a3af654134f246c0c9d1d`.
- Both output patches passed applicability checks using a separate index against
  their proper bases. The authoritative index remains empty. Do not reapply these
  patches to the already-updated worktree.
- The in-repository handoff is updated at
  docs/hardening/PHASE11B_R3_CHATGPT_HANDOFF.md.

Native CUDA/VMM, Metal/SYCL, SDMA-on R3, real-model startup/layer-split failures,
active stream-capture teardown, concurrent callers, physically failed GPU and
performance remain unqualified. No fresh TSan/Compute Sanitizer result is claimed.
Advisor discovery found no available advisor.

No commit, push, Actions, reset, driver/system change, model download or inspection
of the separate heterogeneous project. Strata-dev remains clean; the authoritative
Phase 11B worktree remains uncommitted with the unresolved R3 device-scope test.
