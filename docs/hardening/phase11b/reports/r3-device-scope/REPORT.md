# R3 allocation-device correction — 2026-11-07

## Result

**The native R3 device-scope blocker is cleared.** The actual production class now
passes the two-device diagnostic in both physical GPU directions. CUDA runtime
and full-model layer-split startup qualification remain outstanding.

Started in clean `/home/chris/workspace/Strata-dev`; applied once to the existing
`/home/chris/workspace/Strata-worktrees/phase11b-prefill-restore` worktree.
Branch remains `fix/phase11b-prefill-restore-lifetime`, HEAD
`0ef75af1730c9374bbf1cb66a361f410d699b805`.

Before integration, exported the actual dirty worktree. It exactly matched the
required R3 full patch SHA-256:
`9b922a075fb1873a9082d8af9ea50f498c54119165b50753b774daf85da36527`.
Saved as pre-integration.patch. No earlier patch was reapplied.

Input: `/home/chris/tmp/strata-dev/STRATA_PHASE11B_R3_DEVICE_SCOPE_INCREMENTAL.patch`.
SHA-256: `46d74512453798d6f1285f616bb2c658738121d210f1be8a870dbe8cb80d2d7f`.
Copied here; `git apply --check` and application passed without adjustment.
No production correction beyond the supplied delta was needed.

## Fresh validation

| Check | Result |
|---|---|
| CPU group, including all 17 expert-cache completion cases | 89/89 passed |
| Same group under ASan/UBSan + leak detection | 89/89 passed |
| Actual ExpertCache and production executable, HIP gfx1201 | Built |
| Actual ExpertCache and production executable, CUDA 12.0 sm_89 | Built |
| Actual class device-scope diagnostic, both physical directions | 2/2 passed |
| Diagnostic's registered CTest entry | Passed (overlaps forward case) |
| Native cache startup/error cases + existing cache tests, R9700 #0 | 26/26 passed |
| Cache subset, R9700 #1 | 5/5 passed |
| Rebuilt R1/R2 native fixtures | 42/42 passed |
| Negative control: old production policy + strengthened fixture | Correctly rejected, exit 2 |

All HIP tests ran bounded and serial with SDMA off. No kernel-journal fault matched
the existing MES/TLB/page-fault/reset stop condition. Synthetic runtime errors
follow valid submission or a real successful fence; no physical GPU fault was
manufactured. Initial/focused/CTest repetitions are not extra unique coverage.
Existing ignored-return warnings outside the supplied correction remain in build
logs; successful builds are not a warning-free or all-backend claim.

NVIDIA is still PCI-visible at 04:00.0, but NVML reports an unknown handle error.
Availability did not change; no fresh CUDA runtime execution is claimed.

## Fixture integration: strengthened, not weakened

The first run with the original fixture released the initial device-1 cache
correctly, then failed during its later sized reopen. The old fixture retained
`cache_device=1` even though successful caller restoration meant the new cache was
allocated on device 0. Log: evidence/device-scope-original-fixture.log.

Corrected only fixture bookkeeping and added assertions:
- Track the actual current device for each cache arena allocation, including reopen.
- Require every close to restore its entry device.
- Require cache releases to occur on the allocation device, after its successful
  fence, with no outstanding operands. Existing early-release/fence checks remain.

The production diagnostic now emits DEVICE_SCOPE_PASS and CACHE_PASS, without a
wrong-device or early-release marker. Both physical directions use logical owner 1
and caller 0, with UUID order reversed between runs.

To verify that the test still detects the original defect, negative_control.py:
1. Extracts expert_cache.cpp from the exact pre-integration state using a separate
   index, without reverting any current source.
2. Compiles that old implementation against the current header for ABI consistency.
3. Links the strengthened native fixture to this isolated old-policy object before
   the current engine archive, leaving production binaries and build objects alone.
4. Requires exit 2 plus WRONG_DEVICE_FENCE / EARLY_RELEASE. This passed; valid
   owner-device work is fenced by diagnostic safety code before the guarded exit.

Negative-control source, build commands, logs and result JSON are under
`evidence/negative-control/`. Its deliberate failure is not a current-code failure
or a physical UAF. The initial stale-bookkeeping failure is retained separately.

## Source review / remaining limits

The supplied correction records the device in open before storage allocation.
Close keeps that owner selected across both fence and frees and restores the
entry device afterward. Empty caches do not select/fence a device. The CPU cases
cover missing owner and get/select/sync/restore failure. Successful fill paths are
unchanged; native async/queued wait assertions still pass.

No dedicated full-model GpuStage startup-error fixture was found in the searched
project tests. The class-level cross-device lifetime test does not establish the
entire layer-split early-return path; this remains explicitly untested. No separate
project paths were searched. Native CUDA/VMM, SDMA-on, active stream-capture teardown,
Metal/SYCL, concurrency, long-context and performance qualification remain open.
No fresh TSan or Compute Sanitizer result is claimed. Advisor discovery found no
available advisor.

Updated the in-repository device-scope handoff with these results and marked the
original R3 device-scope finding as historical/resolved. Earlier external reports
and evidence were not rewritten.

## Actual command summary

All shell tools had explicit outer timeouts. Raw commands/environment for native
runs are also recorded in JSON. Relevant outputs are in evidence/.

```sh
W=/home/chris/workspace/Strata-worktrees/phase11b-prefill-restore
R=/home/chris/workspace/Strata-phase11b/r3-device-scope
B=/home/chris/workspace/Strata-builds
cd "$W"
git apply --check "$R/STRATA_PHASE11B_R3_DEVICE_SCOPE_INCREMENTAL.patch"
git apply "$R/STRATA_PHASE11B_R3_DEVICE_SCOPE_INCREMENTAL.patch"
cmake --build "$B/phase11b-hip" --target strata expert_cache_gpu_test -j 4
cmake --build "$B/phase11b-cuda" --target strata -j 4
cmake --build "$B/phase11b-hip" --target hip_expert_cache_staging expert_cache_per_layer_test expert_cache_slot_guard_test peer_experts_gpu_test remote_opt_gpu_test remote_experts_gpu_test -j 4

# Separately for D=phase11b-cpu and D=phase11b-sanitizers:
cmake --build "$B/$D" --target expert_cache_completion_test peer_experts_completion_test remote_opt_completion_test transfer_failure_test conversation_lifetime_cuda_test conversation_lifetime_hip_test prefill_completion_test pinned_ring_test conv_cache_test conversation_restore_orchestration_test -j 4
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 ctest --test-dir "$B/$D" -R '^(expert_cache_completion_|peer_experts_completion_|remote_opt_completion_|transfer_|lifetime_|prefill_completion_test$|pinned_ring_test$|conv_cache_test$|conversation_restore_orchestration_test$)' --output-on-failure

python3 "$R/run_gpu_tests.py" hip --uuid GPU-e39306ee46993aad,GPU-2f182f01261ba30b --label hip-scope-forward --remaining
python3 "$R/run_gpu_tests.py" hip --uuid GPU-2f182f01261ba30b,GPU-e39306ee46993aad --label hip-scope-reverse --remaining
python3 "$R/run_gpu_tests.py" hip --uuid GPU-e39306ee46993aad --label hip-cache-device0
python3 "$R/run_gpu_tests.py" hip --uuid GPU-2f182f01261ba30b --label hip-cache-device1-small --small
python3 "$R/r1_regression.py" hip --uuid GPU-e39306ee46993aad,GPU-2f182f01261ba30b --label hip-r1-regression
python3 "$R/r2_regression.py" hip --uuid GPU-e39306ee46993aad,GPU-2f182f01261ba30b --label hip-r2-regression
python3 "$R/negative_control.py"
LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib ROCR_VISIBLE_DEVICES=GPU-e39306ee46993aad,GPU-2f182f01261ba30b HSA_ENABLE_SDMA=0 ctest --test-dir "$B/phase11b-hip" -R '^expert_cache_native_device-scope$' -j 1 -V --output-on-failure
nvidia-smi -L
lspci -nn -s 04:00.0
git diff --check
python3 "$R/finalize_evidence.py"
python3 "$R/package_incremental.py"
```

## Deliverables

- review.patch: updated full uncommitted worktree patch against HEAD, including
  untracked source. SHA-256:
  `016ce16e99e505bb76818b743084b1226d25546237ea069757afd26976c4edf2`.
- R3_DEVICE_SCOPE_VALIDATED_INCREMENTAL.patch: this correction plus fixture/docs,
  relative to the exact R3-integrated state. SHA-256:
  `720b704bb6230ebf183914845bc52eabb8a5786ab19954a16b89a57810eb56cf`.
- Both output patches passed applicability checks against their proper bases via
  a separate temporary index. The authoritative index remains empty. Do not reapply
  either output patch to the already-updated worktree.
- MANIFEST.sha256 and evidence source/binary identities record the artifact hashes.

No commits, pushes, Actions, driver/reset/system changes, preservation-payload
writes, model downloads or changes to the separate heterogeneous project.
Strata-dev remains clean; the Phase 11B implementation remains uncommitted.
