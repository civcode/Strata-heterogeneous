# Phase 11B peer primary-device validation

## Result

Passed the requested host-staged ownership milestone on Oberon. P2P qualification remains a separate follow-up.

Starting directory: `/home/chris/workspace/Strata-dev` (still clean).
Existing linked worktree: `/home/chris/workspace/Strata-worktrees/phase11b-prefill-restore`.
HEAD: `0ef75af1730c9374bbf1cb66a361f410d699b805`.

Input patch remains at `/home/chris/tmp/strata-dev/STRATA_PHASE11B_PEER_PRIMARY_INCREMENTAL.patch`.
SHA-256: `dc5b976bd18fb86171d2679f455db19fb87bab877a27e28747dd968749184a0b`.
Checked and applied exactly once to the existing dirty state. Reverse applicability was checked after the interruption; no second application was attempted. No production or test source adjustment was needed.

## Validation

- HIP gfx1201: rebuilt `strata`, `peer_experts_gpu_test`, `expert_cache_gpu_test`.
- CUDA sm_89: rebuilt `strata`; resumed the interrupted build successfully after the user reported the unrelated CUDA error resolved. Initial and resumed logs are retained separately.
- Registered CTest `peer_native_primary-one`: passed twice, logical primary 1/helper 0, with UUID orders:
  - `GPU-e39306ee46993aad,GPU-2f182f01261ba30b`
  - `GPU-2f182f01261ba30b,GPU-e39306ee46993aad`
- Existing peer cases: success-direct, success-staged, close-queued, select-failure, input-issue, refill-success: 6/6 passed.
- Cache cases: success, async-success, readback-unknown, graph-reader, device-scope: 5/5 passed.
- Total: 13/13 native invocations passed, zero skips. Synthetic failure-path cases passed their expected-exit assertions.
- All GPU runs serial and bounded, with `STRATA_PEER_NO_P2P=1`, `HSA_ENABLE_SDMA=0`, and the existing kernel-journal MES/TLB/page-fault/reset stop policy. Journal reads succeeded; no matching kernel faults.
- Tracked and untracked patch whitespace checks passed; authoritative index remains empty.

Commands are preserved in `run_validation.py` and `evidence/results.json`. Build logs, native output, kernel journal and source/binary hashes are in `evidence/`.

## Source identity

Full worktree patch (including untracked tests and the updated validation note): `review.patch`.
SHA-256: `06d74352e1a8909f3ea7588de0b41c355734f65e3bcec357b2494be7f5c379e1`.
Only the documentation validation note changed after test execution. Source file hashes and executable hashes are recorded separately.

## Limits

No P2P-enabled test was attempted. The new case disables inter-GPU P2P but retains the existing mapped-host output path; it is not a staged-output-only test. Native CUDA execution, full-model layer-split startup and broader backend qualification remain untested here. The CPU group was not repeated: this delta changes a header comment, not header behavior. Advisor discovery returned no available advisor tool.

No commits, pushes, Actions, driver changes, model downloads or separate-project changes were made. The supplied incremental patch must not be reapplied to this worktree.
