# Phase 11B R2 integration — 2026-11-07

## Result

Applied `/home/chris/tmp/strata/STRATA_PHASE11B_R2_INCREMENTAL.patch` once, after
`git apply --check` passed in the actual dirty worktree:
`/home/chris/workspace/Strata-worktrees/phase11b-prefill-restore`.

Branch remains `fix/phase11b-prefill-restore-lifetime`, HEAD remains
`0ef75af1730c9374bbf1cb66a361f410d699b805`. Existing R4/R5/R6 and optional-MTP changes retained.
Input patch SHA-256: `e240826753f97c4fd785d6da4d5ffa9f94a1dbe87a1b5c7ed212064af9545042`.
No production correction to the supplied R2 implementation was necessary.

## Fresh validation

| Check | Result |
|---|---|
| Four supplied CPU completion cases + existing 59 regressions | 63/63 passed |
| Same group under ASan/UBSan with leak detection | 63/63 passed |
| HIP gfx1201 production executable | Built; actual remote_expert_opt.cu compiled |
| CUDA 12.0 sm_89 production executable | Built; actual remote_expert_opt.cu compiled |
| New two-card HIP real-optimization fixture | 12/12 passed |
| Existing two-card HIP real-remote-owner fixture | 10/10 passed |
| CUDA native short probes | 0 passed, 3 skipped (exit 77) |

The 22 HIP cases ran serially with `HSA_ENABLE_SDMA=0` and both R9700 UUIDs visible:
`GPU-e39306ee46993aad,GPU-2f182f01261ba30b`. Both primary/helper directions were tested.
No kernel fault matched the existing MES/TLB/page-fault/reset stop condition in the harness.

### Native R2 coverage added locally

Added `tests/core/remote_opt_gpu_test.cpp` and 12 `remote_opt_native_*` CTest entries.
It links the actual engine, using public `attach/open/init/begin/finish/copy_rows/combine`
entry points. It does not copy the destructor or expose private members.

- One cached zero-weight expert executes a one-token optimized helper decode.
- Actual `copy_rows` and `combine` kernels read mapped optimization operands on a
  primary nonblocking stream before the production destructor runs.
- Link wrappers track the optimization's mapped host allocation and helper sum.
  Any release before successful completion checks on both devices is a test failure.
- Success checks the combined output, masked row output, detachment and entry-device
  restoration. The destructor starts with the helper device current.
- Partial initialization covers helper allocation refusal and mapped-alias failure;
  both require the allocated host owner to be fenced and released.
- Primary sync failure, helper sync failure and helper selection failure must cause
  child exit 1, with an injection marker and ERR but no optimization release marker.
- The parent forks/execs before HIP initialization. Actual runtime work is valid;
  sync wrappers report synthetic failure only after real successful completion.
  These tests do not manufacture a GPU hang or fault and do not prove behavior of
  a physically hung device.

Initial native fixture compilation failed on HIP's `void**` mapped-alias output
parameter. Corrected the test-only cast; retained the failed build log. Subsequent
native build and all cases passed. No engine change was needed.

Updated `docs/hardening/PHASE11B_R2_CHATGPT_HANDOFF.md` in the worktree with the
actual validation and remaining limits. R1 PeerExperts teardown was not changed.

## Commands

Shell tools had explicit outer timeouts. Commands below summarize the actual runs;
raw build/test logs and JSON command/environment records are in `evidence/`.

```sh
W=/home/chris/workspace/Strata-worktrees/phase11b-prefill-restore
R=/home/chris/workspace/Strata-phase11b/r2-delta
B=/home/chris/workspace/Strata-builds
cd "$W"
git apply --check "$R/STRATA_PHASE11B_R2_INCREMENTAL.patch"
git apply "$R/STRATA_PHASE11B_R2_INCREMENTAL.patch"
git diff --check

cmake --build "$B/phase11b-hip" --target strata -j 4
cmake --build "$B/phase11b-cuda" --target strata -j 4
cmake --build "$B/phase11b-hip" --target remote_opt_gpu_test remote_experts_gpu_test -j 4

# Separately for D=phase11b-cpu and D=phase11b-sanitizers:
cmake --build "$B/$D" --target remote_opt_completion_test transfer_failure_test conversation_lifetime_cuda_test conversation_lifetime_hip_test prefill_completion_test pinned_ring_test conv_cache_test conversation_restore_orchestration_test -j 4
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 ctest --test-dir "$B/$D" -R '^(remote_opt_completion_|transfer_|lifetime_|prefill_completion_test$|pinned_ring_test$|conv_cache_test$|conversation_restore_orchestration_test$)' --output-on-failure

python3 "$R/run_gpu_tests.py" hip --uuid GPU-e39306ee46993aad,GPU-2f182f01261ba30b --label hip-r2-final-sdma0
nvidia-smi -L
lspci -nn -s 04:00.0
python3 "$R/cuda_probe.py" cuda --label cuda-r2 --small
python3 "$R/finalize_evidence.py"
python3 "$R/package_incremental.py"
```

The native driver reuses the earlier serial bounded harness, changing only the
command selection to the 12 R2 and 10 remote-owner cases. Each executable has a
60-second deadline plus five-second kill grace, and a kernel check after each case.
The final native run is `hip-r2-final-sdma0-results.json`; the earlier bring-up run
is retained separately and is not additional unique coverage. Final CPU logs are
`cpu-final-tests.log` and `sanitizers-final-tests.log`, 63/63 each.

## Artifacts

- `review.patch`: updated **full worktree patch against HEAD**, including untracked source.
  SHA-256: `8418546962721a09a0966d6135cbf8ceb972d6e7bbd25f0e3f58171432c8bcc6`.
- `R2_VALIDATED_INCREMENTAL.patch`: R2 plus native fixture/docs, relative to the
  previously validated dirty worktree (`chatgpt-delta/review.patch`, SHA-256
  `9feb7df844004ff376a9b2cc557b02cfe7173a43c2266e3f3f52edf1d15901f8`).
  SHA-256: `54d7f88056a06de4bd818196fdf2810bc9bf3bb8738afbbbb9d2d6d548ea9d6d`.
- Both output patches passed applicability checks using a separate index against
  their respective bases. Authoritative index remains empty. Do not reapply either
  patch to the already-updated authoritative worktree.
- `evidence/source-identity.json` and `executable-identities.json` record source and
  production binary hashes. Native fixture binary hash is in `native-executable.sha256`.

## Limits / next loop

NVIDIA is still PCI-visible at 04:00.0 but NVML reports an unknown handle error;
CUDA runtime probes report no CUDA-capable device. Compile success is not CUDA
runtime qualification. No native CUDA optimization teardown was executed.
No R2 SDMA-on pass, real-model multi-GPU run, free/get-device error injection,
physically failed GPU, concurrent teardown, long-context test, performance comparison
or SYCL/Metal qualification is claimed. The native fixture is model-independent;
no new model download or full-model R2 test was needed for this bounded pass.

R1 remains for the next shared loop; these R2 results do not validate PeerExperts
teardown or its caller-owned loans. RemoteExperts::close and cache startup remain
outside the production scope of this delta. Advisor tool discovery returned no
available advisor. No earlier audit or Phase 11A preservation pass was repeated.

No commit, push, GitHub Actions, reset, driver change or production model overwrite.
Strata-dev remains clean; the authoritative Phase 11B worktree remains uncommitted.
