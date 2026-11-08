# ChatGPT delta integration — 2026-11-07

## Integration

Authoritative worktree: `/home/chris/workspace/Strata-worktrees/phase11b-prefill-restore`.
Branch: `fix/phase11b-prefill-restore-lifetime`; HEAD: `0ef75af1730c9374bbf1cb66a361f410d699b805`.
Existing R4/R5/R6 dirty implementation retained. Original implementation/review.patch hash verified as
`204994c9196d71b35044cf2b279477e7b6cbfb5d1756907486462e06106074b9`.

Copied `/home/chris/tmp/strata/STRATA_CHATGPT_IMPLEMENTATION.patch` here. SHA-256 verified:
`1531f9a2207b13ceb8dc1af2ada402e23470e5abfd51cf2358c1cb341b11ec11`.
`git apply --check` passed in the actual dirty worktree; applied once with `git apply`.
No integration source corrections were needed. Reverse applicability and whitespace checks passed.
No Phase 11A preservation investigation was repeated.

## Newly executed validation

| Check | Result |
|---|---|
| CPU async-join group | 8/8 passed, including all four new throwing-worker cases |
| CPU relevant regression group | 59/59 passed |
| Same regression group, ASan/UBSan and leak detection | 59/59 passed |
| HIP production `strata` and selected native targets | Built successfully, existing gfx1201 configuration |
| CUDA production `strata`, snapshot and lifetime targets | Built successfully, existing CUDA 12.0 / sm_89 configuration |
| R9700 #0, serial, SDMA off | 19/19 selected native processes passed |
| HIP snapshot CTest variants: ordinary, checkpoint async, VRAM slots | 3/3 passed; 4377 checks each |
| Real model disk SAVE/RESTORE, MTP off | Passed; 118279704-byte file; reference/restored continuation both `T 1`; exit 0 |
| Real model disk SAVE/RESTORE, MTP on | Passed; 118296184-byte file; reference/restored continuation both `T 1`; exit 0 |
| Final-source short SDMA-on HIP group | 3/3 passed: lifetime success, issue-drain, pinned ring |
| Native CUDA short group | 0 passed, 3 skipped (exit 77), no CUDA-capable device |

The new subprocess tests assert child exit 1 and EOF on the owner/resumption marker pipe, not just an error return.
The snapshot test covers no-MTP streamed/captured byte equivalence, bounded reads, validation, restore and cross-mode rejection; the MTP control remains.
No `SERR` occurred in either model transcript, including no `invalid K/V extent` in the no-MTP run.
Model settings remain the earlier bounded Qwen Q2_0 configuration: context 128, prefill 32, 128 expert slots, CPU prefill share off, greedy one-token requests. Sequence: generate, save, reference continuation, switch prompt, restore, identical continuation.
No kernel faults matched the existing harness's MES/TLB/page-fault/reset checks during native or model runs.

## Actual command summary

All shell calls had explicit harness timeouts. Build/test output is under `evidence/`.

```sh
W=/home/chris/workspace/Strata-worktrees/phase11b-prefill-restore
R=/home/chris/workspace/Strata-phase11b/chatgpt-delta
B=/home/chris/workspace/Strata-builds
cd "$W"
git apply --check "$R/STRATA_CHATGPT_IMPLEMENTATION.patch"
git apply "$R/STRATA_CHATGPT_IMPLEMENTATION.patch"
git diff --check

cmake --build "$B/phase11b-hip" --target strata conversation_snapshot_test conversation_lifetime_gpu_test conversation_readback_test pinned_ring_gpu_test -j 4
cmake --build "$B/phase11b-cuda" --target strata conversation_snapshot_test conversation_lifetime_gpu_test -j 4

# Run separately for D=phase11b-cpu and D=phase11b-sanitizers:
cmake --build "$B/$D" --target transfer_failure_test conversation_lifetime_cuda_test conversation_lifetime_hip_test prefill_completion_test pinned_ring_test conv_cache_test conversation_restore_orchestration_test -j 4
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 ctest --test-dir "$B/$D" -R '^(transfer_|lifetime_|prefill_completion_test$|pinned_ring_test$|conv_cache_test$|conversation_restore_orchestration_test$)' --output-on-failure

python3 "$R/run_gpu_tests.py" hip --uuid GPU-e39306ee46993aad --label hip-final-sdma0
python3 "$R/model_smoke.py" --no-mtp
python3 "$R/model_smoke.py"
python3 "$R/run_gpu_tests.py" cuda --label cuda-final --small
python3 "$R/run_gpu_tests.py" hip --uuid GPU-e39306ee46993aad --label hip-final-sdma1 --small --sdma 1
LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib ROCR_VISIBLE_DEVICES=GPU-e39306ee46993aad HSA_ENABLE_SDMA=0 ctest --test-dir "$B/phase11b-hip" -R '^conversation_snapshot_test' -j 1 --output-on-failure -V
nvidia-smi -L
lspci -nn -s 04:00.0
```

Reused existing GPU/model harnesses, copied locally. Model harness changes only add `--no-mtp` (removing the `--mtp` pair) and separate output directories; no downloaded models or dependency changes. An initial invocation used unavailable `python` (exit 127); reran with `python3` successfully. Initial focused async/lifetime runs also used the existing cpu-r4r5-initial and cpu-r4r5-initial-asan build directories (8/8 and 32/32 respectively); these overlap the final 59-test groups and are not additional unique coverage.

## Deliverable and limits

Updated **full worktree patch against HEAD**, including untracked source files: `review.patch`.
SHA-256: `9feb7df844004ff376a9b2cc557b02cfe7173a43c2266e3f3f52edf1d15901f8`.
Generated using the existing finalize_evidence.py, copied here. Verified patch applicability against HEAD using a separate temporary index; the authoritative index remains empty. File and executable identities are under `evidence/`.

NVIDIA remains at PCI 04:00.0, but NVML reports an unknown device-handle error and CUDA runtime probes skip. No driver/reset changes attempted. CUDA compilation is not runtime qualification.
The short SDMA-on result is final-source component coverage on one card, not long-context/model SDMA-on qualification. HIP checkpoint-async test names do not establish execution of CUDA-only async workers.
RV-01 prefill local/borrowed cleanup audit, real-model borrowed/converted failure scenarios, broader backend coverage (including SYCL/Metal), long contexts and performance remain outstanding. No fresh TSan or Compute Sanitizer qualification is claimed. Advisor tool discovery returned no available advisor.
No commit, push, GitHub Actions, driver change, production model overwrite or Strata-dev source modification.
