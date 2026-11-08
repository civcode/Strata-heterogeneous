# Fresh test matrix and commands

Base: `0ef75af1730c9374bbf1cb66a361f410d699b805`, plus the uncommitted review.patch.
Patch SHA-256: `204994c9196d71b35044cf2b279477e7b6cbfb5d1756907486462e06106074b9`.
Per-source and executable hashes: evidence/source-identity.json and executable-identities.json.

## Results

| Group | Passed | Failed / blocked | Skipped | Evidence |
|---|---:|---:|---:|---|
| Relevant CPU CTests | 55 | 0 | 0 | final-cpu-correction-test.log |
| ASan/UBSan + leak detection, same selection | 55 | 0 | 0 | final-sanitizers-correction-test.log |
| TSan selected concurrency/future tests | 0 | 2 startup failures | 0 | tsan-test.log |
| Corrected CUDA selected native processes | 0 | 0 | 3 (no device) | cuda-reviewed-runtime-results.json |
| Corrected HIP #0, SDMA off | 19 | 0 | 0 | hip-reviewed-device0-sdma0-results.json |
| Corrected HIP #1, SDMA off | 19 | 0 | 0 | hip-reviewed-device1-sdma0-results.json |
| Earlier corrected HIP SDMA on, #0 | 3 | 0 | 0 | hip-final-device0-sdma1-results.json |
| Earlier corrected HIP SDMA on, #1 | 3 | 0 | 0 | hip-final-device1-sdma1-results.json |
| Actual HIP model disk restore / continued token | 1 scenario | 0 | 0 | hip-model-result.json |
| Actual HIP Prefill one-shot final-copy error | 1 expected-exit scenario | 0 | 0 | r4-model-terminal/hip-model-result.json |
| Actual HIP Prefill persistently reported error | 1 expected-exit scenario | 0 | 0 | r4-model-unknown/hip-model-result.json |
| Compute Sanitizer memcheck | 0 | tool startup failure, exit 13 | 0 | cuda-compute-sanitizer.log |
| Full CUDA/HIP application builds | 2 | 0 final | — | {cuda,hip}-final-app-build.log |
| Tracked/new-file whitespace | pass | 0 | — | working-diff-check.txt; finalize_evidence.py |

Nineteen native processes include multiple internal checks; they are not nineteen engine requests.
The 55 CPU CTests comprise the earlier 26 relevant cases, one new Prefill completion case and 28 new lifetime/async-join cases.

The final CPU/sanitizer selection and final healthy/failing model scenarios were rerun after cancellation-guard corrections. Native component binaries exercise the same corrected copy/runtime code; the final subsequent Prefill-only tracing-device RAII refinement was compiled on both backends and its actual model terminal boundary was rerun. Multi-GPU tracing/helper runtime is not claimed.

## Exact CPU build and test selection

All shell commands used explicit harness timeouts. Configure/build/test batches were bounded at 120–240 s;
individual new CTests have 30/40-s limits. GPU runner commands use timeout --kill-after=5 60.
No ASLR, toolchain, driver or kernel setting was changed.

```sh
W=/home/chris/workspace/Strata-worktrees/phase11b-prefill-restore
B=/home/chris/workspace/Strata-builds/phase11b-cpu
cmake -S "$W" -B "$B" -G Ninja -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_NATIVE_EXPERTS=OFF -DCMAKE_BUILD_TYPE=Debug
cmake --build "$B" --parallel 4 --target pinned_ring_test transfer_failure_test \
  conv_cache_test conversation_restore_orchestration_test prefill_completion_test \
  conversation_lifetime_cuda_test conversation_lifetime_hip_test
env -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' \
  ctest --test-dir "$B" \
  -R '^(pinned_ring_test|transfer_(remote-|stager-|query-).*|conv_cache_test|conversation_restore_orchestration_test|prefill_completion_test|lifetime_.*)$' \
  --output-on-failure
```

Sanitizer configure:

```sh
B=/home/chris/workspace/Strata-builds/phase11b-sanitizers
cmake -S "$W" -B "$B" -G Ninja -DSTRATA_BUILD_TESTS=ON \
  -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF -DSTRATA_NATIVE_EXPERTS=OFF \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS='-Wall -Wextra -fsanitize=address,undefined -fno-omit-frame-pointer -fno-pie' \
  -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address,undefined -no-pie'
```

Build the same seven targets and run the same selection with
`ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1`.
Detailed actual build/configure output is retained; CPU fakes do not establish DMA safety.

TSan used the same CPU configure structure, a separate phase11b-tsan directory,
`-fsanitize=thread -fno-omit-frame-pointer -fno-pie` and link flags `-fsanitize=thread -no-pie`.
Built transfer_failure_test and conversation_lifetime_cuda_test, then ran:

```sh
env -u LD_PRELOAD CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' \
  ctest --test-dir /home/chris/workspace/Strata-builds/phase11b-tsan \
  -R '^(transfer_stager-concurrent|lifetime_cuda_async-join-ok)$' --output-on-failure
```

Both failed at TSan startup with unexpected memory mapping. No concurrency pass is inferred.

## Backend builds

Final CUDA configuration, in addition to the common Release/test switches:

```sh
cmake -S "$W" -B /home/chris/workspace/Strata-builds/phase11b-cuda -G Ninja \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler \
  -DSTRATA_NATIVE_EXPERTS=ON \
  -DSTRATA_GGML_DIR=/home/chris/workspace/Strata-builds/phase11b-ggml-source \
  -DSTRATA_BUILD_TESTS=ON -DSTRATA_BUILD_CONVERSATION_TESTS=ON -DCMAKE_BUILD_TYPE=Release
```

The matching cached ggml input was copied to that isolated location; not fetched or modified in the original checkout.

HIP:

```sh
cmake -S "$W" -B /home/chris/workspace/Strata-builds/phase11b-hip -G Ninja \
  -DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DCMAKE_PREFIX_PATH=/opt/rocm/core-10.1 \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_GGML_DIR=/home/chris/workspace/llama.cpp \
  -DSTRATA_BUILD_TESTS=ON -DSTRATA_BUILD_CONVERSATION_TESTS=ON -DCMAKE_BUILD_TYPE=Release
```

Built `strata conversation_lifetime_gpu_test conversation_snapshot_test conversation_readback_test pinned_ring_gpu_test conversation_prepare_test`
with `cmake --build BUILD --target ... --parallel 4`.
HIP additionally built transfer_gpu_test and transfer_prefill_inject.
Final source includes R4/R5/R6 and CUDA compatibility changes.

## Serial native commands

The checked-in-to-report runner records every executable command, environment, return code and kernel window.
It stops at the first runtime or relevant kernel failure. It does not retry faults.

```sh
R=/home/chris/workspace/Strata-phase11b/implementation
python3 "$R/run_gpu_tests.py" hip --uuid GPU-e39306ee46993aad --label hip-reviewed-device0-sdma0
python3 "$R/run_gpu_tests.py" hip --uuid GPU-2f182f01261ba30b --label hip-reviewed-device1-sdma0
python3 "$R/run_gpu_tests.py" hip --uuid GPU-e39306ee46993aad --sdma 1 --small --label hip-final-device0-sdma1
python3 "$R/run_gpu_tests.py" hip --uuid GPU-2f182f01261ba30b --sdma 1 --small --label hip-final-device1-sdma1
python3 "$R/run_gpu_tests.py" cuda --small --label cuda-reviewed-runtime
python3 "$R/model_smoke.py"
python3 "$R/model_smoke.py" --fault
python3 "$R/model_smoke.py" --fault-unknown
```

GPU tests remove LD_PRELOAD except the explicit actual-model fault modes. HIP runs select a single UUID and set
LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib, HSA_ENABLE_SDMA=0 (or 1 for the short comparison).
model_smoke.py contains and records the complete model command; it never updates original models/profile inputs.

## Failures and non-results retained

- Valid R5 red: checkpoint-header issue-drain test aborts, exit 134; corrected production header passes.
- Initial red fixture had a cvec/cache selection error; do not count it as the defect reproduction.
- Early CPU HIP readback issue-error test incorrectly expected failed bytes to publish; fixed to assert no publication.
- An older native regression expected unknown completion to return; updated to assert actual process exit.
- Initial loader/CMake-prefix errors, incompatible ggml snapshots and native-experts-off application link errors are retained in logs.
- Initial model CLI/spec/profile errors are retained. No-MTP disk SAVE fails at the application validation gate.
- The first final-copy interposer run missed source selection; its apparent healthy result is explicitly invalid coverage (r4-model-initial-missed-injection.json). Final runners require selection, injected error, ERR/exit 1 and no T/DONE.
- No native CUDA execution, hardware fault recovery, full serving borrowed-prefix matrix, long-context stress,
  all-target suite, Metal/SYCL build or before/after performance benchmark was completed.
