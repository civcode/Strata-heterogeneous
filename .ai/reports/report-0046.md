# Report 0046 — bounded M5–M7 candidate qualification

## Result

- **PASS — focused CPU tests and native builds.** No in-scope source defect was demonstrated; no source changes were needed.
- **PASS — bounded model-free guarded probe and short resident baseline/fast smoke**, as detailed below. This is not M5/M6/M7 completion, production qualification, strict numerical parity, or a robust throughput comparison.
- **PARTIAL — broader qualification:** no full A/B/B/A matrix, HTTP smoke, or real-model guarded asynchronous prelaunch was attempted. The latter is explicitly outside this handoff; an HTTP smoke was optional and no request was sent to a user server.

## Provenance and safety

Branch `feature/heterogeneous-hip-only`; tested source `2ac895dabd5deb0356b96c8e967fe893d28583ca` (fast-forwarded from `bc734657`). Existing Release build directories and checkout venv Python 3.12.3; CUDA `/usr/bin/nvcc` sm_89, HIP `/opt/rocm/core-10.1/lib/llvm/bin/clang++` gfx1201. No model, driver, clock, reserve, vanilla checkout, or unrelated service changes. Untracked `tools/heterogeneous_pool_probe.cpp` was left untouched. No GitHub Actions, long-context stress or extended benchmark. The CPU HTTP regression starts its own ephemeral-port mock server, not the user's service.

Built executable SHA-256:

| Executable | SHA-256 |
|---|---|
| `build-256k-host/strata_heterogeneous_live_smoke` | `ffae0437a3bc61acff3593fed4c316d19bd71a95ca46106cb99cb59cf07f35ea` |
| `build-256k-cuda/experimental/heterogeneous/strata_heterogeneous_cuda_worker` | `64a068db3c716059e5715b5896cf684bfd48db147a185b5773fd89730916cceb` |
| `build-256k-hip/experimental/heterogeneous/strata_heterogeneous_hip_worker` | `0e8ee9bae8cf6fd7ad705e25c6155c01a9abaf0e5cec9e558780ca5de064524c` |
| `build-256k-cuda/experimental/heterogeneous/strata_heterogeneous_direct_graph_probe` | `1f76c3f9c67e727a58e5b9d3be9910abc604d3310d22a10c77a03949fd7b375d` |
| `build-256k-hip/experimental/heterogeneous/strata_heterogeneous_direct_graph_probe` | `1731914f109905b276b4629a7ccc0b61ebc2d1f29cb6395e190010427352d41f` |

## Commands and results

All build commands used 28 jobs, sequentially, with existing caches:

```sh
cmake --build build-256k-host --parallel 28 --target strata_heterogeneous_live_smoke strata_heterogeneous_final_pipeline_tests strata_heterogeneous_handoff_guard_tests strata_heterogeneous_host_tests strata_heterogeneous_process_tests strata_heterogeneous_window_tests strata_heterogeneous_shared_tests strata_heterogeneous_native_pipeline_window_tests strata_heterogeneous_window_admission_tests strata_heterogeneous_window_admission_process_tests
cmake --build build-256k-cuda --parallel 28 --target strata_heterogeneous_cuda_worker strata_heterogeneous_direct_graph_probe strata_heterogeneous_live_smoke
cmake --build build-256k-hip --parallel 28 --target strata_heterogeneous_hip_worker strata_heterogeneous_direct_graph_probe strata_heterogeneous_live_smoke
ctest --test-dir build-256k-host --output-on-failure -R 'strata_heterogeneous_(final_pipeline|handoff_guard|host|process|window|shared|native_pipeline_window|window_admission)(_process)?_tests'
env PYTHONPATH=tools .venv/bin/python -m unittest -v tools.test_heterogeneous_direct_graph_probe tools.test_heterogeneous_resident_bench_summary tools.test_heterogeneous_hip_only_cli tools.test_heterogeneous_quick_smoke serve.test_heterogeneous_resident serve.test_context_overflow
```

Builds passed: host 1.97 s, CUDA 37.55 s, HIP 42.21 s; compiler warnings are retained in logs. CTest passed 9/9 in 8.87 s. Python unittest passed 49 tests in 0.89 s including profile, smoke fixtures and HTTP completion-cap regressions. Initial attempt to run the same Python files via `.venv/bin/python -m pytest -q ...` failed immediately: `No module named pytest`; these are unittest tests and the successful command above used the existing venv without installing anything.

## Subsequent GPU checks

Initially no GPU work was admitted because of the user's running-server warning. After the user requested a new availability check and completion of the handoff, `ss -ltnp`, `ps`, `nvidia-smi`, `amd-smi process`, `amd-smi metric -m`, and `fuser` showed no local Strata HTTP listener or GPU compute process; the two 32 GB AMD devices each had 57 MB used and NVIDIA had 15 MiB used (Xorg). This was a snapshot, not an exclusivity guarantee. No request was sent to any existing server.

Two fresh model-free processes were run, each including four normal verified rounds and a separate terminal rejection case:

```sh
root=$PWD
for mode in nonfinite timeout; do
  .venv/bin/python tools/heterogeneous_direct_graph_probe.py \
    --cuda-worker "$root/build-256k-cuda/experimental/heterogeneous/strata_heterogeneous_direct_graph_probe" \
    --hip-worker "$root/build-256k-hip/experimental/heterogeneous/strata_heterogeneous_direct_graph_probe" \
    --cuda-device 0 --hip-devices 0 1 --rows 1 --rounds 4 --warmup 1 \
    --wait-ms 1000 --guarded --exercise-$mode \
    --output "/home/chris/tmp/strata-heterogeneous/handoff-0046/gpu-$mode.json"
done
```

Both passed, with 51,216-byte payloads verified, nonfinite publication rejected in one process and withheld publication rejected in the other; the native probe verifies bit-exact zeros on both HIP consumers for each rejection. Both workers returned BYE and exit 0 in each process. Host wall times were 0.31 s and 1.27 s; normal-path median observations were 0.203 ms and 0.240 ms respectively, including IPC and readback, not kernel-only or model throughput.

A short mixed CUDA/HIP Q8 resident baseline then fast comparison used the same rebuilt binaries, 32K capacity, 7 prompt tokens, 32 outputs and two requests per process (first warmup). The exact shell-escaped commands are preserved as `resident-baseline.command` and `resident-fast.command` in the private log directory; their common invocation was:

```sh
HSA_ENABLE_SDMA=0 .venv/bin/python strata.py heterogeneous-generate \
  --coordinator "$PWD/build-256k-host/strata_heterogeneous_live_smoke" \
  --hip-worker "$PWD/build-256k-hip/experimental/heterogeneous/strata_heterogeneous_hip_worker" \
  --cuda-worker "$PWD/build-256k-cuda/experimental/heterogeneous/strata_heterogeneous_cuda_worker" \
  --plan cuda:0:0-15,hip:0:15-33,hip:1:33-48 --fingerprint 2026111301 \
  --config /home/chris/tmp/strata-heterogeneous/qualification-256k/mixed-32768/stage0.cfg \
  --config /home/chris/tmp/strata-heterogeneous/qualification-256k/mixed-32768/stage1.cfg \
  --config /home/chris/tmp/strata-heterogeneous/qualification-256k/mixed-32768/stage2.cfg \
  --prompt-file /home/chris/tmp/strata-heterogeneous/handoff-0045/short.txt \
  --max-new-tokens 32 --resident --max-context 32768 --shared --group-native-hip \
  --stream-tokens --kv int8 --workload-mode --native-prefill-chunk 128 \
  --benchmark-repeat 2 --pipeline-mode MODE --output json --run
```

`MODE` was `baseline` then `fast`; inherited `STRATA_*` overrides were cleared. Pack-index SHA-256 `c344c01f59f547ed8a59003857a82eb90499e5299154241dbca787acb54b5b7c`. Both runs exited 0, logged `HETERO_CLOSED`, full expert residency (7680/7680, 9216/9216, 7680/7680), resident KV on all stages, no host expert routes/misses, and 1 GiB reserve per stage. Graph-capture free snapshots were 2611/15974/18272 MiB; these do not establish transient peak headroom. All four requests reported 7 committed prompt tokens, 31 committed decode feedback tokens and 32 length-finished outputs. Native finite gates did not reject any window; this is not an independent full-logit finite/parity capture. All four token-ID arrays had SHA-256 `3c2c9d6e845ef5843e0b97ca679a585d2d8ae259814e10163bc671d55dae5b27` (compact JSON encoding). Warmed request elapsed observations: baseline 827.27 ms; fast 607.92 ms. Process wall times: 12.02 s and 11.51 s. These single observations are not robust throughput estimates. Afterward no native workers or GPU compute processes were visible. Total admitted GPU wall time was approximately 25.1 seconds, below the five-minute budget.

Private logs, including the failed pytest invocation, exact commands, JSON and GPU outputs, are under `/home/chris/tmp/strata-heterogeneous/handoff-0046/` (created with `umask 077`). Do not enable real-model guarded asynchronous prelaunch based on this report.
