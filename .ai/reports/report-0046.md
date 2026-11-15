# Report 0046 — bounded M5–M7 candidate qualification

## Result

- **PASS — focused CPU tests and native builds.** No in-scope source defect was demonstrated; no source changes were needed.
- **PARTIAL — GPU and real-model qualification.** The user stated an HTTP server is running and explicitly prohibited interrupting it. Exclusive GPU availability and permission to use that server cannot be assumed, even though the local process/listener check did not identify it. No GPU test or request to the user server was admitted. The model-free guarded normal, nonfinite and withheld-publication cases, worker retirement, baseline/fast model comparison and HTTP smoke remain unverified. GPU testing time: zero. No M5/M6/M7 completion or throughput claim is made.

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

Private logs, including the failed invocation, are under `/home/chris/tmp/strata-heterogeneous/handoff-0046/` (created with `umask 077`). On a future exclusive, authorized GPU window, first run the three separate guarded model-free probe processes and verify both consumers and normal worker exits as specified in handoff 0046; stop on failure or cleanup uncertainty. Only then consider a bounded same-topology resident comparison. Do not enable real-model guarded asynchronous prelaunch based on this report.
