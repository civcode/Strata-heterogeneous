# Report 0048 — request-local resident Q8 MTP candidate

## Status

**PARTIAL.** Host/CUDA/HIP integration builds, focused CPU protocol/policy and Python tests passed, and CPU preflight verified the pinned runtime files. Real resident Q8 MTP, physical draft allocations/reserve, acceptance and reset on GPU remain **NOT RUN**. The user explicitly stated an HTTP server is running and prohibited interruption; handoff 0048 says not to run GPU tests while a user service is active. Although the local process/listener and GPU snapshots did not identify that server or a GPU compute task, that is not sufficient to override the user's statement. No request was sent to any server and no GPU test was admitted (GPU time zero). No source defect was demonstrated; no source changes were made. The candidate is not hardware-qualified or production-ready.

## Provenance and local validation

Tested source `0df1d5633f1e48f86491e62f1fcc4f016d56359e` on `feature/heterogeneous-hip-only`; existing Release caches, CUDA `/usr/bin/nvcc` sm_89, HIP `/opt/rocm/core-10.1/lib/llvm/bin/clang++` gfx1201, checkout `.venv` Python 3.12.3. Binaries SHA-256:

- `build-256k-host/strata_heterogeneous_live_smoke`: `959c68b202619f664b10062fe61a6d3f749200b411cd592003677408c01a8148`
- `build-256k-cuda/experimental/heterogeneous/strata_heterogeneous_cuda_worker`: `434612537ac01b6b2fbcd8f52cc0e2c53b2c56649ca8362015140f6f29c4848b`
- `build-256k-hip/experimental/heterogeneous/strata_heterogeneous_hip_worker`: `94ef9e828de2acad8387742378f1e2f208efa05f902003e98218b1de773607d0`

Commands, sequential builds with 28 jobs:

```sh
cmake --build build-256k-host --parallel 28 --target strata_heterogeneous_live_smoke strata_heterogeneous_final_pipeline_tests strata_heterogeneous_host_tests strata_heterogeneous_mtp_prefix_cursor_tests strata_heterogeneous_mtp_request_policy_tests strata_heterogeneous_hip_final_verifier_contract_tests strata_heterogeneous_hip_verifier_contract_tests strata_heterogeneous_window_admission_tests
cmake --build build-256k-cuda --parallel 28 --target strata_heterogeneous_cuda_worker strata_heterogeneous_live_smoke
cmake --build build-256k-hip --parallel 28 --target strata_heterogeneous_hip_worker strata_heterogeneous_live_smoke
ctest --test-dir build-256k-host --output-on-failure -R 'strata_heterogeneous_(final_pipeline|host|mtp_prefix_cursor|mtp_request_policy|hip_final_verifier_contract|hip_verifier_contract|window_admission)_tests'
env PYTHONPATH=tools .venv/bin/python -m unittest -v serve.test_heterogeneous_resident tools.test_heterogeneous_hip_only_cli tools.test_heterogeneous_mtp_resident
env PYTHONPATH=tools .venv/bin/python -m unittest -v tools.test_heterogeneous_mtp_admission tools.test_heterogeneous_mtp_generation tools.test_heterogeneous_mtp_qualification tools.test_heterogeneous_mtp_three_round
```

Builds passed (host 2.14 s, CUDA 21.89 s, HIP 11.31 s). CTest 7/7 in 8.61 s; Python 30/30 and 25/25. These include legacy and new protocol fixtures, but cannot prove GPU behavior. Compiler output and full test logs are private under `/home/chris/tmp/strata-heterogeneous/handoff-0048/` (created with `umask 077`).

CPU-only native preflight command, no `--run`:

```sh
build-256k-host/strata_heterogeneous_live_smoke --plan cuda:0:0-15,hip:0:15-33,hip:1:33-48 --fingerprint 2026111301 \
 --config /home/chris/tmp/strata-heterogeneous/qualification-256k/mixed-2048/stage0.cfg \
 --config /home/chris/tmp/strata-heterogeneous/qualification-256k/mixed-2048/stage1.cfg \
 --config /home/chris/tmp/strata-heterogeneous/qualification-256k/mixed-2048/stage2.cfg \
 --resident --shared --group-native-hip --hip-split-overlap --kv int8 --generate 32 --prompt-chunk 8 --context-limit 2048 \
 --stop-ids 248044,248046 --source-gguf /home/chris/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
 --source-index c344c01f59f547ed8a59003857a82eb90499e5299154241dbca787acb54b5b7c \
 --mtp-resident-service --mtp-probe-rt /home/chris/workspace/Strata-data/mtp/rt
```

Passed in 2.34 s: three 2048-cell/max_t=8 INT8 stage configurations and the existing runtime files met the built-in pinned size/SHA-256 validator. This is asset identity and configuration preflight, not actual allocation or full expert residency evidence. Target conversion checkpoint provenance remains unresolved.

Code inspection and fixtures show a distinct terminal service HELLO, one request metadata exchange after reset and before prompt windows, exact prompt-slice and budget checks on coordinator and observer, all-domain prefix commit ACKs, and reset of draft state before a new request. `resident_kv_supported()` requires native Q8 code/scale pointers, dense page table with all pages resident, and excludes F16 pools, host and elastic modes; the previous non-service resident path retains its F16 check. Actual native pointer layout, memory admission and teardown cannot be confirmed without a GPU run.

## Remaining work and safety

On an authorized exclusive GPU window, run handoff 0048's short serial reference versus resident MTP and request-local A/B/A, recording actual proposal/acceptance counts, generated IDs, reserve, full residency, reset and normal shutdown; stop on failure/uncertain drain. The optional separate HTTP test was not run. No long-context tests, benchmarks, GitHub Actions, installs, downloads, repacks, forced GPU kills, driver/clock changes, reserve reductions, or vanilla checkout changes. Preserved unrelated untracked `tools/heterogeneous_pool_probe.cpp`.
