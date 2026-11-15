# Report 0048 — request-local resident Q8 MTP candidate

## Status

**PASS for bounded GPU smoke and CPU integration; PARTIAL for broader qualification.** Host/CUDA/HIP builds and focused CPU tests passed, pinned runtime preflight passed, and subsequent short resident Q8 MTP reference comparison and A/B/A completed on hardware. Initially GPU work was deferred because the user stated an HTTP server was running. The user subsequently authorized testing if fresh checks confirmed Oberon idle; the checks and results are below. No request was sent to an existing server. No source defect was demonstrated; no source changes were made. This limited evidence is not production admission.

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

Code inspection and fixtures show a distinct terminal service HELLO, one request metadata exchange after reset and before prompt windows, exact prompt-slice and budget checks on coordinator and observer, all-domain prefix commit ACKs, and reset of draft state before a new request. `resident_kv_supported()` requires native Q8 code/scale pointers, dense page table with all pages resident, and excludes F16 pools, host and elastic modes; the previous non-service resident path retains its F16 check. The later successful GPU startup and request resets exercised these native checks; they are not an independent dump of the physical memory layout.

## Remaining work and safety

The optional separate HTTP test and committed-boundary cancellation/recovery were not run. Wider prompt/stop patterns, native fault retirement, independent physical state parity and production behavior remain unverified. No long-context tests, extended benchmarks, GitHub Actions, installs, downloads, repacks, forced GPU kills, driver/clock changes, reserve reductions, or vanilla checkout changes. Preserved unrelated untracked `tools/heterogeneous_pool_probe.cpp`.

## Subsequent GPU qualification

Following explicit user authorization, fresh `ss -ltnp`, process inventory, `nvidia-smi`, `amd-smi process` and `amd-smi metric -m` showed no local Strata server or GPU compute process. NVIDIA was 0% with 15 MiB used (Xorg); both 32 GB AMD devices had 57 MB used. This is a snapshot, not an exclusive reservation. The binary hashes above were rechecked; no rebuild was performed. Private commands, stdout/stderr, JSON and the small A/B/A driver are in the evidence directory.

Reference and MTP each used a fresh process set, the exact mixed-2048 stage configs, INT8, prompt chunk 8, seven input IDs `[23066,23066,23066,23066,23066,23066,23066]`, 32 outputs and two requests (first warmup). Exact shell-escaped commands are in `gpu-reference.command` and `gpu-mtp.command`; common invocation:

```sh
HSA_ENABLE_SDMA=0 .venv/bin/python strata.py heterogeneous-generate \
 --coordinator "$PWD/build-256k-host/strata_heterogeneous_live_smoke" \
 --hip-worker "$PWD/build-256k-hip/experimental/heterogeneous/strata_heterogeneous_hip_worker" \
 --cuda-worker "$PWD/build-256k-cuda/experimental/heterogeneous/strata_heterogeneous_cuda_worker" \
 --plan cuda:0:0-15,hip:0:15-33,hip:1:33-48 --fingerprint 2026111301 \
 --config /home/chris/tmp/strata-heterogeneous/qualification-256k/mixed-2048/stage0.cfg \
 --config /home/chris/tmp/strata-heterogeneous/qualification-256k/mixed-2048/stage1.cfg \
 --config /home/chris/tmp/strata-heterogeneous/qualification-256k/mixed-2048/stage2.cfg \
 --prompt-file /home/chris/tmp/strata-heterogeneous/handoff-0045/short.txt \
 --max-new-tokens 32 --prompt-chunk 8 --resident --max-context 2048 --shared --group-native-hip --kv int8 \
 --benchmark-repeat 2 [--mtp /home/chris/workspace/Strata-data/mtp/rt --spec 4 --spec-min-p 0.5] --output json --run
```

Inherited `STRATA_*` settings were cleared. The candidate alone included bracketed options. The CLI enables HIP split overlap for MTP but not the reference, so this is not a matched performance comparison. Both completed requests 1 and 2 with length finish, 7 committed prompt tokens, 31 committed feedback tokens and 32 output IDs. All four token arrays were identical: `[23066,23066,23066,23066,23066,23066,23066,23066,23066,23066,23066,23066,23066,487,328,2083,871,198,262,1992,1698,27896,437,64,292,272,293,378,281,338,304,585]`; compact-JSON SHA-256 `c3d7acaf7bdb445fbcaa78018a4a9ac5432399b6479b68c4759b488d8883e0ac`. MTP proposed 33 and accepted 18 draft tokens in each request. Warm request host elapsed was 746.16 ms reference, 579.13 ms MTP; single observations with different overlap settings, not throughput estimates. Process wall times were 11.19 and 16.48 s, including startup and shutdown.

A/B/A used one new resident MTP process through `env PYTHONPATH="$PWD" .venv/bin/python /home/chris/tmp/strata-heterogeneous/handoff-0048/aba.py` with the same binaries/configs, fixed spec=4/0.5, prompt chunk 8, and variable budgets. Request 1 A and request 3 A used the seven-token prompt above, budget 16; both length-finished with identical IDs `[23066,23066,23066,23066,23066,23066,23066,23066,23066,23066,23066,23066,23066,487,328,2083]`, 15 feedback tokens, 13 proposed/11 accepted. Request 2 B used `[248045,846,198,2523]`, budget 24, stop ID 248046, 13 output IDs `[513,449,14791,17313,49110,440,21156,264,6150,4939,23014,23850,13]`, 13 feedback tokens, 19 proposed/5 accepted. Reset was logged for requests 1, 2 and 3. All committed positions and stop/length metadata were validated by `ResidentEngine`; `HETERO_CLOSED` and process-group disappearance were checked. Successful run wall time 16.30 s.

The first A/B/A driver run ended after B due to **our test-driver assertion**, not native failure: buffered `ResidentEngine.generate` intentionally emits the stop token after the completion IDs. The driver failed to exclude it when comparing against the completion record. Logs `gpu-aba.log`/`.out` retain the failure (16.08 s); the native process closed cleanly, and no GPU workers remained. The private driver was corrected to exclude stop IDs, and the complete A/B/A retry passed. Total admitted GPU process wall time was approximately 60.05 s, below five minutes.

Candidate logs report all target experts resident (CUDA/HIP0/HIP1 7680/7680, 9216/9216, 7680/7680), INT8 target KV, and no expert host routes or misses. Each stage configured 1 GiB reserve. Draft admission on HIP1 logged 19,409,141,760 bytes free, 3,044,868,096-byte conservative estimate, 1,073,741,824-byte reserve; draft loaded with 1,019,533,980 bytes reported VRAM. Successful startup and resets exercised the native Q8 dense code/scale and no-host/no-elastic checks; allocator snapshots do not prove instantaneous peak headroom. Post-test `nvidia-smi`, `amd-smi process` and process inventory showed no native GPU compute workers. No existing HTTP server was contacted or interrupted.
