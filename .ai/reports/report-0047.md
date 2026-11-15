# Report 0047 — bounded real-model guarded prelaunch

## Status

**PASS for the specified bounded fixed-token GPU comparisons and focused local validation.** Guarded CUDA→HIP prelaunch and legacy direct verified shared produced byte-identical complete full-logit STFR packets for T=1/keep=1 and three overlapping T=4/keep=2 windows. Tracing on the one-row guarded run shows native prelaunch, pending arm and consumption. This is not production admission, state-parity proof, failure-retirement qualification, or a throughput claim. HTTP, full A/B/B/A, long-context, deliberate real-model fault injection and broader numerical sweeps were not run. No in-scope source defect was demonstrated; no source changes were needed.

## Provenance and safety

Tested source `6fe46938ac99ad5fb0f2414f353d56736c45188b` on `feature/heterogeneous-hip-only`. Existing Release caches, `/usr/bin/nvcc` sm_89, `/opt/rocm/core-10.1/lib/llvm/bin/clang++` gfx1201, checkout `.venv` Python 3.12.3. Rebuilt host coordinator, CUDA worker and HIP worker SHA-256 respectively:

- `6e2e0d38e91ef90e14a659717f726f93ae19e1376d49c53c36f1379dff12a5e0`
- `27b95936e566b47e4a02701dec10bf7259c3a904eacd3e8d78ba45e0d12e9cd4`
- `2c2fa586a9c3b7c1a90b067383a821842a41f577a512e784730cacbf4b80d935`

Before tests, `ss -ltnp`, process inventory, `nvidia-smi`, `amd-smi process` and `amd-smi metric -m` showed no local Strata HTTP listener or GPU compute process; snapshot only, not an exclusive reservation. No user server was contacted or interrupted. No GitHub Actions, installs, model or vanilla checkout changes, clock/driver changes, spills, reserve lowering or forced kills. Untracked `tools/heterogeneous_pool_probe.cpp` remains untouched. Private evidence in `/home/chris/tmp/strata-heterogeneous/handoff-0047/`, created with `umask 077`.

## Build, CPU validation and inspection

Commands (builds sequential, each 28 jobs):

```sh
cmake --build build-256k-host --parallel 28 --target strata_heterogeneous_live_smoke strata_heterogeneous_final_pipeline_tests strata_heterogeneous_host_tests strata_heterogeneous_handoff_guard_tests strata_heterogeneous_window_admission_tests strata_heterogeneous_window_admission_process_tests strata_heterogeneous_cuda_verifier_contract_tests strata_heterogeneous_cuda_final_verifier_contract_tests strata_heterogeneous_hip_verifier_contract_tests strata_heterogeneous_hip_final_verifier_contract_tests strata_heterogeneous_direct_verified_binding_tests
cmake --build build-256k-cuda --parallel 28 --target strata_heterogeneous_cuda_worker strata_heterogeneous_live_smoke
cmake --build build-256k-hip --parallel 28 --target strata_heterogeneous_hip_worker strata_heterogeneous_live_smoke
ctest --test-dir build-256k-host --output-on-failure -R 'strata_heterogeneous_(final_pipeline|host|handoff_guard|window_admission|window_admission_process|cuda_verifier_contract|cuda_final_verifier_contract|hip_verifier_contract|hip_final_verifier_contract|direct_verified_binding)_tests'
env PYTHONPATH=tools .venv/bin/python -m unittest -v tools.test_heterogeneous_model_gate tools.test_heterogeneous_hip_only_cli
```

Builds passed in 2.12/5.95/5.90 s (host/CUDA/HIP); compiler warnings are retained in logs. CTest 10/10 in 8.62 s; Python 27/27 in 0.20 s. CPU-only native preflight passed for the guarded configuration and rejected guarded without direct, guarded resident without generation, and direct plus external receiver-arm; outputs and commands/configuration are in `preflight.log`. Existing final-pipeline fixtures exercise ordinary and legacy direct paths, mismatched legacy HELLO rejection, corrupt producer poisoning, commit rejection, and slot reuse after ACKs.

Code inspection: coordinator uses distinct guarded HELLO kinds and verifies endpoint identity; it waits for the exact receiver arm ACK before producer dispatch. HIP arm resets the external ready word only for an admitted window after prior commit, fences, launches a captured bounded wait and requires stream-not-ready before ACK. Both HIP chain verifier nodes receive guarded mode; their incoming R/bo/inj paths use checked wait plus conditional zero copy. Internal outgoing ready and error words reset during staging. On run, both native handoff failures are checked after stream synchronization before a provisional result; host metadata, finite and checksum gates and all endpoint commit ACKs remain required. Failure poisons sockets and worker state, and abort releases waits before freeing resources. This inspection and CPU fixtures do not replace real-GPU fault/retirement testing.

## GPU captures

All four commands used identical 2048-cell, max_t=8, Q2_0 full-resident mixed stage configs, INT8 KV, 1 GiB reserve per owner, `HSA_ENABLE_SDMA=0`, with inherited `STRATA_*` overrides cleared. Common command (actual shell-escaped commands saved in `t1-reference.command`, `t1-guarded.command`, `t4-reference.command`, `t4-guarded.command`):

```sh
build-256k-host/strata_heterogeneous_live_smoke \
 --plan cuda:0:0-15,hip:0:15-33,hip:1:33-48 --fingerprint 2026111301 \
 --config /home/chris/tmp/strata-heterogeneous/qualification-256k/mixed-2048/stage0.cfg \
 --config /home/chris/tmp/strata-heterogeneous/qualification-256k/mixed-2048/stage1.cfg \
 --config /home/chris/tmp/strata-heterogeneous/qualification-256k/mixed-2048/stage2.cfg \
 --cuda-worker "$PWD/build-256k-cuda/experimental/heterogeneous/strata_heterogeneous_cuda_worker" \
 --hip-worker "$PWD/build-256k-hip/experimental/heterogeneous/strata_heterogeneous_hip_worker" \
 --shared --group-native-hip --hip-split-overlap --direct-verified-shared --kv int8 \
 SHAPE [--guarded-direct-prelaunch] --capture-prefix ABSOLUTE_UNIQUE_PREFIX --run
```

`SHAPE` was `--rows 1 --keep 1 --windows 1 --tokens 248045`, then `--rows 4 --keep 2 --windows 3 --tokens 248045,846,198,2523,198,2523,513,32767,513,32767,264,198`; reference omitted and candidate included the bracketed option, in separate processes. Only the T=1 candidate additionally set `STRATA_VERIFY_TRACE=1 STRATA_HETERO_OVERLAP_TRACE=1 STRATA_HETERO_PERF_TRACE=1`. All four returned 0 and normal LIVE PASS. One-row accepted prediction `[846]`; three-window accepted prefixes `[846,198]`, `[96220,513]`, `[449,264]` in both modes. Corresponding manifests were byte-identical including plan fingerprint `16764715816942409987`, request/window positions, mode and token IDs; pack-index SHA-256 `c344c01f59f547ed8a59003857a82eb90499e5299154241dbca787acb54b5b7c`.

Entire STFR comparisons, including discarded rows and all logits (`cmp`, exit 0 per pair):

| Shape / window | Bytes per packet | SHA-256, both modes |
|---|---:|---|
| T1 / 1 | 993352 | `af75c8d5440c4f5ae196e09dfcfa7c78e5dc8963067342256f561ea5a5a3a6e4` |
| T4 / 1 | 3973204 | `c2cef11b76eeb409605d7044288933f122bb947ae004d4b210df9222aa0011c7` |
| T4 / 2 | 3973204 | `3aec0ab1efa41d4bfc8f7b55e4b2fc5d230085d0584c928fa8483c464eb4496e` |
| T4 / 3 | 3973204 | `2941394dc8702c759256524087ef68cc7c1a5ffcc82cc1fbb26115f2b38acee5` |

The traced candidate logged `HETERO_RECEIVER_ARMED`, coordinator arm ACK ending at monotonic ns `48286094872247`, and CUDA stage-spec send beginning at `48286094893737`. Native trace showed HIP0 `WINDOW-PRE`, `LAUNCHED`, `EXTERNAL-ARMED`, then `PRELAUNCH-CONSUMED`; HIP1 also showed prelaunch consumption. These are host-side ordering observations, not GPU kernel timings. The pending stream check is in the arm implementation, not an independently measured GPU timeline.

All runs logged full expert residency CUDA/HIP0/HIP1 7680/7680, 9216/9216, 7680/7680, resident KV and no host routes/misses. Graph-capture free snapshots 2753/16246/18510 MiB exceed reserve but are not instantaneous peak measurements. Four process wall times 9.30, 9.25, 9.39 and 9.36 s (37.30 s total admitted GPU work, excluding CPU inspection), under five minutes. No native GPU worker or GPU compute process survived. No real-model rejection, cancellation/drain on fault, committed native-state comparison, other shapes/topologies, HTTP, or throughput gain is established; candidate remains explicitly bounded and disabled in normal profiles.
