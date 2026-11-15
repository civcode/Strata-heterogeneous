# Report 0053 — short requests with 256K-configured resident MTP

## Result and fixes

**PASS for bounded HIP-only and mixed HTTP MTP smoke at configured 262144 cells**, after one integration repair; this is not production, long-position correctness, transient peak, or long-prompt performance qualification. Source synced to `9c60799d66e8f371adbb5535c1af8dd023281650`; changes in this report's commit were tested as described below. Unrelated untracked `tools/heterogeneous_pool_probe.cpp` remains untouched.

The initial HIP attempt loaded both full-resident target owners but then safely refused: a stale long-context worker gate unconditionally rejected overlap and MTP even for the new resident service. Updated it to admit those only for the service while continuing to reject extended diagnostic overlap/MTP, receiver arm and direct verified paths. Added a 64-combination CPU policy test of that gate; existing service-specific validation remains in place. The initial failed process drained and exited without surviving workers. Also corrected a stale native-prefill-stage test which expected an eight-token seed to fail (the limit was expanded in earlier batches); it now tests a prompt at its 2048-cell context boundary instead.

## Builds and CPU tests

Read `.ai/TEST_BUDGET.md`. Existing vanilla Release caches and `.venv`, 28 jobs; built host coordinator, CUDA/HIP workers and focused tests, then rebuilt all affected host/CUDA/HIP targets after the fix:

```sh
cmake --build build-256k-host --parallel 28 --target strata_heterogeneous_live_smoke strata_heterogeneous_final_pipeline_tests strata_heterogeneous_host_tests strata_heterogeneous_window_admission_tests strata_heterogeneous_hip_final_verifier_contract_tests strata_heterogeneous_hip_verifier_contract_tests strata_heterogeneous_mtp_prefix_cursor_tests strata_heterogeneous_mtp_request_policy_tests strata_heterogeneous_mtp_stream_batch_tests strata_heterogeneous_mtp_bulk_prompt_tests strata_heterogeneous_long_context_tests strata_heterogeneous_native_prefill_stage_contract_tests strata_heterogeneous_hip_worker_bootstrap_tests strata_heterogeneous_cuda_worker_bootstrap_tests
cmake --build build-256k-cuda --parallel 28 --target strata_heterogeneous_cuda_worker strata_heterogeneous_live_smoke
cmake --build build-256k-hip --parallel 28 --target strata_heterogeneous_hip_worker strata_heterogeneous_live_smoke
ctest --test-dir build-256k-host --output-on-failure -R 'strata_heterogeneous_(final_pipeline|host|window_admission|hip_final_verifier_contract|hip_verifier_contract|mtp_prefix_cursor|mtp_request_policy|mtp_stream_batch|mtp_bulk_prompt|long_context|native_prefill_stage_contract|hip_worker_bootstrap|cuda_worker_bootstrap)_tests'
env PYTHONPATH=tools .venv/bin/python -m unittest serve.test_heterogeneous_resident tools.test_heterogeneous_hip_only_cli tools.test_heterogeneous_mtp_stream_contract tools.test_heterogeneous_mtp_qualification tools.test_heterogeneous_mtp_generation tools.test_heterogeneous_long_context_config tools.test_heterogeneous_long_context_lifecycle tools.test_heterogeneous_long_context
```

Final result **13/13 CTest, 54/54 Python**; request-policy **85,498 assertions**, bulk **699,146 assertions**. Initial extra CTest run was 12/13 due to the stale eight-token expectation above. Tests cover full-size metadata/ACK wire roundtrip and budget, grouped mock 262143-token metadata/native-prefill deadline/reset, immutable prompt and successor mapping, stale capabilities, mode and exact worker capacity refusal, plus ordinary long-context and worker bootstrap contracts. Full-size cases are CPU-only.

## GPU execution

Fresh process/listener and NVIDIA/AMD process snapshots found no active Strata service or GPU compute process; AMD 0/1 each used 57 MB. This is not a reservation. Private copies of report-0052's five stage configs differed only by `max_cells=8192` -> `262144`; originals untouched. Used pinned Q2_0/Q8 `/home/chris/workspace/Strata-data/mtp/rt`, INT8 KV, full experts, max_t 8, >=1 GiB reserve, spec 4/min-p 0.5, chunk 32, explicit workload and long-context, cleared `STRATA_*`, `HSA_ENABLE_SDMA=0`. CPU-only `prepare_server` preflight passed before worker launches. Private scripts, configs, logs and captures: `/home/chris/tmp/strata-heterogeneous/handoff-0053/`.

```sh
env PYTHONPATH="$PWD" .venv/bin/python /home/chris/tmp/strata-heterogeneous/handoff-0053/bulk_check.py hip bulk
env PYTHONPATH="$PWD" .venv/bin/python /home/chris/tmp/strata-heterogeneous/handoff-0053/http_check.py
```

HIP 24/24 used report-0052's 77-token nonuniform prompt and budget 16. Resident A/cancel/A seeded target chunks 32/32/12 as prime cells 0,8,...,64 with eight rows each and cell 72 with four. Empty checkpoint at position 76 cancelled with empty final IDs; reset/recovery A produced the same 16 streamed/final IDs as first A, each with **12 proposed/12 accepted**. Final prompt verifier checkpoint reached 77; maximum committed position was **92**. All target experts (12288/12288 per owner) and INT8 KV were resident; zero host expert routes/misses. HIP1 admission logged `context_bytes=420482832`, free 10,110,369,792 bytes, estimate 3,465,350,928, base reserve 1,073,741,824. On first request stage 0 prefill before/after allocation was 10,280,239,104/9,938,403,328 bytes; terminal stage 1 was 8,218,738,688/7,876,902,912 with augmented reserve 1,610,612,736; terminal after drain was 7,753,170,944. Snapshots are not peak measurements.

Mixed 15/18/15 used a newly launched private loopback HTTP server on an OS-assigned port and a short chat prompt. HTTP 200 `text/event-stream` yielded four JSON events, `Hello!`, finish `stop` and `[DONE]`; 19 prompt tokens, prefix 18, prime rows 8/8/2, 1 proposed/1 accepted, committed position 21. All three owners reported 262144 INT8 resident KV, full experts and zero host routes/misses. Mixed terminal admission free 18,106,810,368 bytes, same 420,482,832 context and estimate; terminal prefill before/after allocation 16,215,179,264/15,877,537,792 against 1,610,612,736 reserve. CUDA prefill drained free 1,555,038,208 against 1,073,741,824 reserve. No existing service contacted.

Successful HIP process wall **17.35 s**, mixed HTTP process wall **17.04 s**, total **34.39 s**; the initial failed target-load attempt was not independently timed, so total admitted wall including that attempt is unknown. Successful runs logged `HETERO_CLOSED`; HTTP received cooperative SIGTERM and exited 0. Final GPU process inventory was empty. No forced kills, long/near-capacity GPU prompts, 4K prompt, injected faults, changes to assets/drivers/clocks/reserves/defaults or GitHub Actions. High actual positions, peak memory, failure retirement and broader behavior remain unverified.
