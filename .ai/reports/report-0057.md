# Report 0057 — opt-in batched native MTP draft seed

## Result

**PASS for focused integration and one short paired mixed GPU comparison; two real 32-row draft batches executed.** Tested source `b0c59b4f6a99fa62b9357e18e33e62c49a0c0533`; no source corrections were required. The batch option remains opt-in and default narrow mode unchanged. No claim of speedup, production readiness, high-position correctness or GPU fault recovery. Preserved unrelated untracked `tools/heterogeneous_pool_probe.cpp`.

## Builds and tests

Read `.ai/TEST_BUDGET.md`. Existing vanilla Release caches/toolchain, `.venv`, 28 jobs:

```sh
cmake --build build-256k-host --parallel 28 --target strata_heterogeneous_live_smoke strata_heterogeneous_final_pipeline_tests strata_heterogeneous_host_tests strata_heterogeneous_window_admission_tests strata_heterogeneous_hip_final_verifier_contract_tests strata_heterogeneous_hip_verifier_contract_tests strata_heterogeneous_mtp_prefix_cursor_tests strata_heterogeneous_mtp_request_policy_tests strata_heterogeneous_mtp_stream_batch_tests strata_heterogeneous_mtp_bulk_prompt_tests strata_heterogeneous_long_context_tests strata_heterogeneous_native_prefill_stage_contract_tests strata_heterogeneous_hip_worker_bootstrap_tests strata_heterogeneous_cuda_worker_bootstrap_tests
cmake --build build-256k-cuda --parallel 28 --target strata strata_heterogeneous_cuda_worker strata_heterogeneous_live_smoke
cmake --build build-256k-hip --parallel 28 --target strata strata_heterogeneous_hip_worker strata_heterogeneous_live_smoke
ctest --test-dir build-256k-host --output-on-failure -R 'strata_heterogeneous_(final_pipeline|host|window_admission|hip_final_verifier_contract|hip_verifier_contract|mtp_prefix_cursor|mtp_request_policy|mtp_stream_batch|mtp_bulk_prompt|long_context|native_prefill_stage_contract|hip_worker_bootstrap|cuda_worker_bootstrap)_tests'
env PYTHONPATH=tools .venv/bin/python -m unittest serve.test_heterogeneous_resident serve.test_heterogeneous_retirement tools.test_heterogeneous_mtp_stream_contract tools.test_heterogeneous_hip_only_cli tools.test_heterogeneous_oberon_launch serve.test_heterogeneous_http_disconnect
```

All builds passed, including normal CUDA/HIP `strata` downstream callers; existing compiler warnings were emitted. **13/13 CTest, 53/53 Python**; bulk **1,040,444** and policy **86,271** assertions. CPU tests include actual-successor slices, decline/narrow fallback and no fallback after batch failure/poison.

Source review: terminal prefill alone passes its own `Prefill` object and an explicitly borrowed device-local drafter embedding; `draft_kv` verifies matching device and full resident plain INT8 state. It checks scratch size before mutation, uses bounded actual successor IDs and cell offsets, drains the draft stream before appending and synchronizes the prefill stream before success. A clean decline returns to <=8-row graph seeding; nonempty error or failed batch poisons rather than retrying potentially written cells. Tests and one short output comparison are not independent physical KV-state parity or failure-injection proof.

## Paired 256K-configured mixed smoke

Fresh process/listener and GPU checks found no active Strata service or compute process; AMD 0/1 each used 57 MB. Snapshot is not a reservation. Sequential private resident processes used exact report-0053 mixed 15/18/15 262144-cell stage configs, pinned Q2_0/Q8 runtime `/home/chris/workspace/Strata-data/mtp/rt`, full experts, INT8 KV, max_t 8, >=1 GiB reserve, native chunk 32, spec 4/min-p 0.5, explicit workload/long context and `HSA_ENABLE_SDMA=0`. Driver clears all `STRATA_*`, sets only `STRATA_HETERO_MTP_BATCH=1` for candidate, and leaves `STRATA_MTP_BATCH` unset. Same rebuilt binaries for both. Native resident protocol was selected to guarantee exact token IDs. Evidence under `/home/chris/tmp/strata-heterogeneous/handoff-0057/`.

```sh
env PYTHONPATH="$PWD" .venv/bin/python /home/chris/tmp/strata-heterogeneous/handoff-0057/paired.py baseline
env PYTHONPATH="$PWD" .venv/bin/python /home/chris/tmp/strata-heterogeneous/handoff-0057/paired.py batch
```

Both used the identical 77-token nonuniform prompt (`23065` at zero-based index mod 11 == 5, otherwise `23066`), budget 16. Actual bulk prefix 76 comprised chunks 32,32,12. Baseline logged narrow prime cells 0,8,16,24,32,40,48,56,64 with 8 rows each and cell 72 with 4. Candidate logged **`HETERO_MTP_BULK_BATCH_PRIMED cell=0 rows=32` and `cell=32 rows=32`**, then narrow `cell=64 rows=8`, `cell=72 rows=4`; no decline. Both completed at committed position 92, proposed/accepted 12/12, streamed output matched final records and the 16 output IDs were identical across paths (`23065` at output index 5, otherwise `23066`). This does not establish general numerical parity.

Submit-to-first empty bulk checkpoint, same host boundary excluding startup but including request reset and cold prefill: baseline **864.23 ms**, candidate **856.48 ms**. Single observations are not reliable timing or speedup evidence. No 513-token case was planned or run. Process wall baseline **14.25 s**, candidate **14.35 s**, total **28.60 s**. Both closed with `HETERO_CLOSED`, their coordinator process groups were absent, and final NVIDIA/AMD compute inventory was empty.

All three stages logged full resident experts and zero host routes/misses. HIP1 admission free 18,106,810,368, estimate 3,465,350,928, context bytes 420,482,832, base reserve 1,073,741,824. Candidate terminal prefill allocation left 15,873,343,488 bytes against augmented 1,610,612,736 reserve; CUDA allocation left 1,580,204,032 against 1,073,741,824. These snapshots do not establish transient peak headroom. No longer or high-position GPU request, injected failure, forced kill, asset/driver/clock changes, spill, reserve reduction, default profile change or GitHub Actions.
