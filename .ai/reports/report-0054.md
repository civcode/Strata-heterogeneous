# Report 0054 — fail-closed publication and retirement

## Result

**PASS for focused CPU fault/retirement tests and short 256K-configured HIP resident cancellation/recovery plus private SSE completion.** Tested source `a87b9bc0d170e1a0d580bf84dcd38dc577cd353d`; no source corrections were needed. This is not production, high-position correctness, native GPU fault recovery, or transient peak-memory qualification. Unrelated untracked `tools/heterogeneous_pool_probe.cpp` was preserved.

## Builds and CPU checks

Read `.ai/TEST_BUDGET.md`; used existing vanilla Release caches and checkout `.venv`, 28 parallel jobs. Built host coordinator and report-0053 focused targets, CUDA/HIP workers and coordinators, plus `strata_heterogeneous_window_admission_process_tests` and `strata_heterogeneous_process_tests`:

```sh
cmake --build build-256k-host --parallel 28 --target strata_heterogeneous_live_smoke strata_heterogeneous_final_pipeline_tests strata_heterogeneous_host_tests strata_heterogeneous_window_admission_tests strata_heterogeneous_hip_final_verifier_contract_tests strata_heterogeneous_hip_verifier_contract_tests strata_heterogeneous_mtp_prefix_cursor_tests strata_heterogeneous_mtp_request_policy_tests strata_heterogeneous_mtp_stream_batch_tests strata_heterogeneous_mtp_bulk_prompt_tests strata_heterogeneous_long_context_tests strata_heterogeneous_native_prefill_stage_contract_tests strata_heterogeneous_hip_worker_bootstrap_tests strata_heterogeneous_cuda_worker_bootstrap_tests strata_heterogeneous_window_admission_process_tests strata_heterogeneous_process_tests
cmake --build build-256k-cuda --parallel 28 --target strata_heterogeneous_cuda_worker strata_heterogeneous_live_smoke
cmake --build build-256k-hip --parallel 28 --target strata_heterogeneous_hip_worker strata_heterogeneous_live_smoke
ctest --test-dir build-256k-host --output-on-failure -R 'strata_heterogeneous_(final_pipeline|host|window_admission|window_admission_process|process|hip_final_verifier_contract|hip_verifier_contract|mtp_prefix_cursor|mtp_request_policy|mtp_stream_batch|mtp_bulk_prompt|long_context|native_prefill_stage_contract|hip_worker_bootstrap|cuda_worker_bootstrap)_tests'
env PYTHONPATH=tools .venv/bin/python -m unittest serve.test_heterogeneous_resident serve.test_heterogeneous_retirement tools.test_heterogeneous_hip_only_cli tools.test_heterogeneous_mtp_stream_contract tools.test_heterogeneous_mtp_qualification tools.test_heterogeneous_mtp_generation tools.test_heterogeneous_long_context_config tools.test_heterogeneous_long_context_lifecycle tools.test_heterogeneous_long_context
```

All builds passed; **15/15 CTest, 62/62 Python**. Fork/mock MTP failures on both grouped topologies cover metadata, proposal, prefill, reset and observer failure after target commit; each preserves the expected underlying refusal text, poisons the session, forbids reset/publication, and exits the final worker nonzero. HIP zero-keep tail test rejects before sending commits, then provisional EOF retires nonzero. Host process-supervision tests passed. Adapter fixtures check malformed completion identity, cancelled-ACK mismatch, stale timing clearing, physical tail anchor-only progress, reader/timeout failure, live or unverifiable process-group refusal, and no restart over an old group; retirement tests also exercise real cooperative CPU subprocess exits 0 and 7. These are not real-GPU fatal fault injections.

## GPU and cooperative retirement

Fresh process/listener, NVIDIA/AMD process and memory checks showed no active Strata listener or GPU compute process; AMD owners used 57 MB each. This was a snapshot, not an exclusive reservation. Reused report-0053 private 262144-cell HIP 24/24 configs unchanged, pinned Q2_0/Q8 `/home/chris/workspace/Strata-data/mtp/rt`, max_t 8, INT8 KV, full target/draft residency, >=1 GiB reserve, spec 4/min-p 0.5, native chunk 32 and explicit workload/long-context; cleared `STRATA_*`, set `HSA_ENABLE_SDMA=0`. Private evidence: `/home/chris/tmp/strata-heterogeneous/handoff-0054/`.

```sh
env PYTHONPATH="$PWD" .venv/bin/python /home/chris/tmp/strata-heterogeneous/handoff-0054/bulk_check.py hip bulk
env PYTHONPATH="$PWD" .venv/bin/python /home/chris/tmp/strata-heterogeneous/handoff-0054/http_check.py
```

The resident driver ran A/cancel/A using the 77-token nonuniform report-0053 prompt and budget 16. It explicitly set the ResidentEngine cancellation event when the committed empty checkpoint `HETERO_PROGRESS 2 1 76 BATCH 0 0` was yielded; the cancellation ACK had no IDs. Request 3 reset and reproduced request 1's 16 streamed/final IDs, each proposed/accepted 12/12. Maximum committed position **92**. The adapter's cooperative `close()` completed, its reader exited and original coordinator process group was verified absent by the retirement helper; the driver independently probed group absence. No forced signal was used for this resident process.

The separate newly launched private loopback HTTP smoke returned HTTP 200 SSE, four JSON events, `Hello!`, finish `stop`, and `[DONE]`; prompt 19, final position 21, 1/1 proposed/accepted. It shut down cooperatively on SIGTERM, exited 0 and logged `HETERO_CLOSED`. **No HTTP disconnect cancellation is claimed**; cancellation evidence is the explicit verified ResidentEngine checkpoint above. No existing service was contacted.

HIP logs showed full 12288/12288 experts and resident INT8 KV on both owners, no host routes/misses. Draft admission reported context bytes 420,482,832, free 10,110,369,792, estimate 3,465,350,928 and reserve 1,073,741,824 bytes. On the first request terminal prefill free before/after allocation was 8,218,738,688/7,876,902,912 against augmented reserve 1,610,612,736. Snapshots do not establish transient peaks. Both GPU process sets logged `HETERO_CLOSED`; final process and compute checks were empty. Wall time **17.72 s resident + 20.10 s SSE = 37.82 s**. No long prompts, near-capacity GPU requests, GPU fault injection, forced kills, dependency/asset/driver/clock changes, spill, reserve reduction, active-config changes or GitHub Actions. HTTP disconnect timing, high actual positions, abnormal GPU drain and transient peak remain unverified.
