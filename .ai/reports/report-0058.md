# Report 0058 — bounded draft-prefill characterization

## Status

**PASS for request-local accounting and a short paired 513-token mixed GPU characterization.** Source `dcd6b01ab69bd39ebafb911d9b17b9549eee85f9`; no corrections needed. Batch selection remains opt-in; these two warmed samples do not establish a general speedup, production readiness, high-position correctness or GPU fault recovery. Unrelated untracked `tools/heterogeneous_pool_probe.cpp` was preserved.

## Builds and CPU tests

Read `.ai/TEST_BUDGET.md`; used existing matching Release caches, toolchain and checkout `.venv`, 28 jobs. Rebuilt affected HIP worker and new host accounting test; unchanged coordinator/CUDA binaries were reused:

```sh
cmake --build build-256k-host --parallel 28 --target strata_heterogeneous_live_smoke strata_heterogeneous_mtp_bulk_seed_stats_tests strata_heterogeneous_mtp_bulk_prompt_tests strata_heterogeneous_mtp_request_policy_tests strata_heterogeneous_mtp_prefix_cursor_tests strata_heterogeneous_final_pipeline_tests strata_heterogeneous_native_prefill_stage_contract_tests
cmake --build build-256k-hip --parallel 28 --target strata_heterogeneous_hip_worker
ctest --test-dir build-256k-host --output-on-failure -R 'strata_heterogeneous_(mtp_bulk_seed_stats|mtp_bulk_prompt|mtp_request_policy|mtp_prefix_cursor|final_pipeline|native_prefill_stage_contract)_tests'
```

Builds passed, **6/6 CTest**; accounting **42**, bulk pairing **1,040,444**, policy **86,271** assertions. Accounting commits only complete cursor chunks and separates batch/narrow/decline callback wall intervals; reset isolation and failure paths are CPU-tested. No new device synchronization is added by stats.

## GPU setup and results

Fresh process/listener and NVIDIA/AMD compute snapshots showed no active Strata service or GPU compute process; AMD owners each used 57 MB. This is not a reservation. Two sequential private mixed 15/18/15 resident processes used the exact report-0057 262144-cell configs, Q2_0/Q8 `/home/chris/workspace/Strata-data/mtp/rt`, INT8 KV, max_t 8, full expert maps, >=1 GiB reserves, chunk 32, spec 4/min-p 0.5, explicit long/workload, `HSA_ENABLE_SDMA=0`. Driver cleared `STRATA_*`; baseline left batch unset, candidate set `STRATA_HETERO_MTP_BATCH=1`; trace was unset in both. Identical binaries, prompt and request budgets; no existing service contacted. Private driver and raw records/logs: `/home/chris/tmp/strata-heterogeneous/handoff-0058/`.

```sh
env PYTHONPATH="$PWD" .venv/bin/python /home/chris/tmp/strata-heterogeneous/handoff-0058/measure.py baseline
env PYTHONPATH="$PWD" .venv/bin/python /home/chris/tmp/strata-heterogeneous/handoff-0058/measure.py batch
```

Each process ran one warmup then two measured requests: identical 513 nonuniform token IDs (`23065` at zero-based index mod 11 == 5, otherwise `23066`), max-new 16. Every request reset normally, produced one empty checkpoint at 512 and completed at position **528**, length, 16 output IDs, proposed/accepted **12/11**. Streamed IDs matched final records, were stable across resets and were identical across paths. Each successful bulk emitted exactly one summary and no per-prime bulk flood; decode prime logs remained. All summaries had request IDs 1,2,3, 512 rows and 16 chunks. Baseline had 64 narrow calls/512 rows and 16 declines each; candidate had 16 batch calls/512 rows, zero narrow calls and zero declines each. No stale counts appeared across resets.

Raw wall-time samples below are nanoseconds from terminal callbacks, plus host milliseconds from request submission to first empty checkpoint (includes reset, excludes process startup). Callback time includes bookkeeping; component times are nested within it. They are not GPU kernel-only times. Request 1 is warmup, not a measured repeat.

| Path / request | Submit→checkpoint ms | callback_ns | batch_ns | narrow_ns | decline_ns |
|---|---:|---:|---:|---:|---:|
| narrow warmup 1 | 1442.333 | 58,442,605 | 0 | 58,437,915 | 812 |
| narrow measured 2 | 867.543 | 19,654,255 | 0 | 19,649,689 | 541 |
| narrow measured 3 | 866.107 | 19,613,223 | 0 | 19,609,661 | 392 |
| batch warmup 1 | 1423.446 | 21,814,183 | 21,811,978 | 0 | 0 |
| batch measured 2 | 865.764 | 5,153,290 | 5,150,103 | 0 | 0 |
| batch measured 3 | 869.256 | 5,073,390 | 5,070,823 | 0 | 0 |

Callback timing is lower in these samples, while host checkpoint intervals are effectively similar; no throughput or aggregate speedup conclusion is warranted from this small sequential sample.

All owners reported full expert maps, resident INT8 KV and zero host routes/misses. HIP1 admission free 18,106,810,368 bytes, context 420,482,832, estimate 3,465,350,928 and reserve 1,073,741,824. Warmed stage-0 prefill allocated free 1,540,358,144 against 1,073,741,824 reserve; terminal free 15,709,765,632 against augmented 1,610,612,736. These snapshots are not transient peak measurements. Both processes closed normally with `HETERO_CLOSED`, coordinator groups absent and no remaining GPU compute process. Wall **17.04 s baseline + 17.02 s batch = 34.06 s**. No >=4K or near-capacity GPU prompts, fault injection, forced kills, asset/driver/clock changes, spill, reserve reduction, default changes or GitHub Actions.
