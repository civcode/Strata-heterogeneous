# Report 0052 — aligned 8K resident MTP candidate

## Result

**PASS for focused integration and short 8192-cell-configured HIP-only and mixed MTP smoke.** Tested source `d8428f361382fcfb42ea9cf8f481721b51ebfd9e` on `feature/heterogeneous-hip-only`; no source repair was required. Actual positions reached only 92: this is not evidence of correctness near 8K, production readiness, 256K MTP admission, or transient peak headroom. Unrelated untracked `tools/heterogeneous_pool_probe.cpp` was preserved.

## Build and CPU tests

Read `.ai/TEST_BUDGET.md`; existing vanilla toolchain, Release caches, checkout `.venv`, 28 jobs:

```sh
cmake --build build-256k-host --parallel 28 --target strata_heterogeneous_live_smoke strata_heterogeneous_final_pipeline_tests strata_heterogeneous_host_tests strata_heterogeneous_window_admission_tests strata_heterogeneous_hip_final_verifier_contract_tests strata_heterogeneous_hip_verifier_contract_tests strata_heterogeneous_mtp_prefix_cursor_tests strata_heterogeneous_mtp_request_policy_tests strata_heterogeneous_mtp_stream_batch_tests strata_heterogeneous_mtp_bulk_prompt_tests
cmake --build build-256k-cuda --parallel 28 --target strata_heterogeneous_cuda_worker strata_heterogeneous_live_smoke
cmake --build build-256k-hip --parallel 28 --target strata_heterogeneous_hip_worker strata_heterogeneous_live_smoke
ctest --test-dir build-256k-host --output-on-failure -R 'strata_heterogeneous_(final_pipeline|host|window_admission|hip_final_verifier_contract|hip_verifier_contract|mtp_prefix_cursor|mtp_request_policy|mtp_stream_batch|mtp_bulk_prompt)_tests'
env PYTHONPATH=tools .venv/bin/python -m unittest serve.test_heterogeneous_resident tools.test_heterogeneous_hip_only_cli tools.test_heterogeneous_mtp_stream_contract tools.test_heterogeneous_mtp_qualification tools.test_heterogeneous_mtp_generation
```

All builds passed, **9/9 CTest and 45/45 Python**. Standalone built tests report **85,405 request-policy** and **400,130 bulk policy/pairing assertions**. CPU checks exercise aligned capacity, budget and near-boundary proposal guards, real successor pairing, wire shape and limits, stale capability and exact-capacity refusal. Source review confirms wire metadata remains within the existing 8192-token window; terminal policy uses the actual configured capacity, with >2K requiring explicit bulk. Full expert and reserve checks are applied to both CUDA and HIP service owners. No near-capacity GPU requests or fault injection.

## GPU admission and observations

Fresh listener/process and NVIDIA/AMD compute snapshots showed no active Strata listener or GPU compute process; AMD owners used 57 MB each. This is not an exclusive reservation. Private copies of all five report-0051 2K stage configs were checked with `diff`: each differs only by `max_cells=8192`; originals and defaults were not edited. Used same pinned Q2_0 and Q8 `/home/chris/workspace/Strata-data/mtp/rt`, full resident expert maps, INT8 KV, spec 4/min-p 0.5, cleared `STRATA_*`, `HSA_ENABLE_SDMA=0`, max_t 8 and >=1 GiB reserve. Logs, JSON results, config copies and driver are in `/home/chris/tmp/strata-heterogeneous/handoff-0052/`.

```sh
env PYTHONPATH="$PWD" .venv/bin/python /home/chris/tmp/strata-heterogeneous/handoff-0052/bulk_check.py hip bulk
env PYTHONPATH="$PWD" .venv/bin/python /home/chris/tmp/strata-heterogeneous/handoff-0052/bulk_check.py mixed bulk
```

Both used the same nonuniform 77-token prompt as report 0051 (`23065` at zero-based indices congruent to 5 mod 11, otherwise `23066`), chunk 32, completion budget 16. HIP 24/24 ran A/cancel/A in one resident process. Each request seeded 76 rows as target chunks 32/32/12, with prime cells 0 through 64 in increments of 8, eight rows each, then cell 72 four rows. Empty checkpoint `HETERO_PROGRESS 2 1 76 BATCH 0 0` was cancelled with empty final IDs; reset and A recovery matched the first 16-ID streamed/final record. Both A requests proposed/accepted **12/12**; final verifier checkpoint advanced to 77, final committed position 92. Mixed CUDA/HIP/HIP 15/18/15 completed the same prompt and output with **12/12**, matching streamed and final records. These equal outputs do not prove full state parity.

All owners reported `max_cells=8192`, resident INT8 KV and full expert routes with zero host routes/misses. HIP1 draft admission logged `context_bytes=13143824`, free 11,983,126,528 bytes, estimate 3,058,011,920 and base reserve 1,073,741,824. First HIP stage 0 prefill free before/after allocation was 12,152,995,840/11,876,171,776 bytes; terminal HIP1 was 10,500,440,064/10,223,616,000 with reserve 1,610,612,736 (including 512 MiB prime headroom); terminal drained free 10,099,884,032. Mixed terminal prefill before/after was 17,894,998,016/17,618,173,952 with the same augmented reserve; CUDA early-commit free was 2,819,620,864 bytes against 1 GiB reserve. These snapshots are not instantaneous peaks.

Both process sets closed cooperatively with `HETERO_CLOSED`; no native workers or GPU compute processes remained. Wall time **16.86 s HIP + 13.76 s mixed = 30.62 s**. No HTTP smoke (mixed alternative used), existing service contact, long GPU request, near-capacity GPU test, forced kill, dependency/asset/driver/clock changes, spill, reserve reduction or GitHub Actions. Further validation requires actual positions beyond 2048, failure retirement and transient peak measurement.
