# Report 0050 — bounded native bulk MTP prompt seed

## Status

**PASS for the bounded opt-in 2K smoke and focused tests**, not production or 256K admission. HIP-only bulk seeding, final-token handoff, empty-checkpoint cancellation and recovery were observed; a short mixed bulk request also passed. This does not prove physical state parity, transient peak memory, or injected failure retirement. No HTTP SSE smoke was run.

Fast-forwarded `feature/heterogeneous-hip-only` to `d7b6fbfba1b81f90cf304a6fb0f7dc89810ceb72`. Fixed a stale resident test that still expected all native prefill with MTP to be rejected: it now checks that bounded prefill is admitted, while lookup and an eight-token bulk prompt are rejected. No production code or legacy gate was changed. Unrelated untracked `tools/heterogeneous_pool_probe.cpp` was preserved and not committed.

## Builds and tests

Existing checkout `.venv` and Release build caches, vanilla toolchain, 28 build jobs:

```sh
cmake --build build-256k-host --parallel 28 --target strata_heterogeneous_live_smoke strata_heterogeneous_final_pipeline_tests strata_heterogeneous_host_tests strata_heterogeneous_window_admission_tests strata_heterogeneous_hip_final_verifier_contract_tests strata_heterogeneous_hip_verifier_contract_tests strata_heterogeneous_mtp_prefix_cursor_tests strata_heterogeneous_mtp_request_policy_tests strata_heterogeneous_mtp_stream_batch_tests strata_heterogeneous_mtp_bulk_prompt_tests
cmake --build build-256k-cuda --parallel 28 --target strata_heterogeneous_cuda_worker strata_heterogeneous_live_smoke
cmake --build build-256k-hip --parallel 28 --target strata_heterogeneous_hip_worker strata_heterogeneous_live_smoke
ctest --test-dir build-256k-host --output-on-failure -R 'strata_heterogeneous_(final_pipeline|host|window_admission|hip_final_verifier_contract|hip_verifier_contract|mtp_prefix_cursor|mtp_request_policy|mtp_stream_batch|mtp_bulk_prompt)_tests'
env PYTHONPATH=tools .venv/bin/python -m unittest serve.test_heterogeneous_resident tools.test_heterogeneous_hip_only_cli tools.test_heterogeneous_mtp_stream_contract tools.test_heterogeneous_mtp_qualification tools.test_heterogeneous_mtp_generation
```

All builds passed; existing CUDA/HIP compiler warnings were emitted. CTest **9/9**, Python **45/45** after the test correction (initial Python run had that one stale expectation). The new C++ test checks 701 policy/pairing assertions, including nonidentical prompt successors across chunk boundaries, extent and identity failures, poison behavior and final verifier handoff. Python contract tests cover malformed progress/empty first checkpoint and limits. Wrong-device rejection is present in the native callback but was not fault-injected on GPU; one-token bypass is covered by policy, not a GPU run.

Source review: `native_prefill_domain.hpp` installs the consumer only on `owned.nodes.back()`; native prefill fences its compute stream before `on_chunk` in `src/prefill/prefill.cpp`. The consumer checks the current device against the draft owner and synchronous prime consumes the borrowed row before return. Native owners are drained before the prefill function returns; `finish_bulk_prompt` requires the cursor at the prefix end, draft idle and reserve check before adoption. Worker gate completes prefill only after its callback succeeds. The observed two-chunk grouped execution exercises this path, but is not a direct injected asynchronous failure test. Failed execution is rejected rather than ACKed; GPU failure retirement was not independently injected.

## GPU smoke

Fresh process/listener and NVIDIA/AMD compute checks found no active Strata service or GPU compute processes; AMD 0/1 each had 57 MB used, NVIDIA had 15 MiB graphics use. Snapshot is not a reservation. Reused report 0049's exact 2048-cell Q2_0 configs, full resident expert maps, INT8 KV, >=1 GiB reserve, pinned Q8 draft `/home/chris/workspace/Strata-data/mtp/rt`, spec=4, min-p=0.5, overlap settings, `HSA_ENABLE_SDMA=0`, with `STRATA_*` overrides cleared. No existing service was contacted. Private driver and raw logs/records: `/home/chris/tmp/strata-heterogeneous/handoff-0050/`.

```sh
env PYTHONPATH="$PWD" .venv/bin/python /home/chris/tmp/strata-heterogeneous/handoff-0050/bulk_check.py hip bulk
env PYTHONPATH="$PWD" .venv/bin/python /home/chris/tmp/strata-heterogeneous/handoff-0050/bulk_check.py hip serial
env PYTHONPATH="$PWD" .venv/bin/python /home/chris/tmp/strata-heterogeneous/handoff-0050/bulk_check.py mixed bulk
```

Prompt `[23066,23066,23066,23066,23066,23065,23066]`, budget 16, bulk chunk 3. HIP-only 24/24 executed A/cancel/A in one resident process. Each request logged `PRIMED cell=0 rows=3`, `cell=3 rows=3`, then native prefill committed six rows. Request 1 emitted 16 IDs, all 23066, proposed/accepted 12/12. Request 2 cancelled on `HETERO_PROGRESS 2 1 6 BATCH 0 0` with empty cancelled final record; request 3 reset, repeated both seed chunks and returned the identical 16 IDs with 12/12 proposals accepted. First normal checkpoint after bulk was position 7 with the final prompt token verified and one output. Streamed batches matched the final records. Matching HIP serial MTP reference used the same prompt, settings and overlap, produced the same IDs and 12/12. Mixed 15/18/15 bulk also seeded 0+3 and 3+3 and produced the same IDs, 12/12. Equal sampled outputs do not prove state parity.

An initial distinct prompt `[248045,846,198,2523,1234,5678,42]` stopped immediately (0 proposals); it also demonstrated empty-checkpoint cancel/reset and matching recovery, retained in `hip-bulk-stop.*`. All four GPU process sets closed normally with `HETERO_CLOSED`; no GPU compute processes remained. Wall times were 16.35 s initial stop, 16.61 s HIP A/cancel/A, 15.44 s HIP serial, 13.69 s mixed: **62.09 s total**. HIP logs show KV resident on both owners, full expert routes and zero host routes/misses; HIP1 MTP admission had 12,029,263,872 bytes free, 3,044,868,096 estimated and 1,073,741,824 reserved. During verifier capture HIP free snapshots were 11,492 and 9,912 MiB; mixed CUDA minimum logged 2,717 MiB. These snapshots are not a measured transient peak. Normal close and no remaining workers do not validate abnormal retirement.

No long context, 256K MTP, performance campaign, dependency/asset/driver changes, reserve reduction, forced kills or GitHub Actions. Remaining work includes broader prompt diversity, GPU wrong-owner/failure injection, transient peak sampling, and optional private serial-MTP HTTP SSE integration.
