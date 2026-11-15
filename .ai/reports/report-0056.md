# Report 0056 — immutable MTP admission and launcher regression

## Status

**PASS for integrated builds, focused CPU policy/pairing, both actual launcher preflights and a short mixed launcher SSE A/B regression.** The requested >32-token GPU prompt was **not exercised**: actual chat-template prompts were only 19 and 17 tokens, so this run did not cross a native chunk boundary. No second GPU process was launched to repeat the test. No production, high-position GPU correctness, GPU fault recovery or speedup claim. Tested source `122d1fbf5830ae6b64427ef7cac3c34e70d49e2e`; no source corrections required. Unrelated untracked `tools/heterogeneous_pool_probe.cpp` preserved.

## Builds and CPU validation

Read `.ai/TEST_BUDGET.md`; existing vanilla Release caches/toolchain and checkout `.venv`, 28 parallel jobs. Rebuilt affected host coordinator and tests, CUDA/HIP coordinators and actual workers:

```sh
cmake --build build-256k-host --parallel 28 --target strata_heterogeneous_live_smoke strata_heterogeneous_final_pipeline_tests strata_heterogeneous_host_tests strata_heterogeneous_window_admission_tests strata_heterogeneous_hip_final_verifier_contract_tests strata_heterogeneous_hip_verifier_contract_tests strata_heterogeneous_mtp_prefix_cursor_tests strata_heterogeneous_mtp_request_policy_tests strata_heterogeneous_mtp_stream_batch_tests strata_heterogeneous_mtp_bulk_prompt_tests strata_heterogeneous_long_context_tests strata_heterogeneous_native_prefill_stage_contract_tests strata_heterogeneous_hip_worker_bootstrap_tests strata_heterogeneous_cuda_worker_bootstrap_tests
cmake --build build-256k-cuda --parallel 28 --target strata_heterogeneous_cuda_worker strata_heterogeneous_live_smoke
cmake --build build-256k-hip --parallel 28 --target strata_heterogeneous_hip_worker strata_heterogeneous_live_smoke
ctest --test-dir build-256k-host --output-on-failure -R 'strata_heterogeneous_(final_pipeline|host|window_admission|hip_final_verifier_contract|hip_verifier_contract|mtp_prefix_cursor|mtp_request_policy|mtp_stream_batch|mtp_bulk_prompt|long_context|native_prefill_stage_contract|hip_worker_bootstrap|cuda_worker_bootstrap)_tests'
env PYTHONPATH=tools .venv/bin/python -m unittest serve.test_heterogeneous_resident serve.test_heterogeneous_retirement tools.test_heterogeneous_mtp_stream_contract tools.test_heterogeneous_hip_only_cli tools.test_heterogeneous_oberon_launch serve.test_heterogeneous_http_disconnect
```

All builds passed; **13/13 CTest, 53/53 Python**. Policy test **86,271 assertions**, bulk/pairing **1,040,230 assertions**, including immutable owned policy, reset, actual successor pairing, partial acceptance and stale suffix ledger and anchor-only physical tail at synthetic numeric 32K/256K. Those CPU positions do not exercise GPU attention. Source review confirmed coordinator admits the owned snapshot after metadata identity ACK and terminal binds independently; per-window operations use the immutable policy while retaining position/stop/budget checks.

Rebuilt executable SHA-256: host `29fb9a079a0d3b2a567263f5f73ec8c9a51d8ba0c88a887045c2eb0fc54fc902`; CUDA worker `108e3b8c21f8d8c2e986ab4d9677455d179f8bfa1c4d3fd6113c9aadfd1c2735`; HIP worker `49e587e8c97102d269282f0e823dbca6c98b811eb1e8ac5392c814567994c8f9`.

## Actual launcher preflight

`./run-q2_0-heterogeneous.sh --prepare-only` and `./run-q2_0-hip-only.sh --prepare-only` both passed CPU-only preflight using the recorded 0044 references. The mixed generated private config `/home/chris/tmp/strata-heterogeneous/launchers/server-jr68lfe0/server.json` bound `cuda:0:0-15,hip:0:15-33,hip:1:33-48`, fingerprint 2026111301, cells 262144, `max_t=8`, INT8, MTP `/home/chris/workspace/Strata-data/mtp/rt`, spec 4/min-p 0.5, native chunk 32, max-new 256 and the three rebuilt worker/coordinator paths above. Stage configs bind `/home/chris/workspace/Strata-data/packs/q2_0`, index `c344c01f59f547ed8a59003857a82eb90499e5299154241dbca787acb54b5b7c`, full expert slots and 1 GiB reserves. HIP-only ordinary launcher also passed without changing its profile or reference. Neither preflight started workers.

## Real launcher hardware smoke

Fresh process/listener and NVIDIA/AMD compute snapshots showed no Strata service or GPU compute process; AMD owners each used 57 MB. Snapshot is not exclusive ownership. The private driver launched the **actual** `./run-q2_0-heterogeneous.sh --port <free loopback port>` after clearing `STRATA_*`; the launcher set `HSA_ENABLE_SDMA=0` and used the exact 0044-bound configs. No existing service was contacted. Command and logs:

```sh
.venv/bin/python /home/chris/tmp/strata-heterogeneous/handoff-0056/launcher_smoke.py
```

One resident process served two short SSE requests, max_tokens 16: both HTTP 200, `stop` and `[DONE]`. Request 1 had 19 prompt tokens, bulk prefix 18, proposed/accepted 1/1, committed position 21; request 2 reset, had 17 prompt tokens, prefix 16, proposed/accepted 9/6, committed position **26**. They both fit within one chunk, so multi-chunk native execution remains a specific omission. CUDA/HIP0/HIP1 reported full experts 7680/7680, 9216/9216, 7680/7680, resident INT8 KV, zero host routes/misses. Draft admission HIP1 free 18,106,810,368, estimate 3,465,350,928, context bytes 420,482,832 and reserve 1,073,741,824. Request 2 CUDA prefill drained free 1,544,552,448 against 1,073,741,824 reserve. These snapshots do not prove transient peak headroom.

The private server received cooperative SIGTERM, returned 0 and logged `HETERO_CLOSED`; server and coordinator process groups were independently checked absent, and final GPU compute inventory was empty. GPU-process wall **17.73 s**. Private logs and responses: `/home/chris/tmp/strata-heterogeneous/handoff-0056/`. No long/high-position GPU requests, forced kills, injected GPU errors, asset/driver/clock changes, spill, reserve reduction, active config edits or GitHub Actions.
