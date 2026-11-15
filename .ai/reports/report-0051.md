# Report 0051 — expanded 2K native MTP bulk

## Result

**PASS for focused builds/tests and bounded 2K HIP-only, mixed and private HTTP SSE smoke.** Source `94a0a5736d08b42889aeeadb7adf6885c8109ae2`, fast-forwarded on `feature/heterogeneous-hip-only`. No integration source corrections were needed. This is not production readiness, 256K MTP admission, a peak-memory proof, or a speedup claim. Preserved unrelated untracked `tools/heterogeneous_pool_probe.cpp`.

## Build and verification

Read `.ai/TEST_BUDGET.md`; used existing Release caches, vanilla toolchain and checkout `.venv`, 28 parallel build jobs:

```sh
cmake --build build-256k-host --parallel 28 --target strata_heterogeneous_live_smoke strata_heterogeneous_final_pipeline_tests strata_heterogeneous_host_tests strata_heterogeneous_window_admission_tests strata_heterogeneous_hip_final_verifier_contract_tests strata_heterogeneous_hip_verifier_contract_tests strata_heterogeneous_mtp_prefix_cursor_tests strata_heterogeneous_mtp_request_policy_tests strata_heterogeneous_mtp_stream_batch_tests strata_heterogeneous_mtp_bulk_prompt_tests
cmake --build build-256k-cuda --parallel 28 --target strata_heterogeneous_cuda_worker strata_heterogeneous_live_smoke
cmake --build build-256k-hip --parallel 28 --target strata_heterogeneous_hip_worker strata_heterogeneous_live_smoke
ctest --test-dir build-256k-host --output-on-failure -R 'strata_heterogeneous_(final_pipeline|host|window_admission|hip_final_verifier_contract|hip_verifier_contract|mtp_prefix_cursor|mtp_request_policy|mtp_stream_batch|mtp_bulk_prompt)_tests'
env PYTHONPATH=tools .venv/bin/python -m unittest serve.test_heterogeneous_resident tools.test_heterogeneous_hip_only_cli tools.test_heterogeneous_mtp_stream_contract tools.test_heterogeneous_mtp_qualification tools.test_heterogeneous_mtp_generation
```

All builds passed with existing compiler warnings; **9/9 CTest, 45/45 Python**, bulk test **54,011 assertions**. CPU cases include near-capacity bounds, actual successor pairing, partial final chunks and later subbatch failure poisoning; Python checks exact 2K bounds and stale capability rejection. No near-capacity GPU test.

Source review: `resident_context_bytes` uses the same `qsa_state_bytes(g,cells,false,0)` fully resident shared-RoPE sizing as native `load`, rounds capacity to 64 and accounts for the same `max_t*cap` identity and native QSA attention scratch with 256-byte bump alignment. The service-only check adds this to the original fixed 1 GiB cushion and rejects unsupported Q8 modes; diagnostic lanes retain their previous estimate. This is a conservative admission estimate, not measurement of all transient allocations. The cursor advances only after all subbatches succeed; a later failure poisons it. Target prefill owners drain before completion, and the terminal adoption requires the drafter idle and cursor complete; coordinator waits for all domain ACKs before proceeding. Actual fatal GPU failure/retirement was not injected.

## Hardware

Fresh process/listener, NVIDIA and AMD compute checks showed no active Strata service or compute process; AMD owners had 57 MB used each. Snapshot is not an exclusive reservation. Reused report 0050's exact Q2_0 2048-cell full-resident configs and Q8 `/home/chris/workspace/Strata-data/mtp/rt`, INT8 KV, >=1 GiB per-owner reserve, spec 4/min-p 0.5, grouped HIP, cleared `STRATA_*`, `HSA_ENABLE_SDMA=0`. Evidence and private drivers: `/home/chris/tmp/strata-heterogeneous/handoff-0051/`.

```sh
env PYTHONPATH="$PWD" .venv/bin/python /home/chris/tmp/strata-heterogeneous/handoff-0051/bulk_check.py hip bulk
env PYTHONPATH="$PWD" .venv/bin/python /home/chris/tmp/strata-heterogeneous/handoff-0051/bulk_check.py mixed bulk
env PYTHONPATH="$PWD" .venv/bin/python /home/chris/tmp/strata-heterogeneous/handoff-0051/http_check.py
```

HIP 24/24 A/cancel/A and mixed 15/18/15 used a nonuniform 77-token prompt (`23065` where zero-based index modulo 11 is 5, otherwise `23066`), chunk 32, budget 16. Each 76-row prefix was processed as target chunks 32, 32, 12, producing ordered synchronous prime logs cells 0,8,16,24,32,40,48,56,64 each 8 rows and cell 72 four rows. The first empty checkpoint was position 76; request 2 cancelled there with an empty final record. Request 3 reset and reproduced request 1's 16 IDs, with 12 proposed/12 accepted each. The first post-seed single-token verifier checkpoint advanced to position 77. Streamed batches matched final records. Mixed produced the same IDs, 12/12, and clean completion. No separate serial comparison was necessary for this bounded contract smoke; equal outputs do not establish physical-state parity.

HIP1 admission logged `context_bytes=3288848`, free 12,029,263,872, estimate 3,048,156,944, reserve 1,073,741,824 bytes. First HIP prefill stage 0 free before/after allocation was 12,199,133,184/11,924,406,272 bytes; terminal stage 1 was 10,554,966,016/10,280,239,104 with reserve 1,610,612,736 (base plus 512 MiB prime headroom). At drain terminal free was 10,156,507,136 bytes. Mixed terminal free before/after allocation was 17,934,843,904/17,660,116,992 with the same augmented reserve. All reported fits; these are snapshots, not transient peaks. Resident INT8 KV and full expert maps were logged; host routes and misses were zero on every owner.

Private HTTP used a newly launched `python -m serve.server --engine heterogeneous` bound to OS-selected loopback port, not an existing service. A short chat prompt, max_tokens 16 and native chunk 32 yielded HTTP 200 `text/event-stream`, four JSON SSE events, `Hello!`, finish `stop` and `[DONE]`; 19 actual prompt tokens, bulk prefix 18, prime rows 8/8/2, proposed/accepted 1/1. SIGTERM requested normal shutdown; server exited 0 and logged `HETERO_CLOSED`. Two earlier CPU-only wrapper startup attempts failed configuration validation before GPU admission; direct heterogeneous server was used instead. All three GPU process sets closed normally with `HETERO_CLOSED`; final process/compute checks found no workers. Wall times 16.93 s HIP, 13.84 s mixed, 19.31 s HTTP: **50.08 s total**.

No long-context GPU tests, injected fatal faults, forced kills, installs, assets/repack, clocks/drivers, reserve reductions, spill, launcher changes or GitHub Actions. Outstanding: broader prompts, measured transient peak headroom, GPU failure-path retirement and independent state parity.
