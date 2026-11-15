# Report 0044 — Oberon 256K resident lifecycle

## Result

**PASS for the scoped synthetic Q8 full-resident capacity and resident-adapter lifecycle qualification on both topologies.** Each 256K session ran seven independent requests on one worker set, including two near-capacity completions and two cooperative cancellations. This does not establish HTTP server production readiness, general fault recovery, numerical parity, arbitrary-prompt quality, or a transient peak-memory bound. Historical numerical discrepancies remain unchanged. No implementation defect was observed and no inference code was modified.

## Provenance and checks

- Branch `feature/heterogeneous-256k`, starting revision `d49283c00a2e6b9790981060d2cfa41562fdf344` on Oberon. Native source and CMake inputs are unchanged from capacity revision `2b1016bf7d0a7d781966ba28a6f6708cb8825312`; the same separate binaries built with 28 jobs were reused. Their SHA-256 values match [the capacity report](../../experimental/heterogeneous/docs/LONG_CONTEXT_256K_OBERON.md). HIP gfx1201, ROCm `/opt/rocm/core-10.1`, HIP prefill MMQ on; CUDA sm_89; `HSA_ENABLE_SDMA=0`. No GitHub Actions.
- Checkout-local `.venv/bin/python` 3.12.3; `pip check` passed. Targeted lifecycle, resident adapter and config Python tests: 21 passed. Host CTest long-context, resident, worker bootstrap/config and window admission selection: 8/8 passed. Earlier full host CTest 102/102 and Python discovery 378/378 are recorded in the capacity report.
- Private server configurations used the previously qualified exact 262144-cell Q2_0 configs, max_t=8, 512 experts per owned layer, uncapped owned arena, >=1 GiB reserve, INT8 KV, shared transport, grouped HIP, streamed tokens, native prefill chunk 128, four output tokens and no restart-clean. No fallback or spill was requested. CPU preflight passed for each. HIP plan `hip:0:0-24,hip:1:24-48`; mixed `cuda:0:0-15,hip:0:15-33,hip:1:33-48`. Runtime devices were CUDA PCI 04:00.0 and R9700 PCI 03:00.0/07:00.0. Each owner reported full expert residency and zero host expert misses.
- The exact prior 262136-token repeated ` hello` prompt was reused (file SHA-256 `5b8474f3f3cee86001a81a6c366df4119af3475a7ed4ef92ba276fb2f51d10ea`; token-ID digest `e5965724938ea58b6ad45c3e7e3399cb547a45eb34658e2a95f3db8a0644a439`). Private server config hashes: HIP `70c96e56db5c35be1a5ab3761a5396c4e74066ebdba739fed4b6314d83fa3907`, mixed `ed360799dbee90d65aec1b34661a95ae1e95288f7fddbfbf004d6d158c4d5ef1`.

## Lifecycle evidence

| Request | Action | Prompt tokens | Finish | Committed next position | Streamed outputs |
|---:|---|---:|---|---:|---:|
| 1 | short baseline | 7 | length | 10 | 4 |
| 2 | long A | 262136 | length | 262139 | 4 |
| 3 | cancel at long prefill checkpoint | 262136 | cancelled | 262135 | 0 |
| 4 | short after cancel | 7 | length | 10 | 4 |
| 5 | cancel after first long prediction | 262136 | cancelled | 262136 | 1 |
| 6 | long B | 262136 | length | 262139 | 4 |
| 7 | short reset control | 7 | length | 10 | 4 |

The table applies independently to both topologies. In each session the runner verified request IDs, cancellation ACKs, returned/streamed outputs, metrics, identical A/B output hashes and identical three short-control hashes. The last output token is not fed back. This determinism is same-session reset evidence, not a cross-backend numerical parity claim.

`STRATA_HETERO_WINDOW_TRACE=1` was examined per physical protocol endpoint: HIP grouped endpoint stage 1; mixed CUDA endpoint stage 0 and grouped HIP endpoint stage 2. Each endpoint emitted seven `request_ready` records at position zero. On every non-prefill-cancelled request, the first admitted and provisional verifier window started at the prefill-committed prefix (6 or 262135); subsequent admitted/provisional/committed triplets advanced contiguously one position. Long A/B committed 262135 through 262138, yielding next=262139. The prefill-cancel request had no verifier window; prediction-cancel committed only 262135 to next=262136. No subsequent verifier commit occurred for either cancelled request. Grouped HIP stage 1 in the mixed plan and stage 0 in the HIP-only plan are internal owners of their grouped endpoint, not separate protocol endpoints; per-owner capacity records and full-residency statistics cover both, but there is no independent protocol window trace for those internal owners.

The runner reported normal `HETERO_CLOSED` and verified the old coordinator process group no longer existed for both sessions; no surviving native workers were observed afterwards. No GPU process was forcibly killed.

## Admission, memory and negative checks

All logged `HETERO_CAPACITY` checks reported fit and free >= reserve + required extra. Minimum observed native free MiB in plan order: HIP **9304, 9144**; mixed **1437, 14224, 16772**. Mixed CUDA's minimum is 413 MiB above the 1024 MiB reserve; this is narrow headroom and not a guarantee against transient peaks or other workloads. Independently sampled every two seconds: HIP AMD peak used **23396, 23558 MiB** (299 samples/device); mixed NVIDIA/AMD/AMD peak used **14507, 18484, 15931 MiB** (226 samples/device). Runtime and monitor accounting differ; neither sampling method establishes exhaustive transient peak. No reserve was lowered.

CPU preflight at this revision rejected 262136 prompt tokens plus nine requested output tokens at 262144, and rejected a 131072 request against the 262144 worker configs (exit 2). Bootstrap and worker-admission CTests passed, including current parser acceptance of the explicit long-context chunk and rejection of invalid options. Two separately built older local workers, identified by binary SHA-256 CUDA `6915d0c305db2b530074d784e3f5d453a008f883cc6f9587703944e4e1e4bb3c` and HIP `2ce8cac09b028133d739e1d088ca61a6cf4878dab5199f26f86f0672fc9ca13a`, lacked the capability option string and each rejected the actual long-context invocation at argument parsing (exit 2), before config loading or GPU work. GPU visibility was disabled for those negative invocations. Their precise source revisions were not independently established; this demonstrates rejection by these two binaries, not by every possible stale worker.

Private artifacts, including config manifests, preflights, JSON results, endpoint traces, admission records and independent memory samples: `/home/chris/tmp/strata-heterogeneous/handoff-0044/`; prior tier evidence remains under `/home/chris/tmp/strata-heterogeneous/qualification-256k/`. No HTTP loopback service test or injected GPU fault was performed in this batch. The user-owned untracked `tools/heterogeneous_pool_probe.cpp` was preserved.
