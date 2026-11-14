# M7 — Vanilla-first backend-domain integration

**Production target:** CUDA0 [0,12) in one CUDA process, HIP0 [12,30) → HIP1 [30,48) in one native HIP process. Only CUDA/HIP communication is new. The full three-device physical plan/fingerprint, ownership and resource limits remain unchanged. The per-GPU process route is a historical regression control, not the target.

| ID | Work | State |
| --- | --- | --- |
| M7.0 | Vanilla-first contract | Implemented |
| M7.1 | Same-runtime native multi-GPU | Grouped launch, mapped local handoff and native verifier chain wired; CPU tests authored, native qualification pending |
| M7.2 | Native bulk-prefill | Worker-native Prefill, source/cache/session binding, R-only framing, drain and position adoption wired; qualification pending |
| M7.3 | Native asynchronous prompt pipeline | Existing chunk-major/local async successor pipeline reused; initial bounded socket bridge with per-chunk staging fence. Native overlap/rates unverified |
| M7.4 | Native MTP | Pending approved model-compatible MTP runtime assets and native draft lifecycle |
| M7.5 | Native speculative overlap | Vanilla window-policy extraction reused in generate.cpp; native parity/state/scheduler and rollback integration pending |
| M7.6 | Cross-runtime synchronization | Initial bounded bridge. Qualify first, then optimize only bridge; no same-backend IPC |
| M7.7 | Real agentic throughput/quality | Pending; median >=160 accepted output tok/s and 95–100% productive utilization/sustained useful power within existing limits |

## Commands and implementation seam

- Keep the original physical plan and three stage configs. Native live smoke has explicit `--group-native-hip` qualification mode, creating **two** children and endpoints 0 and 2. HIP validates both ordered configs before any load, constructs and links the real stage verifiers using vanilla `set_next` and mapped native pinned handoff. Both HIP stages are reset/committed by the one domain; one ACK covers both. The initial decode control is serial accepted-prefix execution, **not** MTP overlap.
- Add `--native-prefill-chunk N` (1..2048) to the grouped **resident generation** invocation. This is independent of decode max_t. Bulk-prefill tokens [0,n-1) with native chunk-major Prefill on each backend, using existing full-resident expert source/cache/session and local streams; process the final token through the ordinary verifier/head.
- Bridge residual packet kind carries contiguous float32 R only, rows * hc * n_embd. Never confuse with verifier R | bo | inject. Frames <=16 MiB, split inside chunks as needed; position and exact length checked at both boundaries, finite gate, both domains must drain/ACK. Pinned upload source stays valid until local H2D completion. Even with shared decode transport, prefill currently uses bounded socket framing. Native HIP Prefill links its two stages directly via its own double-buffered successor path. No foreign device pointers/events.
- Owned Prefill memory allocated per request, retaining original reserve and residency; explicit nonzero arena budget currently declines rather than overspend. No automatic expert-slot loans or reserve changes. Select fitting chunks under original settings, no silent fallback. Stage cleanup and independent same-process allocation ownership require real GPU qualification. Native layer-major/multi-prompt mixed paths disabled; no output before completed prefill. Cancellation at drained prefix boundary, not mid-GPU operation. Grouped mode currently rejects diagnostic probe flags; add a correct real CUDA/HIP boundary capture if needed for validation. 
- Existing vanilla speculative window state/lookahead/acceptance policy factored into `include/strata/core/pipeline_window.hpp` and used by vanilla loop. Do not clone its scheduler. Use native MTP, NativeHead, draft prompt K/V, verifier capture/launch/service/finish/commit, native parity snapshots/rollback; expose runtime-local operations across only the vendor boundary. No partial speculative commit ownership or fake MTP assets.

## Local evidence

Direct checkout from this environment blocked by GitHub DNS; repository connector source edits and commits succeeded. Re-ran local policy 1811 assertions (exact repository header/test blob hashes), isolated actual external Prefill entry 12 checks (prior seam before added exception/drain handling), native init helper 9 fake-runtime checks (exact helper/resources blobs), prior domain and chain helper fixtures 11 + 20 checks. These do not qualify the final integrated code. Added actual host grouped process/handshake/failure/reset/commit, fake native HIP chain and native prompt protocol tests to existing targets; **not run** here. Neither full native build nor hardware runs executed. No Actions. Handoff 0011 is the new qualification/implementation batch.

## Remaining

Qualify owner-device setup and all-owner cleanup/refill safety, compile separate workers and affected CPU tests. Validate grouped serial control only as a basis of the new topology, then real bulk-prefill continuity, full native prompt drain and agentic quality. Verify MTP assets in agreed inputs, implement native scheduler integration where possible, validate rollback before enabling speculative overlap. Adapt benchmark/trace tooling to grouped endpoints and distinct native prefill markers; do not infer three independent worker spans from two processes. Compare with existing historical performance and matched authorized vanilla where possible. Keep all existing correctness gates and hardware settings.
