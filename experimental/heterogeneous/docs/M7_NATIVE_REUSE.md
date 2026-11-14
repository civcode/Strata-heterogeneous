# M7 — Vanilla-first integration

Goal: extend proven vanilla Strata to mixed CUDA/HIP, not build another inference engine. Keep original 12/18/18 residency, existing reserves, model/math and accepted-prefix safety. Performance gate: median >=160 accepted tok/s on representative real agentic coding; high productive GPU use and sustained useful power.

| ID | Work | State in this patch |
| --- | --- | --- |
| M7.0 | Vanilla-first contract | Strengthened |
| M7.1 | Same-runtime multi-GPU stage chains | Native Prefill chain preserved; worker grouping/owner hardening pending |
| M7.2 | Native bulk-prefill | External input and output seam; native resource view and owned-buffer init helper implemented, not wired into protocol |
| M7.3 | Native asynchronous prompt pipeline | `run_external` uses the existing chunk loop and drains native successors only at prompt end; cross-runtime transport/worker integration pending |
| M7.4 | Native MTP | Need approved matching runtime assets, NativeHead and native draft ownership/prompt-KV lifecycle |
| M7.5 | Native speculative overlap | Extracted original vanilla window state, lookahead construction and acceptance decisions into shared core header used by vanilla; native scheduler/rollback integration pending |
| M7.6 | Cross-runtime synchronization | Work at the CUDA/HIP bridge only, after correct native paths |
| M7.7 | Actual agentic performance/quality | Oberon qualification |

## Integration seam

- Native `Prefill` hand-off is contiguous float32 `R` only: `rows * hc * n_embd` floats. The decode verifier's `R | bo | inject` format is different. Never reinterpret one as the other.
- CUDA front: `set_stage(0, 12, nullptr)`, `set_external_handoff({}, true)`, use the existing synchronized `on_chunk` callback to transmit `R` with the exact request/chunk/position/geometry. HIP front: `set_stage(12, 30, &hip_tail)`, external input callback receives the packet and enqueues H2D to its local `R`/stream. HIP tail: `set_stage(30, -1, nullptr)`. Native HIP-to-HIP double-buffered successor execution is unchanged. Choose one common native chunk plan for both runtime groups via `Prefill::pipeline_chunk` where appropriate; do not use verifier `max_t` as the prompt cap. `run_external` takes that chunk size and uses native `run_impl` and `drain_pipeline`.
- Both providers expose the real, verified full-resident `FileExpertSource`, `ExpertCache`, host residency table and owning device/layer range. `init_owned_native_prefill` calls vanilla `bytes_needed_owned` and `init`, without an alternate allocator or expert-slot lending. If it cannot fit, decline; do not reduce residency or reserves automatically. User-provided arena/budget policy must be enforced by the worker. Caller must ensure prefill is idle and destroyed before its providers/sessions/streams; native resource ownership and cleanup must be qualified together.
- Initial bridge is explicitly chunk-major, not layer-major/multi-prompt. External callbacks are local, bounded, fail-closed, and must not throw. Keep pinned transfer source alive until its local upload completes; output callback must establish safe lifetime before native R is reused. No device pointers or events cross process boundaries. All chunks must drain before reset or decode; failure poisons request. Existing vanilla run and run_multi reject external mode instead of silently doing the wrong arithmetic.
- Use vanilla prompt-to-decode convention: bulk-prefill through the penultimate prompt token, then ordinary native one-token verifier for the final prompt token to obtain first prediction. The verifier/admission position bookkeeping must move together with successful globally completed prefill. Do not fake prefill as a series of decode commits. Later MTP uses vanilla draft prompt K/V via `Prefill::draft_kv`/drafter fallback, final residual and accepted-prefix path.

## Remaining integration and validation

First materialize a small native group fixture (existing owners on two HIP GPUs, with qualified same-process allocation-device and all-owner pre-release/refill safety) and a minimal cross-runtime prompt command using distinct typed residual framing; use native chunk pipeline with bounded buffering/backpressure at the boundary. Wire successful prefill position handoff and normal final-token verifier, cancellation, poisoning and ordinary cleanup into the real resident path. Keep old serial path as control. Do not claim this patch already implements a runnable mixed bulk-prefill or speculative engine.

Then integrate the **existing** MTP/overlapping-window scheduler, not a lookalike: isolate the native scheduler's runtime-local actions using `Verifier::capture_all`, `pl_launch`, `service`, `done`, `pl_finish`, `pl_commit_async`, native stage-owned parity state, snapshots, rollback and native MTP first-stage placement. Validate correct rollback and no uncommitted output. Native MTP asset availability is not yet established. No speculative commit safety bypass.

Build separate CUDA/HIP workers and native host tests; compare vanilla homogeneous vs heterogeneous with same weights/mode/chunking. First demonstrate actual bulk-prefill state continuity and decode correctness, then measured phase-aligned utilization/rates and representative coding acceptance. No Actions, no hardware/config/precision changes.
