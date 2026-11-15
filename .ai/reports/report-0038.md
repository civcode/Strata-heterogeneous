# 0038 — serial verified direct edge on real model

**Status: PARTIAL.** The opt-in serial verified shared CUDA→HIP edge ran actual 15/18/15 inference on Oberon. Its fixed-window committed full-logit packets matched the existing shared reference byte-for-byte for T=1, T=4 with keep=2 and continuation, and T=8 with keep=4 and continuation. A small resident reset/cancel/continue case matched generated IDs and completed cleanly. This is bounded evidence, **not full state parity, performance qualification, fully asynchronous CUDA→HIP prelaunch, or production admission**. Direct/reference inference benchmarking was not run because full native-state parity and broader lifecycle negatives remain outstanding.

## Implementation and safety boundary

Starting commit `2d9641e40448db6de74d360a6b00c6eede1d5ad2` includes the experimental `--direct-verified-shared` implementation. It independently registers the same size-sealed shared slot with each runtime; each process resolves its own alias. CUDA uses the unmodified native publisher; after producer drain, host checksum and per-element finite scans and full frame/identity checks remain before HIP graph execution. HIP reads directly using native mapped reads and ready wait, then the existing native HIP0→HIP1 chain. This eliminates the separate activation staging copy at this boundary, **not** the CPU admission scans or the serial producer/consumer ordering. The ready flag is not a finite verdict. Existing accepted-prefix ACKs and poisoned-session behavior remain; no default or native kernel changed.

Added CPU-double HIP adapter coverage for borrowed direct input, missing ready, wrong window, foreign slot, no upload, partial commit and resource cleanup. Added a protocol CPU-double nonfinite producer-output test proving rejection, session poisoning and no commit. Prior tests cover direct binding and capability mismatch, finite input, reset and reuse. These are host fixtures, not GPU fault injection. Hardware did not inject GPU faults. Initial test compile failed due to a missing `<cstring>` include; corrected and rebuilt.

## Build and host tests

Regenerated existing host, CUDA and HIP caches and built with `-j28`, existing Strata toolchain and `.venv`: host complete, native CUDA and HIP worker targets both succeeded. CUDA nvcc 12.0.140/sm_89, HIP 7.16.26385/gfx1201, CMake 3.28.3. Native kernel source diff relative to report 0037 is empty. Worker SHA-256: CUDA `a8d96b78f042a87175a6c1f75bb1759524f0046ec626493592ee5a814d6eea57`; HIP `7622e43fde8266aaf961a1bce1cf72cce19c85a5fc7bfec7f85555ec5825c902`.

Host CTest `-j28` passed 101/101 after the HIP fixture; final targeted protocol/HIP adapter tests passed 2/2 after the additional producer negative. Python direct-graph, HIP CLI/pair and model-gate modules passed 33/33 with `PYTHONPATH=tools`. An initial Python invocation without that path failed to import the HIP pair module; corrected and rerun. CPU CLI preflight accepted the direct candidate and rejected missing shared mode and incompatible receiver-arm. A bulk-mode refusal also occurred, but the parser rejected missing generation options first, so that run does not specifically prove the direct/bulk guard.

## Actual model runs

CPU-only preflight passed for all three existing 2K configs, max_t 8, original pack index SHA-256 `c344c01f59f547ed8a59003857a82eb90499e5299154241dbca787acb54b5b7c`, fingerprint `2026111301`, FP16 default KV and 1 GiB reserve per owner. No fresh full asset checksum was performed. Worker startup confirmed CUDA ordinal 0 RTX 4070 Ti SUPER `0000:04:00.0`, HIP ordinal 0 and 1 Radeon AI PRO R9700 `0000:03:00.0` and `0000:07:00.0`. All owners reported full residency and zero host expert misses. Reference and direct were separate fresh launches, same configs, shared/grouped HIP overlap/full-logit mode, no MTP/lookup/bulk/receiver-arm. Normal process exit and route-stat shutdown records were observed; native graph destruction and unregister ordering is enforced by the implementation but not independently traced.

| Rows / keep | Windows / positions | Committed predictions per window | Result |
|---|---|---|---|
| 1 / 1 | 1 / 0 | `[846]` | exact STFR packet match |
| 4 / 2 | 3 / 0, 2, 4 | `[846,198]`, `[96220,513]`, `[449,264]` | each committed full-logit STFR byte-identical |
| 8 / 4 | 2 / 0, 4 | `[846,198,96220,513]`, `[449,264,2724,248046]` | each committed full-logit STFR byte-identical |

These comparisons cover accepted-prefix results and continuation; they do not separately compare native KV/GDN/QSA/PLE state or discarded-row logits. Exact equality is stronger than the normal numerical thresholds for these captured packets, not proof of general model parity. One-row packet SHA-256 in both arms: `4a88d8c5dbe213c6800648bb233fdfcea4701d1751b9d1ffe1f8050dcd50f86d`.

In separate resident reference/direct launches with prompt IDs `[248045,846,198]`, an 8-token cap and stop ID `248046`, requests 1 and 2 both generated `[96220]` and stopped; request 3 was cancelled at a progress checkpoint; request 4 after reset again generated `[96220]` and stopped. Both processes exited 0. This is a cooperative cancellation at a completed checkpoint, not arbitrary in-flight GPU failure or uncertain-drain qualification. An earlier reference invocation sent the next request instead of the mandatory progress `continue`; it failed with `expected continue or cancel`, then the correct interactive protocol succeeded in fresh processes. No probe/worker remained alive after runs; no forced GPU kill.

## Outstanding admission and performance

- **Native state parity NOT_RUN:** add a bounded capture seam compatible with this direct mode and compare all relevant committed CUDA/HIP state across full and partial prefixes, including discarded rows and reset; then broaden prompt/window coverage and negative identity, setup failure, timeout and cancellation tests. Current full logits and one continuation cannot substitute for this.
- **Trace-off balanced direct/reference latency, greedy rate and GPU profiling NOT_RUN:** required state and failure-path admission is incomplete. No rate or speedup inferred from the synthetic graph probe. Report 0036's optimized compact 62.64 tok/s is a different mode and is not a matched full-logit direct comparison. Memory telemetry and peak reserves were not independently sampled for these short runs.
- **Fully prelaunched CUDA→HIP real inference NOT_RUN:** current CPU finite scan happens after CUDA drain and before HIP consumes. For GPU-only prelaunch, implement and independently validate an equivalent every-element finite gate before consumption, or a proven quarantine that prevents unsafe data from changing durable state. The native timeout wait may continue executing downstream nodes, so simply withholding ready is unsafe. Bind request/window/shape/generation identities, fence flag reset, establish producer and all-stage drains, propagate errors, and poison on uncertain completion before any reuse.
- Native bulk prefill has a separate chunk/event lifecycle and is not qualified with this candidate. 12/18/18 direct reference, matched authorized vanilla, near-32K and MTP remain NOT_RUN. Report 0036's coding control returned `30` instead of `14`; cause remains unresolved.

Private logs, captures and control results are in the designated 0038 scratch directory. No GitHub Actions, downloads, repack, reserve reduction, spill, hardware setting change, unrelated service stop or unauthorized resource use. The pre-existing untracked pool probe was preserved.
