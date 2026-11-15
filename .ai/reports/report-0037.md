# 0037 — direct native graph hardware qualification

**Status: PARTIAL.** The model-free CUDA→HIP→HIP graph candidate passed the staged positive hardware matrix and a separate withheld-publication timeout on Oberon. It is not integrated into real inference. Live-model parity, inference benchmarks and production qualification are **NOT_RUN**; the passing synthetic probe is not inference performance. No native synchronization kernel, inference default or safety gate was changed.

## Build and environment

Started from `53d8f42cd95af6aeada0e941aab143f96045dbc8` on `feature/heterogeneous-hip-only`. Used the existing vanilla-aligned CMake caches and checkout venv. Explicit CMake regeneration was necessary after the fast-forward: the initial target builds reported no rule; regeneration followed by `-j28` succeeded for the host layout test and both native graph probes. CUDA cache: `/usr/bin/c++`, `/usr/bin/nvcc` CUDA 12.0.140, sm_89; HIP cache: `/usr/bin/c++`, ROCm clang from core-10.1, HIP 7.16.26385, gfx1201; CMake 3.28.3. CUDA and HIP binaries SHA-256 respectively `457a4ff8c4379508cb90cdb80d6284bf20de03bc2613a7b81b1c611816fd851a` and `031cb1c7fef01270ad179b330c040dd8877a3a7325e773f6f349aeb86db1dbbe`.

Host CTest `-j28`: 100/100 passed, including layout and CLI help. Direct graph Python tests: 5/5 passed, including expected fake-worker failure. An additional combined unittest invocation passed those five and eight critical-path tests but exited nonzero because the additionally requested `tools.test_heterogeneous_native_prefix` module does not exist; this was a test invocation error, not a passing test.

Device identity from each worker: CUDA ordinal 0 NVIDIA RTX 4070 Ti SUPER PCI `0000:04:00.0`; HIP ordinal 0 and 1 AMD Radeon AI PRO R9700 PCI `0000:03:00.0` and `0000:07:00.0`. Before testing, NVIDIA showed no compute processes and AMD SMI showed no running processes; after testing neither showed compute processes and no probe worker remained. No GPU worker was forcibly killed.

## Model-free hardware evidence

Each independent invocation created a fresh size-sealed memfd, independently registered it in both runtime processes, resolved runtime-local aliases, created all graphs before live waits, prelaunched HIP1 then HIP0, checked both pending and zero flags, and launched the CUDA native publisher. Every successful round checked ready/error words, drained both HIP graphs, read back **every float** on each HIP device against the sequence-dependent pattern, then drained CUDA before reuse. Explicit graph destruction precedes unregister; both processes returned BYE and exit 0. The payload is native R/bo/inj geometry HC=4, embedding 2560; no CPU payload relay during handoff. Successful READY from each worker implies registration and graph creation succeeded, rather than being a separate instrumentation trace.

| Rows | Bytes per edge | Initial rounds | Independent 128-round runs | Independent 1024-round runs |
|---:|---:|---:|---:|---:|
| 1 | 51,216 | 16 pass | 2 pass | 2 pass |
| 4 | 204,864 | 16 pass | 2 pass | 2 pass |
| 8 | 409,728 | 16 pass | 2 pass | 2 pass |

All positive runs used two warmup rounds. The two 1024-round host orchestration medians were 0.1521/0.1548 ms (1 row), 0.3113/0.3102 ms (4), and 0.4614/0.4552 ms (8); these include IPC, full readback and drains and are neither GPU kernel times nor model throughput.

A separate fresh one-row process completed 16 rounds, then withheld publication with the default 2 s wait. Both HIP streams drained, cross-edge ready remained 0 and error became 1, both workers acknowledged REJECTED and entered poisoned state, then BYE and exit 0. No timed-out slot was reused. An explicit *attempt* to reuse the poisoned slot was not made on hardware, so active refusal of that attempt remains unqualified; the probe state machine rejects non-idle reuse. The native wait permits later nodes to execute after timeout, so its output was discarded. Full data, failure logs and JSON remain in the designated private scratch directory.

## Integration decision and blockers

**Real-model direct mode: NOT IMPLEMENTED / NOT_RUN.** Positive graph evidence alone does not satisfy the safety admission. Existing `HipFinalVerifierExecutor` receiver-arm allocates HIP-only staging and separately CPU-publishes its ready flag after copying validated transport bytes. Existing `FinalPipelineSession` serially forwards a checksum-bearing activation frame, while `WindowAdmission::validate_bytes` and `provisional` independently reject nonfinite activation bytes. CUDA mapped egress is likewise a separate CUDA-owned allocation. The existing final logit nonfinite guard is not equivalent to validating every intermediate activation. Wiring the probe's independently registered mapping into these paths without a separately validated finite-activation verdict and exact per-window identity, error propagation, drain/commit and ownership binding would bypass the current admission contract. A safe end-to-end implementation and its negative-path validation were not completed in this iteration; no guard was relaxed and no direct live path was enabled.

Next: design and independently validate a finite gate over the complete activation with publication/commit ordering and failure poisoning; bind a distinct sealed mapping and runtime-local aliases to exact plan/request/window/shape and receiver-arm ACK; prove all-stage drain and accepted-prefix commit before reuse, including cancellation and stale/timeout/error cases. Then run bounded full-logit and native-state parity for one/multiple rows and partial prefixes before reset/greedy tests. Native bulk prefill remains a separate chunk/event ownership lifecycle and is explicitly incompatible with current receiver-arm mode; it needs independent design and qualification.

## Milestones and remaining comparisons

- Shared memory: **PASS, model-free only** — sealed memfd, separate registrations and local aliases on tested hardware; no real-model ownership binding.
- Native synchronization: **PASS, bounded synthetic cases** — unchanged native publish/wait/copy, repeated windows and withheld publication; attempted poisoned reuse untested.
- Graph integration: **PASS, model-free only** — captured graphs precreated and prelaunched; real verifier graph integration not done.
- Unified pipeline: **NOT IMPLEMENTED** — finite gate and identity/lifecycle integration above remain.
- Lifecycle: **PARTIAL** — success and terminal timeout drain/BYE qualified in probe; real model cancellation, rollback, reset and uncertain-drain paths not qualified.
- Optimization and benchmark: **NOT_RUN** for direct inference, including controlled trace-off pairs, GPU timelines and bulk prefill. The existing combined opt-ins and prior actual model rates are reported separately in report 0036; they are not evidence for this candidate.
- Matched authorized vanilla and coding control: **NOT_RUN**, identity/authorization remain as in report 0036. Its wrong `30` versus expected `14` remains unresolved; no model-quality conclusion is implied.

No GitHub Actions, downloads, model repack, reserve reduction, spill, hardware clock/power change, unrelated service stop or forced kill. The pre-existing unrelated untracked pool probe was preserved.
