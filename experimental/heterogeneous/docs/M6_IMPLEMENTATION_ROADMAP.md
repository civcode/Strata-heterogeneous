# M6 — Performance implementation roadmap

Target: median >=160 accepted output tok/s on real-world single-agent coding; 95–100% productive GPU utilization and sustained useful power within existing hardware limits. No repetitive-copy success criterion.

| ID | Implementation milestone | State |
| --- | --- | --- |
| B1 | Join committed-window stage timings; measure worker phases, IPC, host copies, commit and inter-window gaps; summarize warm prefill/decode | Implemented; CPU fixtures passed; Oberon pending |
| B2 | Remove redundant frame serialization and activation-sized metadata copies without changing v4 bytes or safety checks | Implemented; isolated golden wire check passed; integrated/native test pending |
| B3 | Identify native GPU idle gaps, layer vs head cost, and matched vanilla-vs-heterogeneous differences | Oberon measurements required |
| B4 | Reuse native bulk-prefill and backend-owned event/transfer mechanisms; design independent slot/state ownership for overlapping known prompt chunks | Pending B3 profiling and coordinated lifecycle hardening |
| B5 | Reduce single-agent decode critical path: native graph/compute optimization, safe pipeline overlap with accepted-prefix restoration, native MTP if assets available | Pending B3/B4 evidence and ownership/asset prerequisites |

## Implementation already available

- `tools/heterogeneous_critical_path.py`: consumes opt-in native stderr traces, joins by request/window/stage, requires coordinator committed markers, separates prefill/decode and counts accepted vs evaluated rows. Reports per-window critical path, union of worker host spans, gaps, stage/phase P50/P90/P99 and coverage. `--benchmark` cross-checks committed counts, positions and host monotonic phase bounds of the **same successful resident run**. Host spans do not measure GPU kernel active time; nested spans must not be added.
- `tools/heterogeneous_utilization.py`: adds warm-prefill prompt throughput including reset, phase-aligned utilization/power/coverage, retaining warm decode. Missing prompt counts stay null.
- Actual transport optimization: data-frame encode hashes and appends directly from original float32 bytes (no temporary activation-sized vector); workers validate without throwaway encoding and copy only scalar metadata instead of duplicating activation payloads. Wire layout/checksum, shape/identity/finite/positive-commit gates, shared slot ownership, timeouts and transfer fences unchanged. Shared-memory and socket modes must both be qualified; expected gain is not yet measured.
- Native trace also isolates coordinator send-spec/send-activation/receive durations and CUDA/HIP staging host-copy scopes. Opt-in only; no new device synchronization or timing events.

Local validation: Python 6 critical-path + 8 utilization tests PASS; all four source/test blob hashes verified against GitHub at 3400b7c3. GCC C++17 -O2 -Wall -Wextra -Werror -pedantic + UBSan isolated encode function with minimal fixture types passed fixed protocol-v4 golden data (including negative zero/NaN bits), empty control and commit-prefix cases. This did not execute the real frame validator, decoder, workers or native kernels. Added integrated golden-wire regression and critical-path CTest registration. Full direct clone blocked by GitHub DNS; no linked repository builds, GPU test, speedup or new profiler measurement claimed. No Actions.

## Execution order

B1–B3 form the next Oberon batch. On return, use actual GPU timelines and matched rates to choose B4/B5 implementation rather than blindly remove fences or tune copy-only speculation. See [bottleneck rationale and historical measurements](M6_GPU_BOTTLENECKS.md) and [agentic acceptance contract](M6_AGENTIC_CODING_BENCHMARK.md). Original full-resident 12/18/18 remains reference; placement parity failure is unresolved. No parallel development or changes to existing hardware/precision budgets.
