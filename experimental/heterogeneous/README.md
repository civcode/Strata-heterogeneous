# Experimental heterogeneous layer handoff — host-only foundation

Status: **not an inference backend**. The demo and tests remain CPU-only.
An opt-in [CUDA non-final-stage adapter](docs/CUDA_VERIFIER_ADAPTER.md) now
compiles and links against the real verifier, but has not run on a GPU.
Model loading, the real final-stage CUDA/HIP head implementation and mixed-vendor inference remain unfinished.
No numerical correctness or performance claims.
This directory can be added to `civcode/Strata-heterogeneous` without changing the
upstream CUDA/HIP CMake paths or running GitHub Actions.

## Exact Strata boundary used

`include/strata/core/verify.hpp` defines `Verifier::set_stage()` and
`Verifier::handoff_floats()`. For every row/token, a stage transfers the
**ordered float32** arrays:

1. `R`: `hc * n_embd` floats;
2. `bo`: `n_embd` floats;
3. `inject`: `hc` floats.

This prototype preserves that per-row layout. It is **not** interchangeable
with a complete Strata session snapshot: layer-owned K/V, GDN/recurrent state,
PLE history and prefill semantics are retained by their owning execution stages.

## Manual placement for the target workstation

Example only; determine actual geometry and ranges from the selected model:

```
cuda:0:0-12,hip:0:12-30,hip:1:30-48
```

The CUDA ordinal is from `CUDA_VISIBLE_DEVICES` *inside the CUDA worker*;
HIP ordinals are from `HIP_VISIBLE_DEVICES` *inside the HIP worker*. They do
not name a common process-wide device table.

`Plan::validate()` rejects missing/overlapping/empty layer ranges and repeated
backend+ordinal pairs. `Plan::fingerprint()` includes the *model fingerprint*,
geometry and complete stage layout, so incompatible worker configurations
cannot silently exchange activations. Model fingerprint derivation from Strata's
loaded model is **not yet implemented**; callers must provide the same stable
nonzero identity on every worker.

## Wire protocol v4 — verifier window metadata, deferred commit, and shared memory

POSIX Unix-domain stream sockets. Little-endian, IEEE754 float32.
A fixed 56-byte header carries magic/version, kind, stage, rows, geometry,
request/window IDs, plan fingerprint, payload length and FNV-1a corruption
checksum. Protocol v3 keeps the identity hello, ready handshake, and
per-window commit/committed control messages. The optional shared-memory
mode uses a distinct `hello_shared` handshake and passes one sealed,
16 MiB Linux memfd to each isolated worker through SCM_RIGHTS. After setup,
`shared_data` frames carry a four-byte FNV-1a digest instead of the float32
activation payload. The owner writes the mapped slot, then signals the worker
on its control socket; a worker completes its CPU operations before replying.
Each stage validates mapped bytes before execution, and the coordinator
validates replies before advancing. Socket-data mode remains the default.
Frame capacity is 16 MiB in both modes; prefill needs chunking. A complete
socket frame has a monotonic I/O deadline (30 seconds by default). Protocol
v3 does not interoperate with v1/v2 worker binaries.

The coordinator first verifies every worker's declared backend, ordinal,
layer interval, plan fingerprint, and geometry, then authorizes execution.
Each stage computes a provisional window and waits for a matching commit;
only after *all* stages acknowledge commit is the final result published.
On timeouts, stage failures, truncated transfers, stale IDs, incompatible
plans, bad shapes or commit failures, the session is poisoned and its sockets
are shut down to wake outstanding workers. A partially committed transaction
is **not rolled back**: the session must be abandoned and recreated. The
stateful StageExecutor API is a CPU-only contract; a real stage must not
advance durable model state during the provisional run.
The unit tests use **actual separate POSIX processes**, via `fork()` and
`socketpair()`, plus fault injection for handshake, worker-exit and commit
failure. The `strata_heterogeneous_demo` executable now uses the reusable
`WorkerProcessGroup` supervisor to launch **two different CPU-only mock
worker binaries**: `strata_heterogeneous_mock_cuda_worker` for the CUDA
stages and `strata_heterogeneous_mock_hip_worker` for the HIP stages.
They have compile-time backend markers, use separate address spaces and
communicate only through local Unix sockets. They do **not** link or
initialize either GPU runtime, and the marker is not hardware attestation.
Actual CUDA and HIP executors will replace these mock binaries.

The shared-memory path avoids shipping bulk float32 activations through a
socket. In M1.7, a `StageExecutor` can override `run_staged_window` to
operate directly on the borrowed host byte mapping, avoiding the intermediate
full-window CPU vector copies. The default virtual method preserves the
existing vector-based `run_window` interface through a validated fallback.
This is **not zero-copy into a GPU** and has not been benchmarked. Real CUDA/HIP workers must explicitly synchronize their
D2H/H2D transfers and use their own runtime-owned device memory; mapped
host pointers are not automatically accessible in both runtimes. Only one
window may use the shared slot at a time; no ring buffers, overlap, or
pipelined prefill is implemented. FNV-1a detects accidental corruption,
not malicious concurrent writes. Do not expose this protocol publicly.

## Build/test locally, no GPU required

```bash
cmake -S experimental/heterogeneous -B /tmp/strata-hetero-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build /tmp/strata-hetero-build --parallel 28
ctest --test-dir /tmp/strata-hetero-build --output-on-failure
/tmp/strata-hetero-build/strata_heterogeneous_demo \
  cuda:0:0-3,hip:0:3-7,hip:1:7-10 10 2 6 12345 8
# Opt-in shared-memory transport, still CPU-only:
/tmp/strata-hetero-build/strata_heterogeneous_demo \
  cuda:0:0-3,hip:0:3-7,hip:1:7-10 10 2 6 12345 8 --shared
```

To run with sanitizers (Clang/GCC on supported Linux environments):

```bash
cmake -S experimental/heterogeneous -B /tmp/strata-hetero-asan -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer' \
  -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address,undefined'
cmake --build /tmp/strata-hetero-asan --parallel 28
ctest --test-dir /tmp/strata-hetero-asan --output-on-failure
```

## Limitations before calling this mixed-vendor inference

- **Not integrated into the Strata engine or server.** This work does not
  modify `Verifier::run`, `Prefill::run`, layer weights or `generate.cpp`.
- `fork()` is used only in CPU tests that exercise the core protocol.
  `WorkerProcessGroup` uses `posix_spawn`/`exec` for backend-tagged worker
  binaries, owns child PIDs and sockets, waits for normal EOF shutdown and
  forcibly reaps stuck/abandoned workers. These workers are CPU mocks;
  production GPU workers must be independently compiled and GPU-tested.
- Workers don't restore real model state. Full saved-conversation semantics
  require coordinating the stage-local snapshots as a transaction.
- Only synchronous single-window scheduling exists. Bounded socket timeouts
  and fail-closed socket shutdown exist; application-level cancellation,
  worker restart/recovery, retry, parallel scheduling and authentication do
  **not** exist.
- FNV-1a catches incidental corruption, **not malicious input**. Use only
  local trusted sockets; don't expose the worker protocol over a network.
- A future worker must validate actual device memory allocations, handle
  device/stream errors, and return failure rather than decoding partial state.
- The coordinator handshake verifies worker-declared identity, not actual GPU
  hardware attestation. Backend workers must validate and report their own
  runtime/device identity during real `StageExecutor::prepare`.
- No speedup can be inferred without profiling the R9700/R9700/4070 Ti SUPER
  setup and the PCIe/OCuLink transfer behavior.

Coding agents: read [AGENTS.md](AGENTS.md) before modifying this experimental module.

## Next implementation increments

1. Replace the mock backend in the existing `posix_spawn`/`exec` workers
   with separately compiled CUDA and HIP worker executables. Keep runtime
   isolation and implement a real stateful stage lifecycle.
2. Wrap Strata's `Verifier::set_stage()` and `set_handoff_rows()` in CUDA/HIP
   stage executors, with GPU state/weights resident only on their owner.
3. Connect the host's now-available shared-memory slot to explicit,
   synchronized pinned/staged GPU copies with backend-local device buffers
   and events; preserve strict failure-before-publication semantics.
4. Apply the same scheme to `Prefill` chunks and session save/restore.
5. Use local GPU tests for logits/parity, checkpoints, multi-GPU throughput,
   and CPU/GPU expert-cache accounting.

The module is opt-in and leaves existing Strata build settings, CI workflows,
and inference defaults unchanged.

### Milestone M1 completed on the host

Implemented: stateful stage-executor lifecycle (prepare, provisional run,
commit, abandon, shutdown); versioned identity handshake; bounded Unix-socket
I/O; session poisoning; worker exit/commit failure tests. **Only host-side
correctness is demonstrated.** No GPU validation was possible here.

### Milestone M1.5: exec worker supervision (host-only)

`WorkerProcessGroup::launch(plan, cuda_executable, hip_executable)` creates
one isolated OS process for each stage, inherits only a local control socket
on file descriptor 3, and owns the PIDs and coordinator sockets. Call
`PipelineSession(plan, group.sockets())` while the group remains alive.
`group.finish(error)` closes the coordinator sockets, waits a bounded time
for worker EOF shutdown and reports abnormal exits. Its destructor force-kills
and reaps any remaining workers; it does not attempt session recovery.
The worker executable paths must be absolute. The mock executable verifies
its compiled backend marker matches the stage, but does not attest hardware.

The third standalone CTest target, `strata_heterogeneous_process_tests`,
exercises successful three-stage execution, backend-mismatch rejection,
failed worker startup, worker exit and PID/descriptor cleanup. No model
weights or GPUs are used. Existing production engine targets are untouched.

### Milestone M1.6: Linux shared-memory handoff (host-only)

`SharedActivation` creates a size-sealed 16 MiB memfd, maps it once in
the coordinator and passes the same descriptor to each worker using the
existing AF_UNIX socket. `PipelineSession(..., shared_memory=true)` and
`WorkerProcessGroup::launch(..., shared_memory=true)` select v3 shared mode.
Shared mode is explicitly opt-in; ordinary socket-data mode is the default.

Two new CTest checks cover shared object creation/descriptor verification,
map visibility across processes, transfer bounds, checksum failure,
wrong-mode handshakes, multiple windows and multirow execution. CPU-only
GCC, Clang, ASan and UBSan builds/tests are required for changes here.
No actual GPUs or full model engines were exercised.

### Milestone M1.7: Borrowed in-place activation views (CPU-only)

`ActivationView` validates row geometry and the 16 MiB cap, and exposes
each row's exact `R | bo | inject` byte ranges through `HandoffRow`.
The float accessors use `memcpy`, not a reinterpret-cast from the untyped
Linux mapping to `float*` (which would violate C++17 object lifetime rules).

In shared mode, workers dispatch to
`StageExecutor::run_staged_window(metadata, view, error)`. The mock
executor implements a direct in-place CPU path; other existing executors
can rely on the default vector fallback. The latter validates result
identity/shape **before writing to shared memory**. The socket-data mode
and wire protocol v3 remain unchanged. The view is borrowed for only one
provisional window; it is not a CUDA/HIP pointer or an owned allocation.

`strata_heterogeneous_activation_tests` verifies multirow layout,
bit preservation including a NaN payload, bounds and overflow rejection,
direct/fallback parity, direct worker dispatch, and fail-closed behavior
when an executor modifies mapped bytes and subsequently returns an error.
Full Strata, model weights, GPU DMA and numerical GPU parity are still
untested and unimplemented.

### Milestone M1.8: Backend-neutral device transfer ordering (CPU-only)

`include/strata_heterogeneous/device.hpp` defines `DeviceDriver`, a narrow
interface for a **single** backend-specific worker. `DeviceStageExecutor`
adapts it to the existing staged/session protocol without linking any GPU
runtime into the host library. Each provisional window must execute:

1. enqueue host-to-device transfer, then **wait for completion**;
2. enqueue this stage's layer execution, then **wait for completion**;
3. enqueue device-to-host transfer, then **wait for completion**;
4. return a provisional result; only after the pipeline asks for commit,
   enqueue stage-local commit and **wait for completion**.

Both socket-data and opt-in memfd transports use this ordering. On a failed
enqueue, wait, commit or thrown driver exception, the stage is poisoned and
aborted; neither result nor next window is published. A failed D2H operation
may have partially modified the shared host slot, so *fail-stop is not
rollback*. Real drivers must impose finite GPU wait deadlines and must drain
in-flight DMA before relinquishing borrowed host memory, or terminate their
worker process when an unrecoverable GPU hang prevents safe cleanup.

`strata_heterogeneous_device_tests` uses a CPU fake with queued operations.
It checks upload/execute/download/commit ordering, stage identity, successful
two-stage worker pipelines in both transports, and injected transfer/commit
failures (including actual child-process error propagation). It does **not**
execute or test CUDA/HIP kernels.

**Remaining real-Verifier requirements:** `Verifier::run(T, tokens, pos0,
pool, user, out, err)` requires token IDs, position, pool callbacks and head
outputs. Its `commit(n_keep)` needs the accepted-prefix length. Protocol
v3 only carries activations and request/window IDs; its metadata is not yet
sufficient for real inference. Derive model identity from loaded weights,
extend the control contract deliberately (with versioning and negative tests),
then implement CUDA and HIP drivers in separate worker binaries. Existing
stateful sessions, KV/GDN/PLE/expert caches must remain stage-owned.

Oberon agent handoff: validate commit in
`/home/chris/workspace/Strata-heterogeneous`, stage any patch or test logs
only in `/home/chris/tmp/strata-heterogeneous`, and do not inspect the
unrelated `Strata-dev` locations.

### Milestone M1.9: verifier request and accepted-prefix commit (CPU-only)

Wire protocol **v4** adds a per-window `WindowSpec` control frame carrying
the starting token position (`pos0`) and ordered token IDs. Workers must
receive this metadata before activation bytes. The coordinator now supports
`PipelineSession::execute_window(input,spec)` for provisional execution,
followed **after acceptance is known** by
`PipelineSession::commit_pending(n_keep)`. Each worker must acknowledge
the exact accepted-token count, including zero, before the coordinator
returns committed output. A failed stage or mismatched acknowledgement
poisons the session. Partial commits cannot be rolled back.

The original `run_window(Frame)` remains for CPU mock tests and still
commits the whole window immediately; it does not implement real speculative
decode. Both original socket-data and opt-in shared-memory modes continue
working under the v4 framing. v3 worker executables are not compatible.

`DeviceStageExecutor` passes the new metadata to
`DeviceDriver::begin_window` and `enqueue_commit_prefix` and requires a
completed commit fence. Backends without these methods **reject** metadata
rather than silently dropping it. The test suite includes real mock worker
processes, both transport modes, invalid window metadata, zero/partial
acceptance and incorrect commit acknowledgements.

See [docs/VERIFIER_WINDOW_V4.md](docs/VERIFIER_WINDOW_V4.md) for the
protocol and the Oberon integration handoff. No actual CUDA/HIP execution,
model loading, state restoration, logits or GPU-specific acceptance logic
is implemented or verified.

### Milestone M1.10: final-stage result codec (CPU-only)

The new `FinalStageResult` contract has a standalone, versioned little-endian
packet format (result magic `STFR`, format v1) with predicted token IDs for
every row, explicit nonfinite-head markers and optional bit-exact float32
logits. It validates request/window/model/stage identity, payload bounds,
shape, vocabulary indices and checksums. Its AF_UNIX helpers have bounded
transfer deadlines and reject truncated or fragmented-invalid packets.
A `FinalResultGate` prevents publication before the caller explicitly
reports successful commit acknowledgements from **all** stages.

M1.10's codec was initially standalone. M1.11 adds an **opt-in
FinalPipelineSession** and **serve_final_stage_worker** that negotiate a
distinct final-stage hello and exchange the STFR packet on their existing
local control socket. The ordinary PipelineSession and mock demo are still
activation-only. This is **not** numerical GPU inference, an implemented
last-stage CUDA/HIP head, or a solution to zero-prefix restoration.

Run `strata_heterogeneous_final_result_tests` and see
[docs/FINAL_RESULT_V1.md](docs/FINAL_RESULT_V1.md) for the wire layout,
failure semantics and next GPU-dependent integration tasks.

### M1.11 — Opt-in final-stage worker and coordinator (CPU-only)

`FinalPipelineSession` executes verifier-window metadata and activations
through intermediate workers and receives the final worker's actual
`FinalStageResult` structure instead of another activation frame.
`FinalStageExecutor::run_final_window` is a distinct virtual interface for
the last worker; production implementations must return real head predictions.

Protocol v4 adds `hello_final` and `hello_shared_final` capability kinds
without changing the 56-byte v4 frame. After that handshake, the final
worker sends an STFR/v1 packet on the same trusted local socket; the
coordinator knows to expect that format. Intermediate stages continue to
send v4 activation frames. Both socket and shared-memory transports work.

No result is published until every worker has acknowledged
`commit_prefix(n_keep)`, with an exact echoed accepted count.
Nonfinite-head markers, identity mismatches, corrupted results, invalid
acknowledgements and worker failures poison the session. This is fail-stop,
not cross-process state rollback.

`strata_heterogeneous_final_pipeline_tests` uses forked independent CPU
mock workers with three stages, optional logits, zero/partial/full prefix
handling, both transports and failure injection. See
[docs/FINAL_PIPELINE_M111.md](docs/FINAL_PIPELINE_M111.md) for the specific
contract, validation boundary and Oberon handoff. It was CPU-only at M1.11. See M1.12 below for the separately compiled CUDA final-stage source adapter.

### M1.12 — CUDA final-stage verifier source adapter (CPU contract tested)

`CudaFinalVerifierExecutor` implements `FinalStageExecutor` for the last
CUDA stage. It passes staged activations to a backend-owned device input,
runs Strata's actual `Verifier::run` with the supplied token IDs/position,
and returns real head predictions with optional `copy_logits` diagnostics
as a provisional STFR/v1 result. It uses explicit CUDA fences and defers
positive-prefix `Verifier::commit` until the coordinator's commit request.

The default exports logits so even unmarked NaN/Inf values are caught.
Turning logits off requires the real verifier nonfinite guard to be enabled.
Zero-prefix commit is rejected (state restoration is not implemented).
A real final model loader, sampling configuration, prefill, GPU parity and
mixed-vendor inference remain **not implemented or verified**.

With `STRATA_HETERO_CUDA_VERIFIER=ON`, the real adapter is added to the
existing CUDA-only archive. The standalone CPU contract test
`strata_heterogeneous_cuda_final_verifier_contract_tests` compiles the
same source against explicit runtime/verifier doubles. See
[docs/CUDA_FINAL_VERIFIER_M112.md](docs/CUDA_FINAL_VERIFIER_M112.md).

### M1.13 — Stage-local native pack selection (CPU-only)

`StageWeightPlan` and `plan_stage_weights(plan,index,pack/index.txt,...)`
read the exact 19-column index produced by Strata's `tools/pack_index.py`.
Only `blk.<layer>.*` weights **outside** a stage's contiguous layer range
are excluded. Unknown and global weights stay selected to avoid silently
dropping embeddings, output-head weights or other shared tensors.

The parser validates index headers, count, unique tensor names, exact numeric
fields, layer coverage, offsets, alignment and overflow. It reports the
stage's compact weight-arena size using the **same per-row alignment
calculation** as `core::WeightTable::pool_bytes(pack,&skip)`. The returned
`skip` set is intended to be passed without modification to both
`WeightTable::pool_bytes` and `WeightTable::load`. Do not allocate CUDA memory
until the real native result matches the estimate.

This is a model-loader **planning component**, not a model-loading worker:
it does not allocate device memory, load weights, initialize stage-local
KV/GDN/PLE state, construct the expert pool or establish model fingerprints.
A source-level test of the existing host `Plan` and parser passes locally;
the complete native CTest suite and loaded-model validation belong on
Oberon. See [docs/STAGE_WEIGHT_PLAN_M113.md](docs/STAGE_WEIGHT_PLAN_M113.md).


### M1.14 — CUDA stage weight and session owner (CPU-double validated)

`CudaStageResources` is an opt-in CUDA-only lifetime owner for the native
`WeightTable::load` selected by M1.13 and stage-carved `session_init`.
It checks the planner against the native pool-size API, GPU free memory and
reserve, and a configurable arena budget before allocating weight and
session device arenas. It zeroes and synchronizes a **fresh** session
(position 0 only) and owns an explicitly attached expert callback provider
until after the verifier is destroyed. It handles partial load failures,
stage-offset QSA/RoPE registration cleanup and fail-stop runtime failures.

This is **not** a runnable CUDA model worker. The actual expert-source/cache
factory, model geometry/fingerprint provenance, prompt prefill, sampling
state, HIP worker and real numerical parity are not implemented. Do not run
native GPU loading until owner-approved NVIDIA recovery. The default
CPU-only build runs `strata_heterogeneous_cuda_stage_resources_contract_tests`
with runtime and engine doubles. See
[docs/CUDA_STAGE_RESOURCES_M114.md](docs/CUDA_STAGE_RESOURCES_M114.md).

### M1.15 — Real Q2_0 file-expert source and static CUDA residency

`CudaFileExperts` implements M1.14's `CudaStageExperts` interface with
the native Strata `FileExpertSource`, `ExpertCache`,
`kernels::cpu::ExpertPool` and `expert_pool_dispatch_multi`. It maps the
actual canonical Q2_0 `experts.bin`, fills profile-selected GPU slots with
real blobs, verifies each slot by reading it back, publishes matched host
and device residency tables, and computes all other routed experts with
the native CPU pool. No synthetic expert values are used.

A trusted STRP/v1 expert profile or explicitly ranked stage-local
`(layer, expert)` pairs is **required**; empty/fabricated placements
are not silently accepted. The provider refuses native/IQ expert formats
and enforces a GPU-memory reserve. Failed source/pool callbacks terminate
the worker rather than releasing partially computed GPU state.

This is **not yet a runnable worker executable**. Live GPU loading,
expert numerical parity, watchdog/failed-CUDA teardown and mixed CUDA/HIP
inference remain untested or unimplemented. See
[docs/CUDA_FILE_EXPERTS_M115.md](docs/CUDA_FILE_EXPERTS_M115.md).


### M1.16 — Supervised CUDA model-loading worker (opt-in)

An explicitly enabled native CUDA-only executable now assembles the M1.12
nonfinal/final verifier with M1.13 stage weights, M1.14 owned CUDA session,
and M1.15 real file-backed Q2_0 experts. It reuses the existing
`WorkerProcessGroup` child supervision and v4/STFR worker protocols.
Startup validates a versioned, operator-authored configuration file,
pack-index SHA-256, expert profile header, stage/plan identity and compiled
model geometry before **any CUDA runtime call**. The supervisor passes
`--config /absolute/path` only to CUDA child executables. No paths are
searched or inferred. As of M2.9, an explicitly launched supervised
worker no longer needs an additional approval environment variable.

CPU-only configuration and argv tests run in standalone builds. To build the
CUDA worker executable from the top-level native CUDA engine, explicitly
configure with `STRATA_BUILD_HETERO_CUDA_WORKER=ON` (OFF by default). No
GPU loading, parity, prefill, HIP inference or live teardown was run in this
increment. See [docs/CUDA_WORKER_M116.md](docs/CUDA_WORKER_M116.md).


### M1.17 — NativeDense projections and layer-1 PLE

A CUDA-only `CudaNativeStage` now attaches real `core::NativeDense` GGUF
projections to the M1.14 `WeightTable` and initializes the mandatory
layer-1 PLE n-gram table, weights, history, pinned staging and graph scratch.
It validates the first-stage `blk.0.attn_qkv.weight` native alias and
**refuses** to expose production verifier bindings until the native stage is
ready. Shutdown checks ownership order so native projections outlive verifier
graphs. `CudaWorkerConfig` v2 accepts explicit, ordered original GGUF shards
and a PLE-bearing shard; old v1 remains CPU-preflight-only, not live-runnable.

The CPU-only contract test compiles real native-owner source against API
doubles, including missing QKV, PLE types, partial allocations and teardown.
GPU loading, PLE numerical parity and full inference **remain unverified**.
See [docs/CUDA_NATIVE_PLE_M117.md](docs/CUDA_NATIVE_PLE_M117.md).


### M1.18 — Bounded supervised CUDA stage windows

The real CUDA worker now passes explicit context, vocabulary, window and
position limits to CPU-only `WindowAdmission` guards in both intermediate
and final-stage protocol servers. A window must have v4 `window_spec`,
valid bounded token IDs and positions, finite input/output activations and a
**positive accepted prefix**; no state advances before the verifier's commit
completes. Invalid data, nonfinite outputs, request replays and zero-prefix
commits fail closed without a successful output or commit acknowledgment.
`STRATA_HETERO_WINDOW_TRACE=1` optionally logs only stage/window phase and
position metadata; no tokens, activation values or model paths. Existing
mock-worker protocol behavior is unchanged when no limits are supplied.
The CPU-only unit and forked-worker tests bring the experimental suite to
**20 CTest targets**. No live GPU inference was run. See
[docs/WINDOW_ADMISSION_M118.md](docs/WINDOW_ADMISSION_M118.md).

### M2.0 — HIP subprocess bootstrap (CPU only)

The supervisor now supports independent per-stage `--config` paths without
changing the legacy CUDA-only config argument or unconfigured mock workers.
A strict, host-only HIP invocation parser checks plan/identity, command
syntax, backend index and absolute path before any HIP runtime exists.
The process suite probes real exec argv routing in socket/shared modes.
M2.0 does not implement a HIP model loader or mixed-vendor inference.
See [docs/HIP_BOOTSTRAP_M20.md](docs/HIP_BOOTSTRAP_M20.md).

### M2.1 — HIP native-input preflight without runtime

An independent, CPU-only `strata_heterogeneous_hip_preflight_worker` now
parses the supervisor's HIP stage argv, reuses the strict v2 GGUF shard/PLE
and pack-index SHA-256 config format, verifies backend-local stage identity,
and **always refuses supervised inference**. It does not initialize ROCm,
own GPU state, or claim any successful stage execution. The executable is
built for host tests or opt-in with
`STRATA_HETERO_BUILD_HIP_PREFLIGHT=ON`. See
[docs/HIP_PREFLIGHT_M21.md](docs/HIP_PREFLIGHT_M21.md).

### M2.2 — Non-final HIP verifier adapter (opt-in, not hardware-validated)

The HIP-only `HipVerifierDriver` now implements the existing `DeviceDriver`
interface on the real `core::Verifier` and stage-local HIP runtime. It uses
backend-owned H2D/D2H buffers, pinned staging and event fences, and only
acknowledges a matching positive prefix after commit completion. Standalone
CPU-double tests reuse the tested lifecycle and fault cases. Native compile
is opt-in (`STRATA_ENABLE_HIP=ON` and
`STRATA_BUILD_HETERO_HIP_VERIFIER=ON`), separate from any CUDA build.
This is not a model loader, final-stage adapter, actual HIP inference or
numerical parity. See [docs/HIP_VERIFIER_M22.md](docs/HIP_VERIFIER_M22.md).

### M2.3 — HIP final-stage head and accepted-prefix adapter

A separate `HipFinalVerifierExecutor` now implements the STFR final-result
boundary with real verifier head predictions, optional bounded logits,
nonfinite guards and positive-prefix commit; the code uses only the HIP
runtime shim in a native HIP build. CPU doubles cover failure/replay/final
results without running hardware. This is still **not** a runnable HIP
model-loading worker and numerical parity is unverified. See
[docs/HIP_FINAL_M23.md](docs/HIP_FINAL_M23.md).

### M2.4 — HIP stage resource and native Q2_0 owner (compile pending)

The opt-in HIP adapter archive now includes stage-owned pack weights,
carved session state, native GGUF dense projections, layer-1 PLE and a
canonical Q2_0 file-backed expert cache/CPU pool. The CUDA architecture and
lifecycle sources were adapted to a distinct wave32 HIP-only build,
with three additional CPU-double contract targets. This is not a runnable
HIP worker; native HIP builds and real model/device execution have **not**
been validated. See [docs/HIP_STAGE_OWNERS_M24.md](docs/HIP_STAGE_OWNERS_M24.md).

### M2.5 — Supervised HIP model-loading worker (opt-in, not executed)

The first separate HIP-only executable now composes the M2.0–M2.4
components: parsed stage argv, v2 GGUF/pack/profile config, compiled model
geometry, a stage-local weight/session owner, NativeDense/PLE lifetime,
real Q2_0 file-backed experts, and either a non-final HIP verifier or an
STFR final-stage HIP verifier. The worker uses the existing v4 bounded
window admission with fail-closed commits.

As of M2.9, an explicitly launched supervised HIP worker starts its native
runtime without an additional approval environment variable. CPU preflight
does not start the worker or initialize a GPU.
Native HIP builds are opt-in via `STRATA_BUILD_HETERO_HIP_WORKER=ON`,
which is OFF by default. Compilation/link and all GPU results remain
unverified until Oberon validation. See
[docs/HIP_SUPERVISED_WORKER_M25.md](docs/HIP_SUPERVISED_WORKER_M25.md).

### M2.6 — Plan-aware canonical Q2_0 asset gate (no GPU)

tools/heterogeneous_model_gate.py` can optionally invoke an additional
CPU-only native file preflight; it independently checks the complete cross-stage
plan, per-stage identity, ordered original GGUF shards, Q2_0 expert types,
correct PLE-bearing shard, canonical expert pack size/layout and
complete STRP/v1 profile. Stage-device ordinals remain backend-local.
This is structural validation, **not** a full-model authenticated hash,
native runtime test or numerical-parity claim. No model files are modified.
See [docs/MODEL_PLAN_GATE_M26.md](docs/MODEL_PLAN_GATE_M26.md).

### M2.7 — Passive GPU/driver readiness inventory

The CPU-only `tools/heterogeneous_hardware_readiness.py` reports Linux
PCI functions, bound driver metadata, optional owner-supplied offline
kernel-fault markers, and per-stage Q2_0 compact weight/cache planning
figures from the actual pack index. It does **not** initialize CUDA/HIP,
run `nvidia-smi`/`rocminfo`, load models, or infer runtime ordinals.
Missing optional `--bind` declarations no longer block passive inventory;
contradictory supplied bindings or insufficient device counts still do.
Even declared mappings remain *runtime unverified*. Session/native
weights and free VRAM are not measured. The 19 synthetic fixture tests
are the standalone suite's **29th CTest** if Python3 is available.
See [docs/HARDWARE_READINESS_M27.md](docs/HARDWARE_READINESS_M27.md).

### M2.8 — Offline runtime identity evidence reconciliation

The CPU-only `tools/heterogeneous_ordinal_reconciliation.py` correlates
a pre-exported M2.7 JSON PCI inventory with separate, operator-supplied
CUDA and HIP runtime ordinal-to-BDF evidence collected during an
explicitly initiated GPU test. It never touches driver APIs or
devices itself. With absent evidence, this **optional identity check**
reports `BLOCKED`; this is not a prerequisite to starting controlled
runtime/inference integration. Even consistent evidence yields only
`IDENTITY_CORRELATED_ONLY`, not a GPU-health or numerical-parity claim.
The 23 synthetic tests are registered
as the standalone project's **30th CTest** when Python3 is present.

See [docs/ORDINAL_EVIDENCE_M28.md](docs/ORDINAL_EVIDENCE_M28.md).

### M2.9 — Manual native three-GPU smoke launcher

The host-only `strata_heterogeneous_live_smoke` executable performs
CPU-only v2 config/plan checks by default. **Only an explicit `--run`**
starts the separate CUDA/HIP native workers, exercises one or more real
Q2_0 token windows, and requires an STFR final result plus acknowledged
positive-prefix commits. CUDA/HIP native workers no longer require the
redundant `I_APPROVE_*_RUNTIME` environment variables. On worker startup,
each stage reports backend-local ordinal, actual runtime GPU name, PCI
function when available and free VRAM. This is observability, not a new
PCI identity gate. The new host-only CLI help test is registered as CTest;
**no live GPU test is registered in CTest or Actions**.

Native GPU initialization and cross-backend correctness are **not yet
validated on Oberon**. The full manual compilation, preflight, socket,
shared-memory and successive-window test sequence is in
[docs/LIVE_GPU_HANDOFF_M29.md](docs/LIVE_GPU_HANDOFF_M29.md).
