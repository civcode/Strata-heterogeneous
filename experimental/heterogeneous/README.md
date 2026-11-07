# Experimental heterogeneous layer handoff — host-only foundation

Status: **not an inference backend**. The demo and tests remain CPU-only.
An opt-in [CUDA non-final-stage adapter](docs/CUDA_VERIFIER_ADAPTER.md) now
compiles and links against the real verifier, but has not run on a GPU.
Model loading, final-stage outputs and mixed-vendor inference remain unfinished.
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
cmake --build /tmp/strata-hetero-build
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
cmake --build /tmp/strata-hetero-asan
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

This module is additive and **not wired to PipelineSession, a final-stage
worker, or Strata's real verifier**. The existing activation protocol stays
v4 with no wire change; do not mix these standalone packets on its control
sockets. It is **not** numerical inference, an implemented last-stage head,
or a solution to the zero-prefix restoration problem.

Run `strata_heterogeneous_final_result_tests` and see
[docs/FINAL_RESULT_V1.md](docs/FINAL_RESULT_V1.md) for the wire layout,
failure semantics and next GPU-dependent integration tasks.
