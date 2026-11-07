# Experimental heterogeneous layer handoff — host-only foundation

Status: **CPU-only prototype, not an inference backend**. No GPU execution, CUDA/HIP
integration, model loading, KV state, speculative decoding, or performance claims.
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

## Wire protocol v3 — staged handoff, commit, and optional shared memory

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
socket. It still copies between the memfd and the current vector-based CPU
`StageExecutor` interface. It is **not zero-copy into a GPU** and has not
been benchmarked. Real CUDA/HIP workers must explicitly synchronize their
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
