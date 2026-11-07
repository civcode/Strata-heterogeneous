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

## Wire protocol v2 — staged handoff and commit

POSIX Unix-domain stream sockets. Little-endian, IEEE754 float32.
A fixed 56-byte header carries magic/version, kind, stage, rows, geometry,
request/window IDs, plan fingerprint, payload length and FNV-1a corruption
checksum. The v2 wire protocol adds a 16-byte worker hello (backend, local
ordinal, first and last layer), ready handshake, and commit/committed control
frames. A normal data payload remains an ordered float32 handoff window;
errors carry a bounded message. 16 MiB per-frame cap; prefill needs chunking.
A complete frame has a monotonic I/O deadline (30 seconds by default),
including short reads/writes. This is a breaking protocol change: v1 worker
binaries are not compatible with v2 coordinator binaries.

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
failure. The `strata_heterogeneous_demo` executable launches workers with
`posix_spawn`/`exec`: each runs in its own address space and has only a local
Unix socket. Both paths use deterministic CPU executors. Different GPU
runtimes will later be compiled and linked into separate worker executables.

The socket protocol is for correctness experiments and interface stability.
It is *not* the intended high-throughput activation path; mmap/shared-memory
buffers and explicit completed GPU→host / host→GPU DMA will replace the
bulk socket copies. No CUDA-pinned pointer is ever treated as a HIP pointer.

## Build/test locally, no GPU required

```bash
cmake -S experimental/heterogeneous -B /tmp/strata-hetero-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build /tmp/strata-hetero-build
ctest --test-dir /tmp/strata-hetero-build --output-on-failure
/tmp/strata-hetero-build/strata_heterogeneous_demo \
  cuda:0:0-3,hip:0:3-7,hip:1:7-10 10 2 6 12345 8
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
- `fork()` is used only in the CPU unit tests, before any GPU runtime
  initialization. The executable smoke test uses `posix_spawn`/`exec`.
  Production GPU workers must similarly be independently executed.
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
3. Implement pinned/staged GPU copies + reusable shared host slots and
   explicit events; preserve strict failure-before-publication semantics.
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
