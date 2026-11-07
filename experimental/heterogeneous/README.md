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

## Wire protocol v1

POSIX Unix-domain stream sockets. Little-endian, IEEE754 float32.
A fixed 56-byte header carries magic/version, kind, stage, rows, geometry,
request/window IDs, plan fingerprint, payload length and FNV-1a corruption
checksum. Payload is either an ordered float32 handoff window or a bounded
error message. 16 MiB per-frame cap; prefill needs chunking.

The coordinator submits a window to each stage in order. On errors, truncated
transfers, stale IDs, incompatible plan/geometry or wrong output sizes, the
window is **not published**. The worker exits after an execution failure.
The unit tests use **actual separate POSIX processes**, via `fork()` and
`socketpair()`. The `strata_heterogeneous_demo` executable additionally launches
workers with `posix_spawn`/`exec`: each runs in its own address space and has only
a local Unix socket. Both paths use deterministic CPU executors. Different GPU
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
- Only synchronous single-window scheduling exists; concurrency, timeouts,
  cancellation, worker restart and authentication are not implemented.
- FNV-1a catches incidental corruption, **not malicious input**. Use only
  local trusted sockets; don't expose the worker protocol over a network.
- A future worker must validate actual device memory allocations, handle
  device/stream errors, and return failure rather than decoding partial state.
- No speedup can be inferred without profiling the R9700/R9700/4070 Ti SUPER
  setup and the PCIe/OCuLink transfer behavior.

## Next implementation increments

1. Add an `exec`-based worker launcher and device-runtime isolation.
2. Wrap Strata's `Verifier::set_stage()` and `set_handoff_rows()` in CUDA/HIP
   stage executors, with GPU state/weights resident only on their owner.
3. Implement pinned/staged GPU copies + reusable shared host slots and
   explicit events; preserve strict failure-before-publication semantics.
4. Apply the same scheme to `Prefill` chunks and session save/restore.
5. Use local GPU tests for logits/parity, checkpoints, multi-GPU throughput,
   and CPU/GPU expert-cache accounting.

The module is opt-in and leaves existing Strata build settings, CI workflows,
and inference defaults unchanged.
