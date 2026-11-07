# CUDA verifier adapter integration increment

Status: **compiled and linked, not hardware-validated inference**.
The standalone demo still launches CPU mock workers. No production default,
wire format, existing engine source or CUDA/HIP build guard changed.

## Implemented boundary

`CudaVerifierDriver` implements `DeviceDriver` using the real
`core::Verifier`. Build it only in an independent CUDA worker and wrap it in
`DeviceStageExecutor`. It accepts **non-final stages only**, including layer
zero. Protocol v4 has no head-result channel, so rejecting the final stage
prevents activation bytes from masquerading as token picks or logits.

The worker's loader must supply a live `WeightTable`, `ModelGeometry`,
range-carved `SessionState`, `VerifyHits` residency/cache pointers and a
`PoolMultiFn` with its worker-local context. These are exclusive borrows:
keep them alive and unchanged until the driver is destroyed. Select the
worker's backend-local device before loading them. The supplied fingerprint
is checked against the plan, but is **not derived from weights or attested**.
Loading weights and launching a real model worker remain unimplemented.

The adapter owns its verifier, two CUDA handoff allocations, a pinned host
staging allocation and an event. It calls `set_stage` and
`set_handoff_rows` before `init`, with no in-process `next` verifier or
cross-device flags. Only CUDA allocations are passed to `set_stage`. The layer-zero stage
embeds tokens and receives a null `handoff_in`, so it now omits the unused
input allocation and H2D transfer; middle stages still upload and wait.
Each required transfer uses the verifier stream and a completion event. Borrowed
socket/memfd bytes are copied through owned pinned storage; DMA never keeps
a pointer into the borrowed IPC slot. Download writes the slot only after
the event succeeds. Graphs are destroyed before captured allocations.

One driver serves **one already initialized sequence**. `initial_position`
must match the caller's stage-local prefill/restored state. Each subsequent
window must use the same request ID, an increasing window ID and the position
advanced by the acknowledged prefix. New requests require new stage state.
Rows are limited to `kVerifyMaxT` (8 in the tested build), vocabulary and
context are checked before execution, and legacy metadata-free calls fail.

## Engine findings and unresolved requirements

These conclusions come from reading `include/strata/core/verify.hpp` and
`src/core/verify.cpp`, not from numerical GPU measurements:

- `run(T,tokens,pos0,pool,user,out,error)` needs a functioning expert pool,
  per-stage device residency, KV/indexer/GDN state and PLE pages/history.
  Activations alone are not a session snapshot. The stage owning layer 1
  owns PLE history; `commit` advances it once for the accepted tokens.
- A T=1 window may self-commit GDN/indexer state. The adapter calls
  `set_always_publish(true)` before init/capture to disable that path.
  This also forces expert-pool publication: it is not a performance claim.
- `Verifier::commit` accepts **1..T**, unlike v4's **0..T**. Zero is rejected
  and poisons the pipeline; it must never receive a success ACK. Skipping
  commit is not a proven zero-acceptance implementation: `run` overwrites
  indexer keys whose tail needs restoration. Add an engine restore/discard
  operation and hardware regression tests before supporting zero.
- Positive prefix commits are followed by an event fence even when the
  engine's process-global async-commit mode is enabled. Host-side PLE/state
  may already have changed on failure; fail-stop is not rollback.
- `out[t]`, head logits, sampler parameters/history, bias, speculative q rows,
  nonfinite checks and final-stage acceptance need a separate, versioned
  result/control contract. They are not transported by this adapter.
- Event polling has a finite deadline. Engine `init`, `run`, `commit` and
  CUDA API calls themselves can block. **A supervising process must enforce
  a worker lifetime deadline and kill/reap it**; this library alone cannot
  guarantee a hard GPU deadline. Abort uses `release_gpu_waits`, checks the
  compute stream, and exits the worker with code 70 if safe draining fails.
  Never reuse the borrowed session after abort, including an init failure.
- Header ABI/build configuration (especially `STRATA_VERIFY_MAX_T`) must
  match the CUDA engine linked into the eventual worker. The standalone
  adapter build uses the headers' default of 8 unless supplied consistently
  by the integrating build. No HIP/SYCL/Metal worker may link this archive.

CUDA completion rules used here: NVIDIA's [API synchronization behavior](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html)
and [event management](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-runtime-api/group__CUDART__EVENT.html).
An Async call returning does not prove transfer completion; event queries
must distinguish `cudaErrorNotReady` from success and asynchronous errors.

## Local validation

Base: `7e132bd1148003e56bbf7cb54bb4731de8cb5b22`.
GCC 13.3.0, CUDA toolkit 12.0.140, Linux x86-64. Builds/logs are under
`/home/chris/tmp/strata-heterogeneous`. No GitHub Actions were run.

Passed:

- Baseline standalone Debug configure/build: **8/8 CTest tests**.
- Final standalone Debug and CUDA-option builds: **9/9 tests each**.
- ASan+UBSan, leak detection, non-PIE executables: **9/9 tests**.
- `STRATA_HETERO_CUDA_VERIFIER=ON`: real adapter compiled against actual
  Strata and CUDA headers with warnings treated as errors.
- Root CUDA build, architecture 89: `strata_engine` and dependencies built.
  Existing engine/kernel warnings remain in the log; they were not modified.
- A staging-only probe linked the adapter with the actual CUDA engine,
  kernels and core archives, without CPU doubles. Linking is not inference.

Blocked/not run:

- `nvidia-smi` reported an unknown device-handle error and no devices.
  The linked runtime probe returned **77**, reporting
  `cudaGetDeviceCount: no CUDA-capable device is detected; count=0`.
- No real model loading, CUDA stage execution, logits/token parity, CUDA
  DMA fault injection or state-restoration validation.
- No HIP adapter/build, R9700 execution or mixed-vendor inference.
- Clang validation was not run; `clang++` was unavailable.

The new contract test compiles the **same adapter source** against explicit
CPU CUDA/Verifier doubles. It covers first/middle stage wiring, owned token
metadata, both activation entrypoints, partial commit, T=1 commit deferral,
zero refusal, invalid vocabulary/context/row count/position, metadata-free
execution refusal, partial allocation cleanup and injected upload, execute,
download, commit and fence-timeout failures. It is not GPU numerical evidence.

Reproduce (choose a build directory under the staging path above):

```sh
cmake -S experimental/heterogeneous -B "$build" -DCMAKE_BUILD_TYPE=Debug
cmake --build "$build" -j4
ctest --test-dir "$build" --output-on-failure
# Add -DSTRATA_HETERO_CUDA_VERIFIER=ON for the real adapter archive.
# Sanitizers used:
# -DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer -fno-pie'
# -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address,undefined -no-pie'
# ASAN_OPTIONS=detect_leaks=1 ctest --test-dir "$build" --output-on-failure
cmake -S . -B "$engine_build" -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build "$engine_build" --target strata_engine -j2
```

## Next hardware gates

1. Restore NVIDIA device access. Implement a loaded-model CUDA worker using
   the existing exec supervisor, derive model/config identity and verify
   adapter/engine ABI settings. Add head results before an end-to-end run.
2. Compare CUDA-only split and unsplit windows from identical prefilled
   states: T=1 and multiple rows, full/partial acceptance, repeated windows,
   context boundary, PLE history and indexer block-boundary rejection.
3. Add a real zero-prefix restore path. Test zero followed by replay at the
   same position. Verify no state/result escapes after a killed worker,
   failed transfer or partially acknowledged commit in either transport.
4. Implement HIP in a **separate executable**. On each R9700 verify gfx1201,
   local ordinals, runtime/driver versions, H2D/D2H completion and stage-local
   cache/state ownership. Test CUDA->HIP, then CUDA->HIP->HIP against the
   same-day CUDA reference, recording tokens and numerical tolerances.
   Mixed-vendor correctness remains unverified until those runs exist.
