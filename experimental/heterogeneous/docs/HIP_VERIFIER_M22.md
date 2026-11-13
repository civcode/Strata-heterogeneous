# M2.2 — Non-final wave32 HIP verifier adapter

**Baseline:** M2.1 baf4ebf7. **Status:** source implemented, native HIP
build/link and GPU execution still unverified.

`HipVerifierDriver` is a separate HIP-only DeviceDriver that configures
`core::Verifier::set_stage` for the worker's contiguous non-final range,
sets handoff row count and delayed state publication, retains a copy of
token IDs/position, and refuses zero-prefix commits. After an H2D copy
or D2H copy, it records an event on the verifier stream and waits within
a configured deadline. Its intermediate activations cross runtimes only
as verified host bytes; no GPU pointers/events cross processes. Aborted
operations fail-stop rather than claiming rollback.

The implementation reuses the CUDA lifecycle, with strict `Backend::hip`
checks and a HIP-only compile guard. Strata's existing
`strata_hip_runtime` target owns the HIP include order, force-includes
the CUDA-shaped HIP shim, and supplies `STRATA_USE_HIP`. The HIP archive
must only be linked into a HIP worker image, never CUDA or supervisor.

## Build/checks

Standalone host-only CTest builds `src/hip_verifier.cpp` with deterministic
CUDA-shaped API/Verifier doubles and the `STRATA_USE_HIP` flag.
There are **23** expected standalone CTest targets. The real native
adapter can be compiled from a separate HIP engine build by setting
`-DSTRATA_ENABLE_HIP=ON -DSTRATA_BUILD_HETERO_HIP_VERIFIER=ON`.
Build with `--parallel 28`, without executing any GPU runtime or model.
No device inference/numerical parity result is claimed.

Still missing: HIP stage-owned GGUF weights/NativeDense/PLE, Q2_0 expert
cache and session, runnable HIP worker main, final-stage output path,
end-to-end CUDA->HIP session and parity. These require further
implementation and separately approved live device testing.
