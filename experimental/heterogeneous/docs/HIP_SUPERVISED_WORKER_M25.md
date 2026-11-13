# M2.5 — Isolated supervised HIP Q2_0 worker (opt-in, not hardware validated)

**Baseline:** M2.4 `f80107c2`. This is source-level assembly of a
full HIP-only worker, pending native compile/link and runtime validation.

The executable `strata_heterogeneous_hip_worker` is built only inside
a Linux wave32 HIP-enabled Strata tree when
`STRATA_BUILD_HETERO_HIP_WORKER=ON` (OFF by default). It cannot be
linked into a CUDA process. An exact invocation from the coordinator
must supply the stage-indexed configuration path, model/stage identity,
backend-local ordinal, and validated Q2_0 compiled geometry. The worker
uses the existing v2 GGUF/PLE config, pack-index SHA-256 and STRP/v1
profile preflight before any HIP runtime initialization.

After these checks, runtime loading is still blocked unless the user has
separately authorized GPU execution and the worker's process environment
explicitly sets `STRATA_HETERO_HIP_LIVE=I_APPROVE_HIP_RUNTIME`.
No authorization was granted by creating this code. A denied startup
returns a bounded v4 failure frame and exits nonzero.

On a separately authorized live startup, `HipStageResources` loads
stage-carved packed weights and fresh session; `HipNativeStage` binds
NativeDense/PLE; `HipFileExperts` loads real Q2_0 file-backed expert
pool and residency; the worker creates either `HipVerifierDriver` or
`HipFinalVerifierExecutor`. M1.18 `WindowLimits` enforce stage token
and position admission, finite intermediate activations, positive prefix
commits and replay rejection in both server types. Cleanup order is
verifier → native dense/PLE → experts/session/weight owner.

This *does not* implement prompt prefill, state restoration, sampling
or front-end serving integration; a fresh-state v4 session is the only
intended initial case. It is not proof of correct HIP model loading,
CUDA→HIP numerical parity, accepted-prefix hardware commits or stability.

## Compile-only validation instructions

Build in a **separate HIP** engine directory with
`-DSTRATA_ENABLE_HIP=ON -DSTRATA_BUILD_HETERO_HIP_WORKER=ON`
and `--parallel 28`. A CPU-only standalone
`cmake -S experimental/heterogeneous` should still run all 27 host
CTest targets. Native HIP linker failures and ownership mismatches
must be corrected before considering live GPU authorization. The owner
must not enable GitHub Actions or touch separate workspaces. No GPU
driver operations are authorized by this milestone.
