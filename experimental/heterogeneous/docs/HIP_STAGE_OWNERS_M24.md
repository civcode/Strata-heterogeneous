# M2.4 — HIP stage-owned resources, native PLE and Q2_0 experts

**Baseline:** M2.3 `a641eab3`. Source prepared; full project and native
HIP compile/link on Oberon remain unverified.

The opt-in HIP-only adapter archive now contains the separately named
`HipStageResources`, `HipNativeStage`, and `HipFileExperts` source paths.
They use the Strata wave32 HIP backend's CUDA-shaped runtime shim, never
the NVIDIA CUDA runtime. The backend-neutral plan is checked for HIP
stage ownership before allocating memory; staged weight planning, packed
Q2_0 expert files, owner-only carved sessions, NativeDense projections,
and (for stages that own layer 1) the mandatory PLE table remain private
to that HIP process.

The adapted source retains the parent CUDA implementation's explicit
arena budgeting and native tensor validity checks, bounded cache/profile
admission, real CPU expert fallback, and fail-stop resource teardown.
The resource owner cannot publish verifier bindings until native weights
and the expert pool are ready, and the verifier must die before native
matrices and session/pack arena allocations.

## Validation and limits

Three new contract tests compile the corresponding HIP source against
CPU-only fake runtime and core APIs. **27 experimental standalone CTests**
are expected (24 from M2.3, three new). These fixtures can detect owner
lifetime/order/identity faults but are not native GPU validation.

A separate root HIP build with
`-DSTRATA_ENABLE_HIP=ON -DSTRATA_BUILD_HETERO_HIP_VERIFIER=ON`
should compile the HIP archive without starting the runtime. Build with
`--parallel 28`. Nothing loads the GPU automatically. Do not enable
GitHub Actions or change default CUDA/HIP engine behavior.

Still missing: runnable HIP model-loading executable and server
integration; native HIP compile/link correction as needed; actual Q2_0
weight and expert provenance; cross-vendor D2H->host->H2D handoff,
accepted-prefix commit and numerical parity. Owner approval is required
for any live GPU model or device operation.
