# M2.1 — HIP configuration and fail-closed supervised preflight

**Base:** M2.0 `6534cb0f`. Runtime-free implementation, not HIP inference.

A CPU-only executable, `strata_heterogeneous_hip_preflight_worker`,
parses the actual supervisor's HIP stage argv and uses the **existing**
strict CUDA v2 configuration parser, SHA-256 pack-index verifier, STRP/v1
profile header checker and GGUF magic/path admission. This is an
intentional reuse of input *preflight*, not of CUDA runtime or allocations.
`HipWorkerConfig` binds those file inputs to a HIP-only plan backend,
ordinal, index, layer range, geometry and caller-supplied model fingerprint.
v1 configs are refused.

`--check-config /absolute/path` performs CPU-only config/file validation
without a coordinator. In supervised mode, the worker validates its stage
and config, then **always emits a failure and exits nonzero**. No ready
handshake can succeed until a real HIP adapter owns its verifier/expert
state. No ROCm libraries are linked, and no model/device inference starts.

The host-only config tests cover legitimate HIP v2, exact stage identity,
v1 refusal, PLE shard requirements and index mutation. A forked process
test launches the preflight worker and expects the coordinator to poison
the session on refusal. Standalone experimental suite: **22 CTest targets**.

Build with `cmake -S experimental/heterogeneous -B <build-dir>`, then
`cmake --build <build-dir> --parallel 28` and `ctest --test-dir
<build-dir> --output-on-failure`. To build only the preflight executable
outside host tests, set `STRATA_HETERO_BUILD_HIP_PREFLIGHT=ON`.
No GitHub Actions or GPU execution is authorized by this milestone.

## Still required for M2

A separate native HIP executable with ROCm-only linkage, loaded owned
Q2_0 stage weights and dense/PLE tensors, expert provider, HIP stream
synchronization and D2H/H2D host staging, intermediate/final verifier
execution and stateful accepted-prefix commit. Real CUDA->HIP inference,
numerical parity and GPU recovery remain separate authorized hardware
gates. Never call preflight success 'inference success'.
