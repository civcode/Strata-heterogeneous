# M2.3 — HIP final-stage verifier and STFR response boundary

**Baseline:** M2.2 `4cd34ea5`. Source-level implementation with a CPU
contract fixture, not live GPU inference.

`HipFinalVerifierExecutor` is the final-stage counterpart to the
non-final `HipVerifierDriver`. It uses the existing HIP engine's
`core::Verifier` with `set_stage`, head sampling enabled and no
intermediate handoff output. A worker-owned device input allocation,
pinned host staging and HIP stream fence enforce complete host-to-device
upload before a verifier window runs. The stage validates token/position
identity and the last stage's range before generating the STFR result:
predicted tokens, optional bounded logits and nonfinite row markers.

The stage returns only a **provisional** result. Positive-prefix
commit waits for the engine commit and completion fence before position
advance. Zero-prefix, nonfinite or mismatched result/commit fails closed
and may poison the stage; it must never fabricate a successful commit.
The actual HIP calls are spelled using the existing CUDA-shaped runtime
API but map through Strata's HIP runtime shim, never NVIDIA CUDA.

The HIP-only adapter archive now contains both non-final and final-stage
sources. The new standalone CTest compiles the *same final source*
against CPU fake verifier/runtime headers. **24 standalone CTests**
expected. Native HIP CMake compilation/link on Oberon remains unverified;
no GPU initialization or numerical parity performed.

Still missing: HIP-owned native GGUF/weight/session/expert resources,
HIP worker executable that enables these adapters, and complete
cross-vendor staged inference with accepted prefix. Real GPU actions
require separate approval.
