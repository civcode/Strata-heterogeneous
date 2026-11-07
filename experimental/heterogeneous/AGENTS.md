# Coding agent instructions: heterogeneous Strata inference

These instructions apply to this directory. Also follow the repository-root
[AGENTS.md](../../AGENTS.md) and [CONTRIBUTING.md](../../CONTRIBUTING.md).
This is an experimental, opt-in research feature; do not change the behavior
of existing CUDA-only or HIP-only inference.

## Mission and current state

Implement **one model with contiguous layer ranges executed by independent
CUDA and HIP workers**, eventually on an NVIDIA RTX 4070 Ti SUPER (16 GiB)
and two AMD Radeon AI PRO R9700 cards (32 GiB each). Use the existing
Strata layer-split algorithm, not tensor parallelism. The first CUDA/HIP
boundary is the priority; do not claim that 80 GiB is unified VRAM.

The code in this directory is presently a **CPU-only Linux host prototype**.
It includes:

- a validated stage plan with backend-local ordinals;
- a versioned Unix-socket frame protocol and deterministic mock executor;
- process-isolated workers launched with posix_spawn/exec by the demo;
- exact-shape, integrity, failure-propagation, and subprocess tests.

There is **no real CUDA/HIP execution**, model loading, KV/cache handling,
GPU DMA, streaming prefill, or Strata server integration in this prototype.
The README's historical increment list should not be interpreted as proof
that an exec-based launcher is missing: the demo already has one. Check the
code before proposing a change.

## Study these files first

- [README.md](README.md): existing executable and limitations.
- [include/strata_heterogeneous/host.hpp](include/strata_heterogeneous/host.hpp):
  API contract, plan, geometry, frames, worker and pipeline functions.
- [src/host.cpp](src/host.cpp): frame encoding and error handling.
- [src/demo.cpp](src/demo.cpp): isolated worker launch and CPU smoke test.
- [tests/host_tests.cpp](tests/host_tests.cpp): current correctness baseline.
- [../../include/strata/core/verify.hpp](../../include/strata/core/verify.hpp):
  \`Verifier::set_stage\`, \`set_handoff_rows\`,
  \`handoff_floats\`, stage-to-stage execution and commit semantics.
- [../../include/strata/prefill/prefill.hpp](../../include/strata/prefill/prefill.hpp):
  later prefill-stage handoff and chunk semantics.
- [../../docs/MULTI_GPU.md](../../docs/MULTI_GPU.md) and
  [../../docs/AMD_HIP.md](../../docs/AMD_HIP.md): supported same-backend split.
- [../../CMakeLists.txt](../../CMakeLists.txt):
  \`STRATA_ENABLE_CUDA\` and \`STRATA_ENABLE_HIP\` are mutually exclusive
  in an ordinary engine build. Do not bypass that guard to link both
  runtimes together; compile separate worker executables instead.

Strata's verifier handoff for **each row** is exactly three contiguous
float32 arrays, in order:

1. \`R\`: \`hc * n_embd\` floats.
2. \`bo\`: \`n_embd\` floats.
3. \`inject\`: \`hc\` floats.

The prototype's \`Geometry::floats_per_row()\` mirrors the upstream
\`Verifier::handoff_floats()\` formula. Handoff payloads are **not**
session snapshots. Keep K/V, GDN/recurrent state, PLE and expert caches
owned by the stages that execute their layers.

## Design invariants (do not compromise)

1. **Process isolation:** a CUDA process only initializes CUDA, and a HIP
   process only initializes HIP. Start production workers with exec/spawn,
   never fork after a GPU runtime is initialized. Device ordinals are
   relative to each worker's own visible-device list.
2. **No raw GPU pointers across runtimes:** never pass CUDA pointers,
   CUDA-mapped host pointers, HIP pointers, runtime events, or stream
   handles as if the other runtime could use them. Transfer bytes through
   explicitly synchronized host staging first. Design shared-memory
   transport later, with explicit ownership and host/device visibility.
3. **Exact layer partition:** each stage owns a nonempty contiguous
   \`[first_layer, end_layer)\` interval; stages cover every layer exactly
   once. Do not use tensor-parallel per-layer splitting for this work.
4. **Explicit identity:** verify model/weights, quantization-relevant
   configuration, geometry, stage plan, request ID, window ID and stage
   index before accepting data. The current model fingerprint is
   caller-provided; deriving it from the real loaded model is unfinished.
5. **Fail closed:** no next-token result, checkpoint or updated session
   may be published after a failed or incomplete stage. Distinguish
   pre-mutation resource refusal from failure after state mutation.
   Abort or rebuild poisoned stage state rather than silently proceeding.
6. **Preserve ABI:** the current protocol is version 1, little-endian
   IEEE754 float32, 56-byte header, maximum 16 MiB payload, and FNV-1a
   checksum for accidental corruption only. Any incompatible wire change
   requires a deliberate version bump and new negative tests. This is
   only for trusted local processes, not network exposure.
7. **Opt-in:** isolate new code and build flags from production defaults.
   Never alter a normal Strata backend's outputs or performance by
   default. Avoid speculative changes to unrelated kernels or model
   formats.

## Work in small, reviewable milestones

### M0 — Baseline and test hygiene (CPU-only)

- Establish that the checkout is on the intended feature branch, and
  record the exact commit before making changes.
- Run the two standalone CTest checks and the demo below. Keep a
  machine-readable log of actual local checks, not invented test results.
- Add tests for malformed/truncated frames, wrong stage/request/window,
  canceled or dead workers, resource cleanup, and process lifecycle where
  useful. Avoid network listeners and untrusted remote clients.

### M1 — Real worker boundary and stage API (CPU-only first)

- Define a narrow backend-neutral \`StageExecutor\` lifecycle:
  \`prepare(model, plan)\`, \`load_stage\`, \`run_window\`,
  \`commit\`, \`save_state\`, \`restore_state\`, \`shutdown\`
  (exact API may be refined after inspecting real call sites).
- First implement a deterministic CPU test double. Do not equate the
  existing stateless \`mock_execute\` with a real stateful verifier.
- Add a worker handshake describing backend, ordinal, supported
  features and loaded model identity. Keep the coordinator backend-free.
- Add bounded timeouts, cancellation/exit handling, explicit errors,
  and tests showing no result is published after a worker failure.
- Ensure every worker is a separate executable or independently linked
  backend binary, and the legacy CUDA/HIP CMake guard remains intact.

**Acceptance:** CPU tests and independent-process round trips pass;
no CUDA/HIP runtime is required for the host-side code.

### M2 — First actual CUDA -> HIP inference (needs GPU validation)

- Build a CUDA stage worker and a HIP stage worker separately. The
  RTX 4070 Ti SUPER requires the appropriate CUDA architecture build
  (typically sm_89); the R9700 is gfx1201. Verify toolchains and device
  enumeration on the real machine rather than guessing ordinals.
- Adapt the existing Strata verifier's stage-local weights, session
  state, \`set_stage\`, handoff buffers, run and commit ordering.
  Earlier stages must not generate final logits; the last stage owns
  the head and sampling-related final state.
- For v1 use complete, explicit D2H -> host buffer -> H2D copies with
  stream completion checks and bound lengths; only then pass valid
  device-visible memory to \`Verifier::set_stage\`. Never point it at
  a normal socket payload or a foreign runtime allocation.
- Compare a short decode against single-backend reference results.
  Record token IDs and logits/numerical tolerance where applicable;
  do not promise bitwise CUDA/HIP equality.
- Add a fail-injection test for interruption halfway through a stage
  handoff and prevent a partially committed next window.

**Acceptance:** demonstrable single-model, two-stage CUDA + HIP
inference on actual hardware, with stateful decode and comparison logs.
A successful CPU build alone does not meet this milestone.

### M3 — Three-device topology, prefill and restoration (needs GPUs)

- Extend from CUDA -> HIP to CUDA -> HIP -> HIP, preferably reusing
  Strata's existing same-backend AMD split only if integration and
  ownership semantics are validated.
- Ensure each card retains only its owned layers' expert cache and
  layer-specific state; evaluate \`STRATA_STAGE_TRIM=1\` with explicit
  boundaries before making memory-capacity claims.
- Implement chunked prefill above the 16 MiB frame cap, and stage-local
  save/restore as a coordinated operation.
- Only after correctness, add reusable buffers, asynchronous DMA,
  pipeline overlap and performance tuning; benchmark decode and prefill
  separately and report PCIe/OCuLink costs.

**Acceptance:** reproducible three-card results, recovery tests,
measured VRAM/RAM occupancy, and same-workload throughput measurements.

## Existing private HIP development work

There is a separate private development repository,
\`civcode/strata-dev-private\`, branch \`debug/isa-guard-4349\`.
At the initial review, two custom commits after \`9f7d977\` were:

- \`05134f5\`: work-in-progress HIP snapshot readback, copy resources,
  restore admission/mutation gate and tests.
- \`8db7ddc\`: audit/test corrections.

Treat these as **reference work requiring review and rebase**, not as
validated fixes ready to cherry-pick. The audit documents two R9700s,
\`HSA_ENABLE_SDMA=0\`, and unresolved hardware/kernel issues. CUDA was not
validated. Keep private logs, files, credentials and diagnostics out
of this public fork. If porting, put that in a separate focused change
and recheck against the current upstream implementation.

## Commands and validation

Use \`vi\`/\`vim\` for command-line editing, never \`nano\`.
All builds, tests, linting and benchmark execution should run locally
when the environment permits.

\`\`\`bash
cmake -S experimental/heterogeneous -B /tmp/strata-hetero-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build /tmp/strata-hetero-build
ctest --test-dir /tmp/strata-hetero-build --output-on-failure
/tmp/strata-hetero-build/strata_heterogeneous_demo \
  cuda:0:0-3,hip:0:3-7,hip:1:7-10 10 2 6 12345 8
\`\`\`

Optional sanitizer run on a supported Linux toolchain:

\`\`\`bash
cmake -S experimental/heterogeneous -B /tmp/strata-hetero-asan -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer' \
  -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address,undefined'
cmake --build /tmp/strata-hetero-asan
ctest --test-dir /tmp/strata-hetero-asan --output-on-failure
\`\`\`

Don't run GPU-dependent tests without access to the actual GPUs and
model artifacts; mark those checks **not run**, with a reason. Do not
download huge model files merely to validate the CPU transport.

## Repository, CI and reporting rules

- Develop on feature branches. Do not silently change \`main\` or
  upstream \`Niko1221/Strata\`. Do not push or open a PR unless explicitly
  requested; commit only changes that are scoped and reviewed.
- GitHub Actions is a last resort, **never** a test execution default.
  Do not run or dispatch workflows. New/modified workflows must use
  only \`on: workflow_dispatch\` unless the owner explicitly requests
  otherwise. On this feature branch inherited workflows are manual-only;
  upstream/\`main\` may still have automatic triggers.
- Before a substantial upstream proposal, follow \`CONTRIBUTING.md\`:
  discuss architectural scope with the maintainer and label issues/PRs.
- Keep documentation consistent with the actual code. Do not claim
  GPU interoperability, 80 GiB unified VRAM, performance improvements,
  or bug fixes not exercised by the corresponding tests.
- End each coding task with: changed paths; exact commits; tests
  **passed / failed / not run**; known limitations; GPU-only validation
  needed; and the next smallest safe increment.
