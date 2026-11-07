# Coding agent instructions: heterogeneous Strata inference

These instructions apply to this directory. Also follow the repository-root
[AGENTS.md](../../AGENTS.md) and [CONTRIBUTING.md](../../CONTRIBUTING.md).
This is an experimental, opt-in research feature; do not change the behavior
of existing CUDA-only or HIP-only inference.

## Local project isolation on Oberon

Run project commands only in `/home/chris/workspace/Strata-heterogeneous`.
Stage project patches and logs in `/home/chris/tmp/strata-heterogeneous`.
**Ignore completely** `/home/chris/workspace/Strata-dev` and
`/home/chris/tmp/strata-dev`: do not search, inspect, modify, build,
test, or use their files for this project. Do not run parent-directory-wide
searches. The root [AGENTS.md](../../AGENTS.md) establishes these boundaries
for the entire fork.

## Mission and current state

Implement **one model with contiguous layer ranges executed by independent
CUDA and HIP workers**, eventually on an NVIDIA RTX 4070 Ti SUPER (16 GiB)
and two AMD Radeon AI PRO R9700 cards (32 GiB each). Use the existing
Strata layer-split algorithm, not tensor parallelism. The first CUDA/HIP
boundary is the priority; do not claim that 80 GiB is unified VRAM.

The code in this directory is presently a **CPU-only Linux host prototype**.
It includes:

- a validated stage plan with backend-local ordinals;
- a v3 Unix-socket protocol with opt-in memfd-backed activation handoff, stage identity handshake and commit acknowledgments;
- separately compiled, backend-tagged CPU mock worker executables launched with posix_spawn/exec by the coordinator;
- exact-shape, integrity, timeout, failure-propagation, and subprocess tests.

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
  `Verifier::set_stage`, `set_handoff_rows`,
  `handoff_floats`, stage-to-stage execution and commit semantics.
- [../../include/strata/prefill/prefill.hpp](../../include/strata/prefill/prefill.hpp):
  later prefill-stage handoff and chunk semantics.
- [../../docs/MULTI_GPU.md](../../docs/MULTI_GPU.md) and
  [../../docs/AMD_HIP.md](../../docs/AMD_HIP.md): supported same-backend split.
- [../../CMakeLists.txt](../../CMakeLists.txt):
  `STRATA_ENABLE_CUDA` and `STRATA_ENABLE_HIP` are mutually exclusive
  in an ordinary engine build. Do not bypass that guard to link both
  runtimes together; compile separate worker executables instead.

Strata's verifier handoff for **each row** is exactly three contiguous
float32 arrays, in order:

1. `R`: `hc * n_embd` floats.
2. `bo`: `n_embd` floats.
3. `inject`: `hc` floats.

The prototype's `Geometry::floats_per_row()` mirrors the upstream
`Verifier::handoff_floats()` formula. Handoff payloads are **not**
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
   `[first_layer, end_layer)` interval; stages cover every layer exactly
   once. Do not use tensor-parallel per-layer splitting for this work.
4. **Explicit identity:** verify model/weights, quantization-relevant
   configuration, geometry, stage plan, request ID, window ID and stage
   index before accepting data. The current model fingerprint is
   caller-provided; deriving it from the real loaded model is unfinished.
5. **Fail closed:** no next-token result, checkpoint or updated session
   may be published after a failed or incomplete stage. Distinguish
   pre-mutation resource refusal from failure after state mutation.
   Abort or rebuild poisoned stage state rather than silently proceeding.
6. **Preserve ABI:** the current wire protocol is **version 4**.
   The 56-byte little-endian header remains, with a bounded
   `window_spec` frame carrying verifier token IDs and pos0 before each
   activation, plus `commit_prefix` and `committed_prefix` frames
   carrying the accepted-token count chosen *after* execution. v3 workers
   are incompatible. The v3-style socket-data and optional memfd transport
   modes remain available under v4. Keep frame and acknowledgement checks.
   FNV-1a is not an authenticator; use only with trusted local processes.
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

### M1 — Host-only stateful worker protocol (implemented)

- `StageExecutor` now exposes `prepare(plan, stage_index)`, provisional
  `run_window`, `commit_window`, `abandon_window`, and `shutdown`.
  `MockStageExecutor` is a CPU test double, **not** a real stateful verifier.
- `PipelineSession::initialize()` validates worker-declared identity;
  `run_window()` accepts results only after each stage acknowledges commit.
  The previous `run_pipeline` entrypoint is gone in protocol v2.
- Frame I/O has bounded deadlines. Faults poison the session and shut
  down communication with workers; **partial commits are not rolled back**.
- The handshake checks declared backend/ordinal/layer interval, plan and
  geometry. Runtime hardware attestation, capability negotiation, and
  a fingerprint derived from loaded model bytes are **still missing**.
- Cancellation, worker restart/recovery, state save/restore, and real GPU
  backends remain future work. Extend negative tests before those changes.
- Keep the legacy CUDA/HIP build guard intact and use separate binaries.

**Evidence:** local CPU CTest and subprocess runs are the acceptance checks.
No CUDA/HIP kernel, device initialization, or model-layer execution has been
implemented or validated in this milestone.

### M1.5 — Independent mock executables and child-process supervision (implemented)

- The experimental library exports `WorkerProcessGroup` in
  `include/strata_heterogeneous/process.hpp`. It owns child PIDs and
  coordinator sockets and starts worker binaries with `posix_spawn`/`exec`.
- The CPU demo now launches separate `strata_heterogeneous_mock_cuda_worker`
  and `strata_heterogeneous_mock_hip_worker` executables, built from the
  same test source with different backend markers. Neither links GPU APIs.
- `WorkerProcessGroup::finish` performs bounded graceful shutdown;
  `terminate` kills and reaps children on failures and during destruction.
- `strata_heterogeneous_process_tests` covers three-stage execution,
  backend marker mismatch, missing executables, nonzero exits, forced
  timeouts and cleanup. Keep those tests passing.
- Real CUDA/HIP worker images, hardware identity attestation, secure GPU
  buffer ownership, model loading and actual inference remain **unimplemented**.

**Next:** use the same `WorkerProcessGroup` launch contract for CUDA/HIP
worker binaries; do not bypass the backend separation or assume these CPU
mock tests validate the device runtime.

### M1.6 — Reusable shared-memory activation slot (implemented, CPU-only)

- `SharedActivation` creates a sealed 16 MiB Linux memfd and uses
  SCM_RIGHTS to transfer descriptor access to each exec-isolated worker.
- Opt-in `PipelineSession(..., shared_memory=true)` uses a distinct
  `hello_shared` handshake. All stages reuse one mapping, sequentially;
  ordinary socket-data mode remains the default.
- Workers verify mapped bytes against the `shared_data` digest before
  execution; the coordinator verifies returned bytes before continuing.
  A mismatch poisons/stops the session and prevents result publication.
- Shared memory no longer requires a full CPU vector copy in each worker:
  a `run_staged_window` override can operate on `ActivationView` directly.
  The default implementation still supports legacy `std::vector<float>`
  executors via a safe validated fallback. This is **not GPU zero-copy**
  and does not justify a performance claim. GPU adapters must use their
  own device allocations and completed D2H/H2D transfers.
- Run the new `strata_heterogeneous_shared_tests` and
  `strata_heterogeneous_shared_smoke` locally, plus existing CTest.
  Keep descriptor/failure/timeout tests passing on any protocol change.
- One 16 MiB slot and one in-flight window is supported. Do not claim
  chunked prefill, asynchronous overlap, recovery or worker restart.

**Next:** define and implement explicit GPU-side transfer/StageExecutor
adapters, preserving process separation and stage-local ownership.

### M1.7 — Borrowed host activation view and in-place execution (implemented)

- `ActivationView` validates geometry, row counts and the 16 MiB cap.
  `HandoffRow` gives exact `R`, `bo`, and `inject` byte ranges. Use
  `memcpy`-based float accessors; do not cast mmap storage to `float*`
  under C++17.
- Shared-memory workers dispatch to
  `StageExecutor::run_staged_window(metadata, view, error)`. They no
  longer require intermediate full-size float vectors for the direct
  path. Preserve the default validated vector fallback for executors
  that only implement `run_window`.
- The view is a temporary borrowed **host** mapping, never CUDA/HIP
  device storage. Real backends must copy to backend-owned buffers,
  synchronize execution and D2H copies before reporting success.
- A failed staged execution may have mutated the shared activation
  bytes, but must not have committed durable stage state. Poison all
  participants and never publish a result after failure. This is
  fail-stop, **not rollback**.
- Run `strata_heterogeneous_activation_tests` and all existing CTest
  targets (GCC, Clang, ASan and UBSan). Regression coverage includes
  malformed layouts, overflow, exact multirow partition, corrupted
  output rejection, direct staged dispatch, and injected post-write
  worker failures.
- Protocol v3 and the default socket-data mode are unchanged. Do not
  alter wire frames for this feature.

**Next:** implement a real CUDA stage adapter against the existing Strata
verifier (including stage-local weights, KV, run/commit and explicit GPU
events); validate CUDA-only stages first, then add the HIP process and
cross-runtime transfer checks on Oberon.


### M1.8 — Device transfer lifecycle (implemented, simulated only)

- `DeviceDriver` is a runtime-neutral asynchronous device interface:
  prepare, enqueue/wait H2D, enqueue/wait stage execution, enqueue/wait D2H,
  enqueue/wait commit, abort, and shutdown.
- `DeviceStageExecutor` provides strict stage/plan/window/view validation.
  Both socket and shared-memory workers can use the adapter. Each wait must
  prove completion before publishing a provisional result or commit ACK.
- Every driver failure or exception poisons the stage, aborts the in-flight
  work and prevents another window. A partially written host activation
  or partially committed stage cannot be recovered by assuming rollback.
- **Driver implementers:** put finite deadlines on GPU waits, ensure abort
  drains all pending DMA before borrowed host pointers can be unmapped,
  or terminate the worker if the GPU cannot be safely drained. Do not put
  CUDA/HIP runtime types into the host library or combine both runtimes
  in one process.
- The `strata_heterogeneous_device_tests` target runs CPU-only queued fake
  device operations, numerical parity of mocked stage ranges, injected
  faults at every transfer/commit boundary, and real subprocess poisoning
  in both transport modes.
- **Real integration blocker:** actual `Verifier::run` requires tokens,
  pos0, pool callbacks and head outputs; `commit(n_keep)` needs accepted
  prefix length. These are absent from v3 `Frame`. Before implementing
  real GPU execution, design versioned control messages and a model-loaded
  identity attestation, with negative tests. The fake deliberately does not
  claim to perform GPU inference or validate actual model state.

**Next:** add the real verifier request/commit metadata and model identity
contract, then implement stage-owned CUDA and HIP binaries and validate on
Oberon. Keep all normal inference defaults untouched.

### M1.9 — Verifier token metadata and deferred prefix commit (CPU-only)

- Protocol **v4** carries `WindowSpec {pos0, token_ids}` in a
  `window_spec` control frame before each activation handoff.
  `WindowSpec::validate` checks lengths, nonnegative tokens/positions,
  a conservative 8192-row protocol cap, and position overflow.
- A caller uses `PipelineSession::execute_window(frame,spec)` to obtain
  provisional data. Only **after** computing acceptance may it call
  `PipelineSession::commit_pending(n_keep)` with (0 <= n_keep <= T).
  The coordinator requires a matching `committed_prefix` from every stage
  before returning the committed result. No rollback is provided.
- `StageExecutor::begin_window` and `commit_prefix` default to rejecting
  unsupported real-verifier work. `DeviceDriver::begin_window` and
  `enqueue_commit_prefix` must be implemented in real backend adapters;
  never silently treat a partial commit as a full-window commit.
- The legacy `run_window` call remains for CPU mocks only.
  `tests/window_tests.cpp` runs wire, subprocess, CUDA-labeled/HIP-labeled,
  both IPC transports, driver metadata and fault-injection checks.
- Read [docs/VERIFIER_WINDOW_V4.md](docs/VERIFIER_WINDOW_V4.md) before
  integrating. This is **not** real inference: final logits, expert pool
  callbacks, model weights, device buffers, GDN, KV, PLE and prefill remain.
- The native Oberon coding agent must work only in
  `/home/chris/workspace/Strata-heterogeneous`, use
  `/home/chris/tmp/strata-heterogeneous` for handoff artifacts,
  and completely avoid both unrelated `Strata-dev` paths.

**Next:** verify the real `Verifier::run` and `commit(n_keep)` interfaces
and implement a concrete CUDA-only stage adapter before adding HIP stages.
GPU validation belongs on Oberon.

### M1.10 — Independent final-stage result codec (implemented, CPU-only)

- `FinalStageResult` is a separate **v1 result packet** with its own
  64-byte header, magic `STFR`, checksum and a bounded 16 MiB payload.
  This does **not** change the existing v4 activation/commit protocol.
- A result carries one predicted token per row, optional row-major logits,
  sorted nonfinite-row markers, and exact stage/request/window/plan identity.
  The field ranges and overall size must be checked before allocating
  untrusted lengths; checksums detect corruption, not malicious peers.
- `FinalResultGate` explicitly distinguishes provisional from committed
  results. Do not call `mark_committed` before ALL stages have acknowledged
  a matching commit; poison on any failure. No distributed rollback exists.
- The codec and standalone AF_UNIX socket helpers are **not integrated**
  with `PipelineSession` or an engine worker. No real head, logits
  sampling, device transfer, or mixed-vendor parity is claimed.
- Run `strata_heterogeneous_final_result_tests` plus every prior CTest
  target. See [docs/FINAL_RESULT_V1.md](docs/FINAL_RESULT_V1.md) before
  designing the opt-in v5 final-stage handshake.

**Next:** extend the worker/coordinator state machine to transport a
real verifier head result from the final stage instead of inventing a
bogus activation handoff, and validate CUDA hardware when available.

### M2 — First actual CUDA -> HIP inference (needs GPU validation)

- Build a CUDA stage worker and a HIP stage worker separately. The
  RTX 4070 Ti SUPER requires the appropriate CUDA architecture build
  (typically sm_89); the R9700 is gfx1201. Verify toolchains and device
  enumeration on the real machine rather than guessing ordinals.
- Adapt the existing Strata verifier's stage-local weights, session
  state, `set_stage`, handoff buffers, run and commit ordering.
  Earlier stages must not generate final logits; the last stage owns
  the head and sampling-related final state.
- For v1 use complete, explicit D2H -> host buffer -> H2D copies with
  stream completion checks and bound lengths; only then pass valid
  device-visible memory to `Verifier::set_stage`. Never point it at
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
  layer-specific state; evaluate `STRATA_STAGE_TRIM=1` with explicit
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
`civcode/strata-dev-private`, branch `debug/isa-guard-4349`.
At the initial review, two custom commits after `9f7d977` were:

- `05134f5`: work-in-progress HIP snapshot readback, copy resources,
  restore admission/mutation gate and tests.
- `8db7ddc`: audit/test corrections.

Treat these as **reference work requiring review and rebase**, not as
validated fixes ready to cherry-pick. The audit documents two R9700s,
`HSA_ENABLE_SDMA=0`, and unresolved hardware/kernel issues. CUDA was not
validated. Keep private logs, files, credentials and diagnostics out
of this public fork. If porting, put that in a separate focused change
and recheck against the current upstream implementation.

## Commands and validation

Use `vi`/`vim` for command-line editing, never `nano`.
All builds, tests, linting and benchmark execution should run locally
when the environment permits.

```bash
cmake -S experimental/heterogeneous -B /tmp/strata-hetero-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build /tmp/strata-hetero-build
ctest --test-dir /tmp/strata-hetero-build --output-on-failure
/tmp/strata-hetero-build/strata_heterogeneous_demo \
  cuda:0:0-3,hip:0:3-7,hip:1:7-10 10 2 6 12345 8
```

Optional sanitizer run on a supported Linux toolchain:

```bash
cmake -S experimental/heterogeneous -B /tmp/strata-hetero-asan -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer' \
  -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address,undefined'
cmake --build /tmp/strata-hetero-asan
ctest --test-dir /tmp/strata-hetero-asan --output-on-failure
```

Don't run GPU-dependent tests without access to the actual GPUs and
model artifacts; mark those checks **not run**, with a reason. Do not
download huge model files merely to validate the CPU transport.

## Repository, CI and reporting rules

- Develop on feature branches. Do not silently change `main` or
  upstream `Niko1221/Strata`. Do not push or open a PR unless explicitly
  requested; commit only changes that are scoped and reviewed.
- GitHub Actions is a last resort, **never** a test execution default.
  Do not run or dispatch workflows. New/modified workflows must use
  only `on: workflow_dispatch` unless the owner explicitly requests
  otherwise. On this feature branch inherited workflows are manual-only;
  upstream/`main` may still have automatic triggers.
- Before a substantial upstream proposal, follow `CONTRIBUTING.md`:
  discuss architectural scope with the maintainer and label issues/PRs.
- Keep documentation consistent with the actual code. Do not claim
  GPU interoperability, 80 GiB unified VRAM, performance improvements,
  or bug fixes not exercised by the corresponding tests.
- End each coding task with: changed paths; exact commits; tests
  **passed / failed / not run**; known limitations; GPU-only validation
  needed; and the next smallest safe increment.
