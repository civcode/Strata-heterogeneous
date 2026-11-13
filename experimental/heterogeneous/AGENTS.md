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
- a v4 Unix-socket protocol with opt-in memfd-backed activation handoff, stage identity handshake and commit acknowledgments;
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

### M1.11 — Opt-in final-stage result execution (CPU-only)

- FinalPipelineSession and serve_final_stage_worker are **separate** from the
  existing activation-only PipelineSession and worker APIs. They require an
  explicit hello_final/hello_shared_final mode for the final stage only.
- Non-final stages return v4 R|bo|inject activations. The final stage
  instead returns the STFR/v1 head-result packet over its negotiated trusted
  control socket. Never read this packet with receive_frame or treat
  activation bytes as actual logits.
- FinalStageExecutor::run_final_window must return model-produced predictions
  and optionally logits; the mock final stage exists only inside tests.
- Before sending commit_prefix, reject any nonfinite result. Verify every
  committed_prefix ACK including its n_keep. Never publish provisional
  predictions; poison all stages on partial commit or mismatched result.
- Use tests/final_pipeline_tests.cpp and
  [docs/FINAL_PIPELINE_M111.md](docs/FINAL_PIPELINE_M111.md). No real
  CUDA/HIP final driver or numerical parity yet. Zero-prefix restoration
  still needs a real engine/state implementation and GPU validation.

### M1.12 — CUDA final-stage verifier adapter (source implemented)

- `CudaFinalVerifierExecutor` implements the separate `FinalStageExecutor`
  interface. Only the final CUDA stage with a preceding activation producer
  is accepted. It returns actual `Verifier::run` token picks, optional
  `copy_logits` rows and nonfinite head diagnostics in `FinalStageResult`.
- All activation H2D buffers, pinned host staging and CUDA fences are owned
  by the CUDA-only process. The CPU host/memfd view is borrowed for one
  window and must never escape into asynchronous CUDA work.
- The default copies logits to detect unmarked nonfinite results. If logits
  are disabled, require `Verifier::nan_guard_on()` before initialization.
  Never commit a nonfinite provisional result, even if the result gate is
  accidentally bypassed.
- Only positive accepted prefixes can reach `Verifier::commit(n_keep)`.
  Reject zero until a real GPU-tested rollback/restore implementation exists.
- `STRATA_HETERO_CUDA_VERIFIER=ON` compiles this into the existing CUDA-only
  adapter archive. New CPU-double tests cover ID/geometry, pooling, picks,
  logits, async completion, zero, nonfinite and allocation/commit failures.
  Run the *entire* CTest suite on Oberon as well as a native CUDA build/link.
- This increment does **not** provide a model loader, prefill, sampling
  configuration, inference parity, or a runnable CUDA-to-HIP model pipeline.
  The NVIDIA device currently needs owner-approved recovery; do not reset it.

See [docs/CUDA_FINAL_VERIFIER_M112.md](docs/CUDA_FINAL_VERIFIER_M112.md).

### M1.13 — CPU-native index plan for stage-local weights

- `plan_stage_weights()` parses the native 19-column pack index and returns
  a conservative `StageWeightPlan::skip` set for out-of-range `blk.N.*`
  tensors. Every unknown/global tensor stays selected. It does not load a
  model, allocate GPU memory, initialize verifier state or derive identity.
- Before allocating the real CUDA weight arena, compare
  `StageWeightPlan::selected_pool_bytes` against the actual
  `core::WeightTable::pool_bytes(pack, native_bytes, err, &skip)` result.
  Reject mismatches; pass the **same** skip set to `WeightTable::load`.
- Validate every owned layer has indexed tensors, stage-carved session
  storage, expert pool, head/embedding residency, and native loader
  compatibility before any real verifier execution. Do not assume that
  duplicate global tensor loading or a planned arena fits GPU VRAM.
- Run the `strata_heterogeneous_stage_weights_tests` and all existing
  CTest tests on Oberon (GCC, Clang, ASan/UBSan, CUDA-enabled builds).
  Use `cmake --build "$build" --parallel 28`. Compare planner accounting
  to the actual native weight loader with a trusted local pack, if present.
- No GPU reset/reboot, automatic GitHub Actions, or unrelated `Strata-dev`
  directory access. See `docs/STAGE_WEIGHT_PLAN_M113.md`.

**M1.14 completed source-side:** The CUDA resource owner now calls the real
native pack loader and session carve, while the expert source remains an
explicit worker-owned provider to be implemented before runnable inference.

### M1.14 — CUDA stage weight/session resource ownership

- The CUDA-only `CudaStageResources` owns the selected native
  `WeightTable` arena and stage-local `SessionState` arena; it validates
  the real `WeightTable::pool_bytes` against the M1.13 planner **before**
  allocating GPU memory. Keep all global tensors until their ownership
  requirements are explicitly verified.
- `CudaStageLoadOptions.initial_position` must be 0 in this increment:
  `session_zero` is not prompt prefill or KV/indexer restoration.
- A real expert/caching service must implement `CudaStageExperts` and be
  attached before the owner exposes `CudaVerifierBindings`. The owner
  manages its lifetime but **does not construct** native expert pools,
  attested model geometry, residency tables or model fingerprints.
- Always shut down the live verifier before destroying the resource owner.
  On errors, abandon worker state and rebuild in a new process. Explicit
  carved-QSA RoPE release precedes `session_release` and device freeing.
- `src/cuda_stage_resources.cpp` is built into the optional CUDA-only
  adapter archive. CPU contract tests use native code with stub CUDA/
  WeightTable/session functions, but native CUDA engine compile/link,
  real weights and GPU parity still require Oberon verification.
- The verified Q2_0 index is read-only at
  `/home/chris/workspace/Strata-data/packs/q2_0/index.txt`. IQ3_XXS
  is not supported by the native packed loader (zero-byte output rows).
  No unrelated worktrees are allowed.
- See `docs/CUDA_STAGE_RESOURCES_M114.md`. Build with
  `cmake --build "$build" --parallel 28`. No GitHub Actions or reboot.

### M1.15 — Canonical Q2_0 expert source and static CUDA cache

- `CudaFileExperts` implements `CudaStageExperts` using the **native**
  `FileExpertSource`, `ExpertCache`, `ExpertPool` and
  `expert_pool_dispatch_multi`; it loads **real** expert blobs from an
  authorized canonical Q2_0 `experts.bin`.
- A trusted STRP/v1 profile or explicitly ranked unique layer/expert pairs
  is required. Filter to the stage, verify every filled slot's bytes,
  publish exactly matching host/device residency maps. Never invent hits,
  use an empty/sentinel cache, or count a source failure as a zero expert.
- The provider implements the CPU callback contract of the real verifier;
  it exits the isolated worker with status 75 if a native expert source or
  pool fails. No in-process rollback or safe GPU teardown is claimed.
- The verifier MUST be shut down before `CudaStageResources`; the latter
  owns/destroys the expert provider and its CUDA cache and host source.
- Only canonical Q2_0 is supported; IQ3_XXS/native packed experts require
  separate code. The caller supplies authorized pack and profile paths.
- The new CPU-double test compiles the same production source with mock
  CUDA/core APIs, but native CUDA build/link, live expert loading, GPU
  parity and driver-fault teardown remain Oberon validation gates.
- The verified Q2_0 input may be read at
  `/home/chris/workspace/Strata-data/packs/q2_0` only when needed;
  never modify it. Use `cmake --build "$build" --parallel 28`.
  Leave the existing untracked diagnostic probe intact. Do not run GitHub
  Actions or initiate NVIDIA recovery.
- See `docs/CUDA_FILE_EXPERTS_M115.md`.

### M1.16 — Supervised CUDA model-loading worker (source integrated)

- Build only with `STRATA_BUILD_HETERO_CUDA_WORKER=ON` in a Linux,
  CUDA-only full-engine build. It does not alter the default `strata`
  executable. The worker consumes the existing supervisor fd 3, requires
  `--config /absolute/path`, and never loads HIP in the CUDA process.
- The new CPU-only preflight validates strict v1 config, stage identity,
  SHA-256 of the authorized pack's `index.txt`, and STRP/v1 expert profile
  header. The model fingerprint comes from the **trusted caller**, not an
  index hash or a claim of binary attestation. No config/profile discovery.
- **M2.9 supersedes the historical approval gate:** explicit supervised
  CUDA worker launch is the runtime opt-in. There is no separate
  I_APPROVE_CUDA_RUNTIME environment variable check. CPU-only preflight
  never launches native GPU workers.
- The worker creates `CudaStageResources`, then attaches real
  `CudaFileExperts`, then creates a `CudaVerifierDriver` (nonfinal) or
  `CudaFinalVerifierExecutor` (final with prior handoff stage). The verifier
  is destroyed before the worker frees the expert/cache and weight/session
  resources. No prefill or rollback is implemented. No HIP worker is added.
- `WorkerProcessGroup::launch` accepts one optional CUDA config path;
  the mock/hip argv contract remains unchanged when absent.
- Expected **17** CTest targets in standalone experimental builds after
  adding config/argv tests; compile native CUDA executable and link against
  real strata_engine on Oberon using `--parallel 28`, with NO GPU runtime
  startup, NVIDIA reset/reboot or GitHub Actions. Preserve the untracked
  pool probe and avoid `Strata-dev`. See `docs/CUDA_WORKER_M116.md`.

### M1.17 — NativeDense and PLE required for real CUDA windows

- A separate `CudaNativeStage` companion lives between a loaded
  `CudaStageResources` and the verifier. The real M1.17 worker sets
  `require_native_stage=true`; without a successful native load and PLE
  (if owning layer 1), `make_bindings` is refused.
- `CudaNativeStage::open` uses the actual native GGUF uploader and
  `weight_bytes_for`, checks the stage's QKV native alias, validates all
  selected tensor representations, and budgets NativeDense + PLE allocations.
  Never pass empty fake/native tensor outputs to a verifier.
- The layer-1 PLE cannot be silently skipped: it uses the canonical six
  `blk.1.ple_*` weights, with the exact F16 conv1d type check from
  `generate.cpp`, and a reader over the original GGUF table. Direct mode
  starts with no implicit RAM cache, lock or keepalive.
- Worker config v2 adds ordered `gguf_shards` (semicolon-separated paths)
  plus `ple_gguf_path` **only** for stages containing layer 1. v1 remains
  readable for CPU preflight but is rejected by live worker startup.
  Only operator-authorized paths may be used; the pack index SHA-256 is NOT
  proof of all original model bytes.
- Teardown order: verifier graphs first, native PLE and NativeDense second,
  expert provider/session/weight owner last. Violating the order in the owner
  causes a fail-stop worker exit (70); this is NOT live GPU-fault proof.
- M1.17's CPU native-stage contract adds one CTest target: **18 total**
  standalone experimental tests expected; use `--parallel 28` on Oberon.
  Validate native CUDA engine build and whole-archive link, then await
  separate authorization for live Q2_0/PLE numerical validation.
- See `docs/CUDA_NATIVE_PLE_M117.md`. Leave the untracked pool probe
  untouched and do not access either `Strata-dev` tree. No Actions,
  NVIDIA reset or reboot.

### M1.18 — Bounded stage-window admission and fail-closed commits

- Only the opt-in live CUDA worker supplies `WindowLimits` to the v4
  intermediate or STFR final worker server. A CPU-only `WindowAdmission`
  validates monotonic request/window IDs, token IDs against `n_vocab`,
  context positions, max_t, geometry and 16 MiB payload bounds before
  calling either native verifier's `begin_window`.
- Shared-memory inputs are byte-checked for finite IEEE-754 float32 values
  after their checksum; intermediate CUDA stage outputs are checked before
  publishing even a provisional activation frame.
- The final-stage result is checked through existing STFR rules.
  A commit is only acknowledged for an exactly matching positive accepted
  prefix after the verifier's commit/wait reports success; n_keep=0 requires
  real restoration and is refused. All failures poison worker state.
- `STRATA_HETERO_WINDOW_TRACE=1` is opt-in and prints only stage, request,
  window, position, row count and status, not token IDs or model contents.
  CPU-only validation must not start any native GPU worker.
- Existing mock and HIP code paths omit WindowLimits and retain their
  legacy semantics. Expect 20 standalone CTest targets (2 new CPU-only
  tests). Native compilation/link and any live CUDA numerical verification
  remain Oberon approval gates. Build with `--parallel 28`; no Actions,
  GPU reset or reboot. Preserve the untracked probe and stash.
- See `docs/WINDOW_ADMISSION_M118.md`.

### M2.0 — CPU-only HIP startup integration (preparatory slice)

- Adds per-stage supervisor --config path routing and strict HIP worker
  invocation validation. No HIP runtime initialization or model loading.
- Existing CUDA-only config routing and mock worker argv remain unchanged.
- New host-only CTest target brings the standalone suite to **21 tests**.
- See docs/HIP_BOOTSTRAP_M20.md for scope and limits.

### M2.1 — HIP v2 config and fail-closed preflight executable

- Reuse the rigorously bounded CUDA v2 GGUF/pack config reader for a
  separate HIP startup preflight. Require v2 and verify the HIP stage
  (backend-local ordinal, layer interval, geometry, model fingerprint).
- The independent CPU-only preflight executable **always fails closed**
  in supervised worker mode. It must never send ready, load ROCm or
  claim inference. A dedicated CPU config CTest and forked refusal test
  bring the standalone suite to **22 CTest targets**.
- See docs/HIP_PREFLIGHT_M21.md. No HIP device runtime or mixed inference
  has been implemented. Hardware requires separate authorization.

### M2.2 — HIP non-final verifier adapter (native compilation pending)

- One opt-in HIP-only adapter uses the backend-neutral DeviceDriver
  interface, the real core::Verifier and Strata's HIP runtime shim.
- The CUDA-shaped calls map to HIP inside a dedicated HIP build; never
  link this archive into the coordinator or a CUDA executable.
- The standalone host suite now expects **23 CTest targets** including
  a CPU-double lifecycle/fault-injection test compiling the HIP source.
- A real HIP worker, model/session owner, final-stage head, HIP build/link
  validation, mixed GPU inference and parity remain open.
- See docs/HIP_VERIFIER_M22.md.

### M2.3 — HIP final-stage verifier/head adapter (no live run)

- Final HIP stages use a separate `HipFinalVerifierExecutor` returning
  STFR predictions/logits rather than activation frames. Head output,
  nonfinite guard, identity and committed positive prefix are constrained
  just as for the CUDA final adapter, with no cross-runtime pointers.
- **24 standalone CTest targets** expected, including the HIP final-stage
  CPU-double contract. Native HIP compilation and GPU parity still unverified.
- No HIP model-loading worker, state owner or live execution is added yet.
  See docs/HIP_FINAL_M23.md.

### M2.4 — HIP stage resources / native PLE / Q2_0 file experts

- In the opt-in HIP-only engine build, the stage owner uses a carved
  `SessionState`, bounded pack/weight arena, and strictly ordered teardown.
  `HipNativeStage` binds NativeDense and the mandatory layer-1 PLE table.
  `HipFileExperts` owns a canonical Q2_0 file source, resident ExpertCache
  and CPU fallback pool; no fabricated expert outputs are allowed.
- Real sources compile in the HIP-only adapter archive. Each owner has a
  separate CPU-double contract test (27 standalone CTest targets expected).
- This source-level adaptation is **unverified on native HIP**; do not claim
  model loading or inference without successful compilation, correct live
  model assets, expert byte parity and separately approved GPU execution.
- See docs/HIP_STAGE_OWNERS_M24.md.

### M2.5 — Supervised HIP worker integration, unverified hardware

- An independent opt-in HIP-only executable composes HIP stage resources,
  NativeDense/PLE and Q2_0 file experts with non-final/final verifier
  adapters and the M1.18 bounded v4/STFR window admissions.
- HIP startup validates config/identity/compiled Q2_0 geometry first.
  **M2.9 removes the redundant I_APPROVE_HIP_RUNTIME environment gate**;
  an explicitly invoked supervised worker launches its own runtime.
- Never link CUDA into this HIP image, or HIP into CUDA/supervisor.
  Native HIP compile/link remains unverified until Oberon. No GPUs,
  inference, parity or session restoration were executed.
- See docs/HIP_SUPERVISED_WORKER_M25.md.

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
cmake --build /tmp/strata-hetero-build --parallel 28
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
cmake --build /tmp/strata-hetero-asan --parallel 28
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

### M2.6 — CPU-only Q2_0 model/plan validation gate

- `tools/heterogeneous_model_gate.py` uses the repository's GGUF
  header reader and CPU-only HIP preflight executable to check each
  per-stage config against an explicit coordinator plan.
- Requires the canonical 48-layer Q2_0 experts, exact 48x512 STRP
  profile, one original IQ4_NL PLE shard and a complete stage partition.
  Rejects native/IQ expert packs, stage identity mismatches and tampering.
- Local Python test fixture uses synthetic GGUF metadata and sparse
  expert files. Standalone CTest count **28** when Python3 is available.
- This gate does NOT verify the bytes of all GGUF shards or
  dense/embd/experts binaries; no GPU inference authorized or performed.
  See docs/MODEL_PLAN_GATE_M26.md.

### M2.7 — Passive PCI/driver and stage-memory readiness

- `tools/heterogeneous_hardware_readiness.py` reads only bounded PCI
  sysfs and module procfs metadata, stage config files and the Q2_0
  pack index. Optional kernel fault history requires a pre-exported log.
- Optional device mappings are declarations only (`--bind`); missing
  declarations do not block a consistent passive assessment. Never infer
  CUDA/HIP runtime ordinals, available VRAM or GPU health from BDF order,
  DRM numbering, PCI BAR sizes or bound-driver visibility.
- Memory arithmetic covers compact selected weights and canonical Q2_0
  cache slots/residency plus configured safety reserve. Device session
  state, native GGUF/PLE, verifier allocations and runtime overhead
  remain unknown. Do not report a VRAM-fit verdict.
- 19 synthetic fixture unittests are registered as one additional
  CPU-only standalone CTest when Python3 is available (**29** total).
  Oberon real host inventory has not yet been collected by ChatGPT.
- No GPU driver API, runtime, `nvidia-smi`, `rocminfo`, model load,
  reset, reboot, Actions, or unrelated workspace access is permitted.
  See docs/HARDWARE_READINESS_M27.md.

### M2.8 — CPU-only offline ordinal reconciliation

- `tools/heterogeneous_ordinal_reconciliation.py` reads only bounded
  pre-exported M2.7 JSON and separately collected backend-local runtime
  identity evidence; **does not invoke CUDA, HIP, device APIs, utilities,
  sysfs, GPU processes or model files**.
- Reject incomplete, missing, duplicate or contradictory ordinal/BDF
  evidence. Do not infer HIP ordinals from BDF or DRM order, particularly
  with Oberon's additional AMD PCI function (0x13c0).
- No evidence means `BLOCKED` for the optional identity reconciliation
  check **only**; not for the overall integration-test plan. Full agreement
  only means `IDENTITY_CORRELATED_ONLY`, not verified host/boot provenance,
  free VRAM, driver health or proven inference correctness.
- Runtime enumeration can initialize CUDA/HIP. Run it as part of a
  user-initiated GPU test or equivalent scoped agent instruction; that
  instruction is sufficient authorization for its scope. Do not
  independently initiate GPU tests. Reboots, device resets and other
  disruptive recovery remain outside scope unless specifically requested.
- 23 local CPU-only synthetic tests pass. One standalone CTest is added
  (30 total when Python3 is available). No Oberon work or Actions run.
  See docs/ORDINAL_EVIDENCE_M28.md.

### M2.9 — Manually launched GPU test path (current instructions)

- Explicit invocation of the user-requested hardware test (or equivalent
  user-directed agent instruction) authorizes runtime enumeration, allocation,
  native model loading and inference *for that test*. No second authorization
  exchange or I_APPROVE_* environment variable is needed. Do not start live
  tests spontaneously or in unattended CI.
- `strata_heterogeneous_live_smoke` is a host-only coordinator. Without
  `--run` it performs CPU-only plan/config checks; with `--run` it launches
  the separate native CUDA and HIP workers and performs a real cross-backend
  STFR smoke test. It is **never** in GPU-enabled CTest or GitHub Actions.
- Workers emit STRATA_DEVICE with backend-local ordinal, PCI function when
  available, runtime GPU name and free/total VRAM. This is an observability
  log, not an optional PCI mapping prerequisite. Verify stage 0 chooses the
  NVIDIA GPU, and HIP stage 1 and 2 choose the two different R9700s.
- The canonical Q2_0 GGUF/profile/PLE geometry, model fingerprints, stage
  identity, mem bounds, finite activations, and positive-prefix ACK checks
  still protect correctness; do not relax them merely to obtain a PASS.
- Stop on actual device/runtime faults, wrong physical GPU, OOM/reserve,
  model-asset mismatch, nonfinite output, timeout or failed commit. Do not
  reset/reboot GPUs or unload drivers by default. Limit builds to 28 jobs,
  use separate CUDA/HIP toolchains, and never touch the two Strata-dev dirs.
- Test in order: native CPU-only preflight; 1-token socket live test;
  1-token shared-memory live test; two consecutive 1-token windows. No GPU
  tests have been run by ChatGPT in M2.9; Oberon compilation/live output
  remains pending.
- Full commands and test evidence checklist:
  `docs/LIVE_GPU_HANDOFF_M29.md`.


### M2.10 — Unsplit same-engine HIP reference and full-logit parity

- M2.9 real CUDA→HIP→HIP execution PASS: initial socket and shared
  memory predicted token 198; two-window predicted 198,271; 31/31
  host CTest on Oberon, commit `28446b0`. This is not numerical parity.
- Both final verifier adapters now support a single [0,48) stage on
  one backend-local ordinal; stage 0 embeds tokens internally and does
  not read or allocate a handoff buffer. Native geometry/source/VRAM
  constraints, positive prefix ACK and finite checks are unchanged.
- The host-only live smoke launcher accepts `--tokens` and
  `--capture-prefix` and writes committed STFR with full FP32 logits,
  a window input manifest, and a completion marker after successful
  process shutdown. Use fresh prefixes; partial failures lack the marker.
- `tools/heterogeneous_reference_config.py` creates a new 48-layer
  HIP config from trusted stage-0 source. Sets obsolete staged
  arena_budget_bytes to 0, which removes only the artificial stage
  budget, NOT real VRAM checks or free-reserve requirements.
- `tools/heterogeneous_compare_parity.py` requires exact Q2_0 input
  source, tokens, positions, STFR checksums, finite logits and committed
  results. The provisional numerical criterion is exact prediction and
  greedy top-1 agreement plus each row KL(ref||hetero) <=0.001.
  Do not silently waive deviations due to HIP Q8 hyper-connection.
- Run M2.10 only under an explicit user-initiated GPU test or equivalent
  scoped agent instruction. The user has already requested this
  milestone; no secondary environment approval ritual. Never run Actions,
  reset/reboot a GPU, unload a driver, or touch either Strata-dev dir.
  Builds max 28 CPU jobs; CUDA/HIP executables must remain separate.
- **M2.10 execution and numerical parity remain UNTESTED until the
  Oberon native reference and heterogeneous captures complete.**
  Full commands: `docs/NUMERICAL_PARITY_M210.md`.


### M2.11 — Diagnostic post-layer tap capture following M2.10 parity failure

- M2.10 GPU runs completed all commits, with matching predictions/top-1
  at both positions. KL ref||het = 0.003376 at position 0 and 0.039019
  at position 1, exceeding the declared 0.001 threshold. A diagnostic
  STRATA_HC_Q8=0 rerun also failed (0.005003 and 0.037718). Maintain
  **numerical parity FAIL**; do not broaden windows or raise tolerance.
- M2.11 reuses native Verifier::set_taps/gather_taps_host for every
  post-layer **HC-mean completed residual**, including pending GR write,
  for layers 0..46. These are not full R|bo|inject handoffs, so an
  observed divergence is a localization clue, not a proof of cause.
- The host-only live smoke `--layer-taps` requires `--capture-prefix`;
  without `--run` it remains CPU-only. It passes one explicit per-run
  output prefix to each exec-isolated native worker. Worker STLT/v1
  records are bounded and exclusive; corrupt/nonfinite recordings
  fail closed, and the host completion marker is published only after
  all worker commits and shutdown succeed.
- The CPU-only `tools/heterogeneous_layer_divergence.py` compares
  completed reference/heterogeneous runs and cross-checks every
  stage/pos/window/model/tap checksum, reporting per-layer relative L2,
  RMSE, max abs delta and first layer above a **diagnostic** threshold.
  That threshold does not replace the M2.10 logit KL criterion.
- Run on Oberon under the existing user-requested hardware test scope.
  No GitHub Actions, no reset/reboot/driver unload, build <=28 jobs,
  separate CUDA/HIP workers, never touch either Strata-dev directory.
  Do not combine STRATA_TAP_FILE with M2.11 --layer-taps.
- **No M2.11 live GPU tap measurements exist yet.** See exact commands
  and stop conditions: `docs/LAYER_DIVERGENCE_M211.md`.


### M2.12 — First-operation isolation of layer-zero arithmetic

- M2.11 real hardware result: layer-zero post-write HC-mean residual
  differs by relative-L2 0.011669/0.012537 for positions 0/1.
  M2.10 numerical parity remains FAIL at head-logit KL
  0.003376/0.039019; STRATA_HC_Q8=0 did not resolve it.
- Use the *same HIP0 R9700* for native unsplit reference hip:0:0-48
  and stage 0 of all-HIP split hip:0:0-12,hip:1:12-48;
  then compare both with original mixed CUDA:0:0-12/HIP0/HIP1.
- The opt-in core Verifier `set_layer_zero_arithmetic_probe` snapshots
  layer-zero FP32 native embeddings, first GR HC-mixed activations,
  raw Q8_1 bytes used by the native GDN QKV MMVQ and projected QKV
  FP32 values, before the first-layer GDN/FFN/MoE remainder.
  Single-token windows only; no extra kernel/VRAM cost when disabled.
- All isolated workers support the fail-closed
  `STRATA_HETERO_L0_PROBE_PREFIX` diagnostic, set by the host-only
  `--layer0-probe --capture-prefix ... --run` explicit manual launcher.
  Unique STL0/v1 files are bounded, checksummed, full input/plan/window
  identified and emitted before prefix commit; the usual complete-run
  marker requires successful worker shutdown.
- `tools/heterogeneous_hip_control.py` creates new all-HIP stage
  configs without touching originals, preserves canonical model
  sources and VRAM reserve, and removes obsolete stage-only budget caps.
  `tools/heterogeneous_layer0_arithmetic.py` refuses incomplete or
  mismatched sources, parses exact probe values and compares three
  completed runs to localize the first divergent arithmetic component.
- 15 CPU-only M2.12 test cases are registered with CTest. Run them
  locally before live GPUs. **Live native M2.12 correctness is
  unverified** until Oberon executes the dedicated handoff.
- No GitHub Actions, GPU reset/reboot/driver unload, or Strata-dev
  access. Builds <=28 jobs; separate CUDA and HIP builds. Preserve
  M2.10 KL<=0.001 and all model, finite and prefix-commit invariants.
  Full handoff: `docs/LAYER0_ARITHMETIC_M212.md`.
