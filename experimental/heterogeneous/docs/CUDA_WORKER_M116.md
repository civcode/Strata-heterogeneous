# M1.16 — supervised CUDA model-loading worker (opt-in, GPU execution gated)


> **M2.9 update:** The historical I_APPROVE_* runtime environment-variable
> gate described below was removed. An explicitly launched native worker is
> the runtime opt-in. See [LIVE_GPU_HANDOFF_M29.md](LIVE_GPU_HANDOFF_M29.md)
> for the current manual live GPU test sequence. Earlier milestone notes
> below are retained for historical context.

**Status:** Source-implemented CUDA-only worker executable plus CPU-only
configuration/argv contracts. Buildable only as an explicitly requested
native-engine target; there is no default engine behavior change. Real NVIDIA
loading, inference, expert parity and live teardown **not validated**. The
pre-existing RTX GPU fault must not be reset/rebooted by this handoff.

## Architecture

The existing `WorkerProcessGroup` is the supervisor: `posix_spawn` executes
one image per worker, passes a dedicated AF_UNIX stream on fd 3, and supplies
plan/stage arguments. Its socket and child process are owned by the parent.
Protocol receive timeouts poison a session and the group can SIGKILL/reap
an unresponsive worker. Do not run GPU initialization in the coordinator.

M1.16 adds a **single optional** `cuda_worker_config` path to
`WorkerProcessGroup::launch`. For CUDA children only, the coordinator appends
`--config /absolute/file` to the existing `--worker[...] <plan> ...` argv.
HIP/mock children receive the original eight arguments. No environment
variable or config path is forwarded to HIP children.

The CUDA executable requires:

1. An exact supervisor plan and stage identity, backend `cuda`, descriptor 3
   bound to an AF_UNIX SOCK_STREAM, and an explicit absolute config path.
2. Strict versioned config and **read-only** pack/profile inputs. SHA-256 of
   `<pack_dir>/index.txt` must equal the config's pinned digest; the profile
   must start with STRP/v1. Missing/unknown/duplicate fields, unsafe values,
   noncanonical paths and unmatched stage indices fail *before* CUDA.
3. The checked in `core::kCompiledGeometry` must agree with plan and config
   (`48 layers, hc=4, n_embd=2560, n_vocab=248320` in the standard Q2_0
   engine). The config's `model_fingerprint` must match the coordinator plan.
   **This is caller-attested, not a cryptographic identity of dense.bin,
   embd.bin, or experts.bin.** Index digest does NOT attest those binaries.
4. Even with all checks passing, **no GPU calls** are made unless the operator
   deliberately sets `STRATA_HETERO_CUDA_LIVE=I_APPROVE_CUDA_RUNTIME` in the
   launch environment *after separate NVIDIA recovery approval*. Never set
   the gate as part of an automated test or unattended validation.
5. `CudaStageResources` loads selected dense weights, initializes a fresh
   stage-carved session at position zero, `CudaFileExperts` loads real Q2_0
   experts from its authorized file and verified static cache, and its
   provider is attached before any `CudaVerifierBindings` are handed out.
6. For a nonfinal CUDA stage, a `CudaVerifierDriver` is adapted through
   `DeviceStageExecutor` and served with `serve_worker_stateful`. For a final
   CUDA stage (which must have a preceding activation-producing stage), use
   `CudaFinalVerifierExecutor` and `serve_final_stage_worker`. Worker protocol
   v4 / STFR/v1 are unchanged. Each verifier is **destroyed first**, while
   the expert source and weight/session GPU arenas remain alive.

No new model loader, expert kernel, wire packet or rollback protocol is
invented. CPU fallback from `CudaFileExperts` uses Strata's real
`expert_pool_dispatch_multi`; runtime failure exits the worker (status 75).
The worker still **does not implement model prefill or state restoration**:
`initial_position=0`, zero-prefix commits are refused by the real adapters.
No executable supporting HIP stages is added here, so a CUDA → HIP → HIP
plan is not yet a live mixed-vendor inference pipeline.

## Configuration example

Create a new operator-owned configuration file **outside** `Strata-data`
using `vi` in an authorized workspace/staging directory:

```ini
version=1
pack_dir=/home/chris/workspace/Strata-data/packs/q2_0
profile_path=/home/chris/tmp/strata-heterogeneous/trusted-expert-profile.strp
index_sha256=<exact lowercase sha256sum of q2_0/index.txt>
model_fingerprint=<external trusted plan fingerprint number>
stage_index=0
device_ordinal=0
first_layer=0
end_layer=12
n_layers=48
hc=4
n_embd=2560
n_vocab=248320
max_cells=2048
max_t=8
cpu_workers=4
max_cache_slots=16
io_timeout_ms=30000
free_reserve_bytes=536870912
arena_budget_bytes=0
```

The `<...>` values are placeholders, **not valid configuration values**.
The profile is a REQUIRED trusted STRP/v1 file, not something the worker
fabricates when absent. Specify a real profile only after it has been
reviewed for Q2_0 and its stage. Profile cap and free memory checks prevent
allocating the entire 16 GB GPU from an unrestricted file.

For CPU-only preflight, once the executable has been built:

```bash
strata_heterogeneous_cuda_worker --check-config /home/chris/tmp/strata-heterogeneous/cuda-worker.cfg
```

This verifies schema, index hash, and profile header with **no GPU calls**.
It does not prove numeric parity, expert profile completeness, authenticated
binary content, or a viable amount of VRAM. Native `--worker` mode requires
an inherited supervisor socket and remains disabled by default.

A coordinator using the existing process supervisor opts in explicitly:

```cpp
WorkerProcessGroup group = WorkerProcessGroup::launch(
    plan, absolute_cuda_worker_path, absolute_hip_worker_path,
    false /* shared memory */, absolute_cuda_config_path);
```

The coordinator must still supply a real HIP worker, an exact prefilled
activation input (or the layer-zero embedding path), an approved provenance
source and a supervised request. The CPU mock HIP worker is only a protocol
test and is **not** a production substitute.

## Build and tests on Oberon (no GPU runtime required)

Standalone CPU tests (new tests add two targets to 15 -> **17 CTest tests**):

```bash
cmake -S experimental/heterogeneous -B /home/chris/tmp/strata-heterogeneous/m116/cpu-build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build /home/chris/tmp/strata-heterogeneous/m116/cpu-build --parallel 28
ctest --test-dir /home/chris/tmp/strata-heterogeneous/m116/cpu-build --output-on-failure
```

Optional native executable built **only** inside the full CUDA engine build:

```bash
cmake -S . -B /home/chris/tmp/strata-heterogeneous/m116/native-cuda -G Ninja \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_HETERO_CUDA_WORKER=ON
cmake --build /home/chris/tmp/strata-heterogeneous/m116/native-cuda \
  --target strata_heterogeneous_cuda_worker --parallel 28
```

Prefer reusing the existing Oberon native engine build directory and its
validated CUDA configure flags rather than consuming space with a full new
build. Run a **link check**, not the runtime worker. CUDA-only optional
archive also builds with `STRATA_HETERO_CUDA_VERIFIER=ON`. Any root/engine
integration compile error should be reported with file/line and fixed on a
small follow-up commit.

Do not search or touch either `Strata-dev` tree. Keep the existing untracked
pool probe. Never write to `Strata-data`. Never dispatch GitHub Actions. No
NVIDIA reset or reboot. Keep `--parallel 28` for all Oberon builds.

## Validation limits

CPU tests include SHA-256 known vector via a synthetic index fixture, strict
key/value parsing, profile header checks, wrong/mutated index digest, plan
identity, absolute-path refusal and supervisor argv parsing. Native build
on Oberon remains necessary to validate full CUDA and Strata API/ABI linkage.
A bounded live `CudaStageResources` load and teardown is the next GPU action,
**only after separate explicit recovery approval**. Do not treat mock pass,
compilation or whole-archive link as an inference/teardown result.
