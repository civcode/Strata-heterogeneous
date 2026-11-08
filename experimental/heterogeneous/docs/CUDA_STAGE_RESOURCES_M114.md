# M1.14 — CUDA stage resource ownership (opt-in, native loader integration)

**Status:** Real native `WeightTable::load` and stage-carved `session_init` calls
implemented in a CUDA-only resource owner, with CPU test doubles. The owner
is **not** yet a model-loading worker executable, does not construct a real
expert source/cache, does not prefill, and does not establish GPU inference
parity. No NVIDIA reset, reboot or real GPU calls were run in development.

## Proven pack baseline and scope

The Oberon agent verified the Q2_0 pack under
`/home/chris/workspace/Strata-data/packs/q2_0` against the native
`WeightTable::pool_bytes()` with the provisional layer split:

| Stage | Weight arena bytes |
| --- | ---: |
| CUDA [0,12) | 2,138,211,200 |
| HIP [12,30) | 2,629,851,392 |
| HIP [30,48) | 2,656,252,288 |

All three match the native loader exactly. Stage-global tensors are still
replicated conservatively; the summed 7,424,314,880 bytes is not a combined
GPU VRAM estimate, because session and expert/head/graph budgets are additional.

`iq3_xxs/index.txt` has 302 zero-byte destination rows, including the
selected `output.weight`. The stage planner correctly rejects it. **Do not
pass that pack to M1.14** or invent alternate native tensor representations.

This milestone implements CUDA stage ownership only. Its code is backend-
exclusive and must never link with a HIP worker or the coordinator process.
The provisional [0,12) CUDA stage is suitable for future load tests, but
running `CudaStageResources::load()` actually invokes `cudaSetDevice`, CUDA
memory allocation, and native pack loading; do **not** execute it on Oberon's
failed NVIDIA device until recovery is owner-approved.

## Resource lifecycle

1. An existing, trusted model-geometry loader and plan validator supply
   `ModelGeometry`, matching nonzero `model_fingerprint`, stage index,
   pack directory, `max_cells`, `experts_per_token`, and context limits.
   A supplied fingerprint is **not** automatically computed/verified against
   any model bytes. Mismatched geometry/fingerprint fails before CUDA work.
2. `plan_stage_weights` validates `pack/index.txt`. The owner supplies its
   exact skip set to the real `WeightTable::pool_bytes`; a size mismatch
   fails before touching the CUDA runtime. Selected rows with no loadable
   bytes fail as in M1.13. Neither routine reads the weight binaries.
3. `session_bytes(g, max_cells, k, first_layer, end_layer)` calculates the
   stage-carved session capacity. The owner checks uint64/size_t overflow,
   optional arena budget, `cudaMemGetInfo`, and configurable free-memory
   headroom (default 512 MiB) before allocating two CUDA arenas.
4. The owner calls `WeightTable::load(pack, weight_arena, bytes, ..., &skip)`,
   then `session_init` for the exact layer interval. It calls
   `session_zero(..., R_init=nullptr, stream=nullptr)` and synchronizes the
   device to complete the **fresh zero-state** setup.
5. M1.14 allows `initial_position == 0` only. A zeroed session is **not**
   a restored/prefilled session: claiming position >0 would silently create
   invalid KV/GDN/PLE history. Adding real prefill/restoration is separate.
6. The caller attaches exactly one `std::unique_ptr<CudaStageExperts>`.
   This owner retains the provider, obtains provider-owned `VerifyHits`, and
   exposes a stable CPU expert callback via `make_bindings()`. No bindings
   are returned before the provider is live. A real expert source, caches,
   residency tables and callbacks must be implemented by the worker.
7. The caller constructs either a `CudaVerifierDriver` or
   `CudaFinalVerifierExecutor` from the bindings, then destroys/shuts down
   that verifier **before** shutting down `CudaStageResources`. Borrowed
   references must not escape the owner; the owner does not track live
   adapter instances. It cannot prevent an incorrect caller from freeing
   state out from under a still-running verifier.
8. `shutdown()` synchronizes pending device work, destroys the expert
   provider first, releases QSA/RoPE registration for carved **global**
   ordinals, calls `session_release`, deletes the native heap-owned
   `QsaState[]`, then frees session and weight CUDA arenas. Any failure
   in resource loading invokes the same cleanup; the owner becomes
   permanently poisoned. On device synchronization failure, the worker
   exits fail-stop instead of freeing potentially in-use GPU memory.

The layer-carved RoPE cleanup is intentional: current upstream
`session_release()` loops `qsa_states[0..qsa_alloc)` while carved sessions
initialize at `qsa_ord0`. M1.14 releases owned registrations explicitly
and clears their ownership flag before calling the native release function.
It does not modify the default engine's release behavior.

## Example integration skeleton

```cpp
// CUDA-only EXEC worker process. The real weight metadata/geometry and
// expert service factory must come from the existing, trusted model loader.
using namespace strata::heterogeneous;
CudaStageResources resources(geometry, plan.model_fingerprint);
CudaStageLoadOptions o;
o.pack_dir = "/home/chris/workspace/Strata-data/packs/q2_0";
o.max_cells = max_cells;
o.experts_per_token = experts_per_token;
o.initial_position = 0;  // fresh session; no prefill in this increment
if (!resources.load(plan, stage_index, o, error)) return false;

// Implement and attach an actual expert source and residency service.
// A stub/zero-valued callback is NOT a valid production model worker.
if (!resources.attach_experts(std::move(real_experts), error)) return false;
{
    auto bindings = resources.make_bindings();
    // The chosen verifier adapter must be shut down before resources.
    CudaVerifierDriver driver(bindings); // non-final CUDA stage only
    // ... supervise/serve and driver.shutdown() ...
}
resources.shutdown();
```

M1.14 does not determine the correct expert provider construction or a
model's dense-head/native representation. Those resources and any
prefill/restore must be verified before a real inference claim.

## Validation and handoff

`tests/cuda_stage_resource_stubs` builds the **real** resource-owner source
against explicit CPU implementations of CUDA, `WeightTable`,
`session_init`, `session_release` and RoPE registration. The CTest target
`strata_heterogeneous_cuda_stage_resources_contract_tests` validates the
load order, skip-set reuse, stage-local arena accounting, provider lifetime,
preflight rejects, partial initialization and GPU-call failure cleanup.
No model weights or GPU are used by the double tests.

On Oberon, use only `/home/chris/workspace/Strata-heterogeneous` for the
worktree and `/home/chris/tmp/strata-heterogeneous` for reports. The two
`Strata-dev` directories are unrelated and must not be accessed. The
explicitly authorized `Strata-data/packs/q2_0` index is read-only; for
native compile tests, no weight loading is necessary. Do **not** reset or
reboot NVIDIA for this change. No GitHub Actions execution.

Run CPU CTest GCC, Clang, ASan, UBSan and CUDA-enabled builds with
`cmake --build "$build" --parallel 28`. Compile and link the optional CUDA
archive against native `strata_engine` to catch any ABI mismatches. Return
logs and a concrete fix commit only if needed. Real model loading, GPU
inference and parity remain pending approved GPU recovery.
