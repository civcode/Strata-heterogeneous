# M1.15 — Canonical Q2_0 expert source and static residency provider

**Status:** New CUDA-only source implementation with GPU-free contract tests.
Native CUDA/Strata compilation and linking have **not** been run in ChatGPT's
container. Real model-weight loading, device teardown and numerical inference
have **not** been run; Oberon must validate the native build first.

## Why this provider exists

`Verifier::init()` rejects a stage with missing `VerifyHits::d_res`,
`cache_base` or `blob` even when most experts are CPU-computed. The M1.14
`CudaStageResources` accepts a generic `CudaStageExperts` implementation,
but it deliberately does not supply a real expert tier. `CudaFileExperts`
provides that tier for the **canonical Q2_0 experts.bin pack only**.
It uses the *existing* Strata implementations, not a new expert kernel:

1. `kernels::cpu::expert_layout_load` checks the model-wide expert layout;
   native/IQ layouts are rejected. The geometry must match the compiled
   canonical expert width and expert count (currently 2560 and 512).
2. `core::FileExpertSource::open` maps the actual `experts.bin` and checks
   the complete file length for every layer/expert.
3. A trusted STRP/v1 profile (using `read_expert_profile`) **or** explicit
   ranked `(layer, expert)` pairs chooses real expert blobs. The provider
   validates duplicate, out-of-range and empty stage-local selections.
   The profile list is filtered to the stage's layer range, then optionally
   truncated by `max_cache_slots`. There is no invented fallback profile.
4. `core::ExpertCache` allocates real slots, `admit`s each selected expert,
   copies its actual file blob with `fill_slot_blocking`, and **reads it back**
   through `verify_slot`. No slot is marked resident before successful copy
   and verification. The provider checks GPU-free memory against cache bytes,
   residency-table bytes and `free_reserve_bytes` (default 512 MiB).
5. A full model-sized host residency table (all other slots -1) is copied to
   the CUDA device and synchronized. `VerifyHits` exports precisely these
   arrays, the cache base, blob size and live slot count. No HIP pointers or
   cross-process GPU mappings are exposed.
6. The real `kernels::cpu::ExpertPool` and
   `core::expert_pool_dispatch_multi` compute nonresident routes. The native
   dispatcher writes zero for resident routes; the verifier's GPU planner
   adds their actual cached expert outputs. `set_tail_skip_rank(0)` ensures
   exact, non-approximate routing in this independently supervised worker.

`CudaFileExperts::run()` receives the native verifier callback's explicit
`layer`, and resets the native dispatch cursor for each token group/window.
Every layer, route ID, row count and `k==10` is checked before delegation.
**If the source or pool fails, the callback exits the worker with status 75**;
returning would allow the verifier's GPU doorbell to accept stale, partial or
fabricated expert rows. The supervisor must reap/restart the process; there
is no in-process rollback. A deliberate process exit bypasses destructors
because a pending GPU kernel might still be blocked. Its suitability with an
unresponsive GPU is NOT validated by CPU doubles.

## Integration with M1.14

Use this only from an exec-isolated, CUDA-only supervised worker with a
prevalidated model geometry, fingerprint and selected Q2_0 weight pack.
The pack and profile paths MUST come from authorized caller configuration,
not a directory-wide search. Construct and attach the provider after the
weight/session owner loads and before constructing the verifier:

```cpp
CudaStageResources resources(geometry, plan.model_fingerprint);
if (!resources.load(plan, stage_index, load_options, error)) return false;

CudaFileExpertOptions eo;
eo.pack_dir = load_options.pack_dir;
eo.profile_path = trusted_profile_path;  // STRP/v1; supply a real file
// Alternatively set eo.ranked_pairs explicitly (but NOT both).
eo.max_cache_slots = requested_stage_slots; // 0: all stage pairs
 eo.cpu_workers = 4;                        // tune later, <= 28

auto experts = std::make_unique<CudaFileExperts>();
if (!experts->open(plan, stage_index, geometry, eo, error) ||
    !resources.attach_experts(std::move(experts), error)) return false;

{
    const auto bindings = resources.make_bindings();
    CudaVerifierDriver verifier(bindings); // CUDA nonfinal only
    // supervise and serve the activation-only CUDA stage
    // verifier.shutdown() must complete before resources.shutdown().
}
resources.shutdown();
```

This is intentionally not a new worker main or loader provenance API. Do not
start a GPU runtime in the coordinator. Do not mix CUDA and HIP in one process.
The actual CUDA stage on Oberon is provisionally `[0,12)`; the later HIP
stages need a separately compiled HIP implementation, which is **not** in M1.15.
Global-tensor duplication is retained until owner requirements are proven.

## Tests and limitations

The CPU-only
`strata_heterogeneous_cuda_file_experts_contract_tests` target compiles
**the real provider `.cpp`** against fake CUDA/Strata API implementations.
Tests cover profile filtering, exact host/device residency, resident-zero and
CPU-miss dispatch, repeated windows, partial source/cache/pool failures,
allocation, upload, synchronization, memory reserve and a forked worker that
must exit 75 on invalid routes, bad layers and failed/throwing pool callbacks.
They do not execute real CUDA kernels or native expert arithmetic.

Other remaining requirements:

- Verify real native CUDA compilation, whole-archive linkage and full CTest
  with GCC, Clang, ASan, UBSan and CUDA-enabled builds on Oberon.
- Run a real `FileExpertSource` open/cache fill and verifier after **approved**
  NVIDIA recovery. No reset/reboot is authorized by this handoff.
- Check numerical parity against an unsplit CUDA reference with exactly the
  same Q2_0 model, sampling settings and position/state. GPU live shutdown,
  hangs, driver faults and poison/fail-stop remain unverified.
- Provide the real model-loading worker executable and an actual authorized
  profile. `Strata-data/packs/q2_0` is an authorized **read-only** input;
  `Strata-data/packs/iq3_xxs` is unsupported here.

## Oberon handoff

Checkout `feature/heterogeneous-cuda-file-experts` from the M1.14
`29761727c2ce59cb2596beea4622efa7de5d52db` baseline. Use only
`/home/chris/workspace/Strata-heterogeneous` and stage logs under
`/home/chris/tmp/strata-heterogeneous`; leave the existing untracked pool
probe untouched. Do not inspect or modify either `Strata-dev` directory.

Use `cmake --build "$build" --parallel 28`. Run the complete standalone
suite; it should have **15 CTest targets**, including this new expert
contract test. Also compile/link
`STRATA_HETERO_CUDA_VERIFIER=ON` with the real `strata_engine` archive.
No GitHub Actions, NVIDIA reset or reboot. Return source/API defects,
exact commit SHA and build/test logs, explicitly distinguishing CPU-double
and native tests from GPU inference.
