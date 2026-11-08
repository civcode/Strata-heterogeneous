# M1.13 — Stage-local native pack selection (CPU-only)

Status: **implemented and locally CPU-tested**. This is not a model-loading
CUDA worker, a weight copy, or a GPU inference path. It is the first
source-backed input to that loader.

## Source evidence and selection rule

The real Strata file `tools/pack_index.py` writes a header
`# align A pool P tensors N`, followed by exactly 19 fields per weight row.
`src/core/weights.cpp` implements `core::WeightTable::pool_bytes` and
`WeightTable::load`, each accepting an optional set of skipped *tensor names*.
When a skip set is supplied, the native loader compacts the retained rows
in index order, rounding each destination span to the header alignment.

`plan_stage_weights(plan, index, pack_dir + "/index.txt", out, err)` validates
this index format and returns an exact skip set for all `blk.N.*` tensors
outside the stage's assigned contiguous layer interval. It never drops a
global or unfamiliar name automatically. The chosen stage must have at
least one indexed block tensor for every layer it owns.

The result includes the original pack pool size, the compact selected
pool size, count of selected tensors, and count of stage-layer tensors.
Malformed rows, duplicate names, noncanonical layer names, missing owned
layers, invalid offsets, overflow, header/count mismatch and an unopenable
file fail closed; the caller's output object is unchanged on failure.

## Native CUDA loader integration

Use this component before allocating the CUDA weight arena:

```cpp
using namespace strata::heterogeneous;
StageWeightPlan selected;
if (!plan_stage_weights(plan, stage_index,
                        pack_dir + "/index.txt", selected, error))
    return false;

uint64_t native_bytes = 0;
if (!strata::core::WeightTable::pool_bytes(pack_dir, native_bytes,
                                          error, &selected.skip))
    return false;
if (native_bytes != selected.selected_pool_bytes) {
    error = "native weight loader and stage planner disagree about arena size";
    return false;
}
// Only after matching sizes and checking the stage-local VRAM budget:
// cudaSetDevice(the_backend_local_cuda_ordinal);
// cudaMalloc(&device_arena, native_bytes);
// weight_table.load(pack_dir, device_arena, native_bytes,
//                   error, &selected.skip);
```

The same `selected.skip` **must** be passed to both native functions.
The resulting compact arena size includes only dense/pack weight rows,
not session buffers, KV/GDN/PLE state, resident expert cache, head work
buffers, pinned host memory, or CUDA graph allocations. Do not interpret
it as a total VRAM budget or an identity fingerprint.

The next CUDA-only resource-owner increment must own and release the weight
arena, `core::WeightTable`, a `session_bytes/session_init`-carved stage state,
and the real expert pool / resident-hit resources. It must follow the engine's
native model-geometry/quantization restrictions, initialize per-sequence
state, and keep everything alive through the corresponding CUDA verifier
executor's shutdown. Unknown global weights and native projection handling
need additional explicit verification before any further selective skipping.
This increment does **not** create that owner or start worker processes.

## Tests and Oberon handoff

The new `strata_heterogeneous_stage_weights_tests` CTest target creates
temporary 19-column pack indexes; tests check stage partitioning across
CUDA/HIP labels, global tensor retention, alignment math, errors,
conservative layer coverage and unchanged output on invalid input.
The planner also compiled and passed its 113 assertions against the
archived real host `Plan` implementation, in GCC, Clang, and sanitizer builds.

On Oberon, start from the exact branch commit. Build with
`cmake --build "$build" --parallel 28` and run the full experimental
CTest suite under GCC, Clang, ASan, UBSan and CUDA-enabled builds. A trusted
native pack, if available, can confirm the exact computed pool size against
`WeightTable::pool_bytes`. Do not download a large model to test this parser.
Return the integration commit SHA, logs and any differences.

Workspace: `/home/chris/workspace/Strata-heterogeneous`.
Log and patch staging: `/home/chris/tmp/strata-heterogeneous`.
Do not access `/home/chris/workspace/Strata-dev` or
`/home/chris/tmp/strata-dev`. Do not run GitHub Actions.
No GPU reset or reboot is requested. NVIDIA parity remains blocked by the
existing unrecovered Xid 79/154 fault.
