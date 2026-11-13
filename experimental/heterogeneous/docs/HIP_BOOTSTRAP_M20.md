# M2.0 — HIP worker startup contract (CPU-only)

**Base:** M1.18 at 6b3b6be3, plus manual upstream gate 17809447.
**Goal:** prepare independent HIP worker execution without initializing ROCm.

- `WorkerProcessGroup::launch` retains its legacy optional CUDA-only
  config argument and supports an additional per-stage config list.
- If per-stage paths are supplied, one entry is required per plan stage.
  Empty paths preserve mock argv. All nonempty paths must be absolute,
  bounded and free of CR/LF/NUL or dot path components; validation completes
  **before spawning any worker**. Legacy and per-stage modes cannot mix.
- A host-only HIP CLI parser accepts exactly the supervised worker argv,
  validates the plan, positions/identities and HIP stage index, and leaves
  its result unchanged on failure.
- Actual fork/exec tests verify config argv delivery to one CUDA and two
  HIP stage slots, both socket and shared-memory modes. The probe does
  not link or initialize GPU libraries.

## Validation and continuation

The new parser has 50 assertions tested locally with GCC/Clang using a
standalone host contract double. Full actual-project CMake/CTest on Oberon
is still required; **21 CTests expected**, including the new parser test.
Oberon builds should use --parallel 28. No Actions runs.

This slice is not a native HIP worker and cannot produce inference.
The independent HIP startup config and model-file preflight follow next,
then runtime-owned verifier/experts and actual CUDA->HIP stage validation.
Live GPU loading, inference, prefix commit and numerical parity require
separate owner authorization. Preserve the probe/stash and project isolation.
