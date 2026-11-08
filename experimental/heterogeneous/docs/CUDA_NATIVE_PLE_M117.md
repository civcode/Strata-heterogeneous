# M1.17 — CUDA NativeDense and PLE resource ownership

**Status:** Implemented from the M1.16 source baseline with local GPU-free
production-source contract tests. No direct Oberon build, CUDA runtime loading,
inference, numerical parity or live teardown has been performed by ChatGPT.
Full CUDA compilation and whole-archive linkage need independent Oberon
validation before any runtime attempt.

## The failure being fixed

The first Q2_0 stage window reached `blk.0.attn_qkv.weight` without a native
GGUF projection. A canonical pack's `WeightTable::load` is not sufficient to
serve this projection. Strata already has the implementation:
`core::NativeDense` maps and uploads the original GGUF matrices and attaches
native pointers to the **same** `WeightRef`s the verifier reads. It must remain
alive until every captured verifier graph is destroyed and GPU work is complete.

Layer 1 additionally requires the n-gram PLE table in an original GGUF shard,
six `blk.1.ple_*` weights, host/device embedding staging, graph scratch,
per-session normalized history and the correct n-gram hash constants.
Silently leaving `SessionState::ple.ready()` false changes the model.

## Configuration and trust

The worker now requires **v2** of its strict configuration for live operation.
Existing v1 configs remain readable for the CPU-only `--check-config` and the
M1.16 regression tests; the worker refuses to start live with v1.

For a stage covering layer 1, add these two fields to the former v1 file and
change `version=1` to `version=2`:

```ini
gguf_shards=/AUTHORIZED/original/model-00001-of-00002.gguf;/AUTHORIZED/original/model-00002-of-00002.gguf
ple_gguf_path=/AUTHORIZED/original/model-00002-of-00002.gguf
```

**These are examples, NOT discovered or verified locations.** Shards are
ordered with the architecture-bearing primary first. Up to 64 distinct,
absolute, non-dot-segment paths are accepted. A stage **without** layer 1
provides `gguf_shards` but omits `ple_gguf_path` entirely. The PLE table path
must appear in the ordered shard list. Preflight checks regular files, GGUF
magic, the trusted pack index SHA-256 and the STRP/v1 expert profile header
before touching CUDA. The native loader additionally validates architecture,
tensor geometry, GGUF span boundaries and canonical pack associations.

**Index SHA-256 is not a signature of the original GGUF or all model blobs.**
The caller remains responsible for obtaining matching, authorized model
artifacts and a nonzero model fingerprint. Do not search other worktrees,
infer an original GGUF path from a pack directory or alter Strata-data.

## Live resource sequence (CUDA-only child)

1. `CudaStageResources::load` selects and loads the stage's compact canonical
   arena and carves/zeroes its session. For the real worker,
   `CudaStageLoadOptions::require_native_stage` is true.
2. `CudaNativeStage::open` validates the ordered GGUF projections via
   `NativeDense::served_names`, sizes extra native CUDA allocations with
   `NativeDense::weight_bytes_for`, checks the cumulative arena budget and
   current GPU memory reserve, then invokes `NativeDense::load` for the stage's
   layer range. **For stage 0 it requires `blk.0.attn_qkv.weight` to be both
   advertised and actually attached as a native `WeightRef`.** Any stage-local
   layer tensor with neither canonical data nor native/Q8 representation is
   rejected.
3. If the stage owns layer 1, it opens the exact authorized PLE GGUF shard in
   `PleIo::Direct` mode, with cache rows/keepalive disabled in this first
   integration. It wires the six canonical weights, the session's normalized
   history, native PLE-key projection/scratch where applicable, n-gram
   constants, pinned host/device embedding and graph scratch. The F16
   convolution type/length check matches the native Strata `generate.cpp`.
   Incomplete PLE is refused, not silently disabled.
4. `CudaFileExperts` opens real experts and attaches to the stage owner; only
   then does `make_bindings` expose the native-ready verifier bindings.
5. The non-final `CudaVerifierDriver` or the final verifier runs within the
   existing independently supervised process and protocol.
6. The **verifier shuts down first**; `CudaNativeStage::shutdown` synchronizes
   and releases PLE and NativeDense; `CudaStageResources::shutdown` finally
   releases experts, session and packed arena. Calling owner shutdown with a
   still-attached native stage exits the child with code 70 rather than
   releasing captured pointers. Driver errors still use the older fail-stop
   rules; this is not a guarantee of safety under a hung kernel.

No PLE table or native CUDA pointer is shared with HIP processes.

## Remaining gates

- **Oberon native build/link:** run the standalone experimental 18-test CTest
  suite under GCC, Clang, ASan, UBSan and CUDA-enabled configurations; compile
  and whole-archive-link the CUDA adapter and optional native worker against
  the real `strata_engine`. Use `cmake --build "$build" --parallel 28`.
- **Authorized inputs:** obtain explicit original GGUF shard paths and the
  exact PLE-bearing shard path. The Q2_0 pack and its `index.txt` alone do not
  contain the 28.8 GB n-gram table or native dense projections.
- **Runtime approval:** only with explicit permission, validate a bounded real
  NativeDense load, PLE table open, first stage window and `commit_prefix`.
  Compare individual outputs and accepted state against a matching unsplit
  Q2_0 reference before attempting a multi-backend run.
- **Not proven:** real native projection arithmetic, PLE n-gram parity, full
  CUDA inference, prompt prefill, zero-prefix rollback, HIP worker and
  numerical parity. Successful CPU-double tests do not prove GPU recovery or
  driver-fault teardown.

## Local CPU-double contract coverage

`strata_heterogeneous_cuda_native_stage_contract_tests` compiles the actual
`src/cuda_native_stage.cpp` and M1.14 `src/cuda_stage_resources.cpp` against
CUDA, native-density, PLE, WeightTable and SessionState doubles. It tests
binding readiness, exact QKV presence, PLE tensor and F16 checks, missing
shards, partial CUDA allocations, memory/budget refusal, cleanup and a
forked wrong-shutdown-order fail-stop. No GPU runtime is invoked.

Worktrees: only `/home/chris/workspace/Strata-heterogeneous` for Oberon source
and `/home/chris/tmp/strata-heterogeneous` for reports. Preserve the untracked
pool probe. Never access either Strata-dev directory. No GitHub Actions runs,
GPU reset, reboot or driver intervention is part of this milestone.
