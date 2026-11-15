# Known-good Strata HIP build and launch configuration

This document records the user-supplied known-good build recipe and the supplied Strata-lab sequential launch configuration (2026-11-15). It is an operational reference, not a new benchmark or a claim that this precise Q2_0 adaptation has been tested.

## Build environment (vanilla Strata)

Use the same existing toolchain and Python virtual environment as vanilla Strata. Run from the source checkout with an unused build directory. No new Python environment or dependencies are required by this recipe.

```bash
export ROCM_PATH=/opt/rocm/core-10.1
export HIP_PLATFORM=amd
export PATH="$ROCM_PATH/bin:$ROCM_PATH/lib/llvm/bin:/usr/local/cuda/bin:$PATH"

cmake -S . -B <build-path> -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DSTRATA_ENABLE_HIP=ON \
    -DSTRATA_ENABLE_CUDA=OFF \
    -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
    -DSTRATA_PREFILL_MMQ=ON \
    -DSTRATA_BUILD_TESTS=ON \
    -DCMAKE_BUILD_RPATH=/opt/rocm/lib

cmake --build <build-path> --parallel 28
```

Replace the bracketed build path before running. HIP targets RDNA4 gfx1201; CUDA is disabled and HIP prefill MMQ is enabled. Use 28 parallel build jobs (and `ctest --test-dir <build-path> -j28` for local tests). Do not dispatch GitHub Actions.

## Known-good sequential launcher configuration

Source: user-supplied `strata-iq3_xxs-sequential(5).json`. The original operational config uses IQ3_XXS. For this project's reference runs, the user directed us to use the existing **Q2_0** model instead. The JSON below preserves the original operational switches and substitutes Q2_0 model/pack/tokenizer entries. Private machine-specific paths are represented by placeholders; resolve all of them to the approved existing local assets before launch. Do not assume an exact Q2_0 filename, profile compatibility or executable provenance from the original IQ3_XXS config.

The original executable is under `Strata-lab/builds/runtime-parity/hip/strata` and its working directory is `Strata-lab/src/runtime-parity`. For a newly built reference, use its actual executable and matching checkout instead.

```json
{
  "exe": "<WORKSPACE>/Strata-lab/builds/runtime-parity/hip/strata",
  "args": [
    "--pack", "<WORKSPACE>/Strata-data/packs/q2_0",
    "--native", "<APPROVED_Q2_0_GGUF_SHARD_1>",
    "--ple-gguf", "<APPROVED_Q2_0_GGUF_SHARD_2>",
    "--expert-profile", "<APPROVED_Q2_0_EXPERT_PROFILE>",
    "--expert-cache", "auto",
    "--mmap-experts",
    "--prefill", "auto",
    "--spec", "4",
    "--spec-min-p", "0.5",
    "--mtp", "<WORKSPACE>/Strata-data/mtp/rt",
    "--max-context", "262144",
    "--kv", "int8",
    "--kv-resident", "0",
    "--vision",
    "--vram-reserve-mib", "700",
    "--control-vector-scaled", "<WORKSPACE>/Strata/data/experimental-speed-projection/Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0",
    "--control-vector-layer-range", "4", "44",
    "--cvec-mode", "project",
    "--cvec-dir", "per-layer",
    "--pipeline-windows", "0",
    "--prompt-cache", "6",
    "--prompt-cache-spread",
    "--conversation-cache-slots", "4",
    "--conversation-cache-min-tokens", "8192"
  ],
  "cwd": "<WORKSPACE>/Strata-lab/src/runtime-parity",
  "tokenizer": "<WORKSPACE>/Strata-data/packs/q2_0/tokenizer",
  "model_name": "qwen3.8-flash-next-q2_0",
  "log": "<PRIVATE_LOG_PATH>",
  "lib_dirs": [
    "/opt/rocm/core-10.1/lib"
  ],
  "port": 8081,
  "backend": "hip",
  "env": {
    "HSA_ENABLE_SDMA": "0",
    "STRATA_SPLIT_OVERLAP": "1",
    "STRATA_PIPELINE_AUTO": "0",
    "STRATA_TRANSFER_TRACE": "1"
  },
  "gpu": [0, 1],
  "gpus_asked": true,
  "layer_split": "auto",
  "vision": {
    "exe": "<WORKSPACE>/Strata/engine/strata-vision-hip",
    "mmproj": "<APPROVED_MMPROJ_GGUF>",
    "model": "<APPROVED_Q2_0_GGUF_SHARD_1>",
    "gpu": true,
    "max_tokens": 1024,
    "cpu_exe": "<WORKSPACE>/Strata/engine/strata-vision"
  },
  "parallel": 1
}
```

The original also contains an `args_written` array identical to `args`. It is omitted above to avoid maintaining a duplicate; if the local launcher requires it, populate it with the same substituted argument list. The model name and Q2_0 pack/tokenizer directory convention are adaptations, not independently verified paths. Confirm the actual asset paths and MTP/vision/control-vector compatibility locally. Keep any resolved machine-specific configuration in private scratch. The source did not supply a launcher shell command; use the existing Strata launcher and venv.

## Controlled reference vs operational baseline

Do not silently change this operational baseline when making a qualification config. Create a separate local config for matched vanilla versus heterogeneous tests:

- Use the same approved Q2_0 pack, both GGUF shards, tokenizer, template and expert profile.
- Start at 2048 context; compare explicit `--kv int8` and `--kv f16` separately. Match actual placement, full residency, reserves (qualification uses 1 GiB per owner), prompt/prefill, sampling, token IDs and termination.
- Disable MTP/speculative decoding, vision, control-vector projection and cross-request caching using the actual native CLI/launcher semantics. Do not assume the operational flags above form a matched reference.
- Record native source revision, toolchain/build configuration, binary hash, effective argv and relevant runtime environment. Keep tracing separate from performance measurements.
- The operational configuration requests 262144 context and INT8 KV. This is a configuration target, not proof of full-length inference, full residency, or Q2_0 quality/performance qualification. The original 700 MiB reserve is recorded for fidelity, not permission to lower the project's qualification reserve.
- Keep Q8 as the project's default and F16 as an explicit reference. Same-format parity gates and graceful GPU shutdown requirements remain unchanged.

No new hardware result or benchmark is asserted by this document.
