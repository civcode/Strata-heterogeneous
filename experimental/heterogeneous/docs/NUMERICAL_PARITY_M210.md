# M2.10 — Native unsplit reference numerical parity handoff

**Baseline:** M2.9 real CUDA→HIP→HIP smoke PASSED; commit 28446b0d503d4067c40f2e95e4a88262e82b3bf9.
**Stage plan:** cuda:0:0-12,hip:0:12-30,hip:1:30-48
**Reference plan:** hip:0:0-48 on a single R9700
**Model:** Q2_0 48 layers, hc=4, n_embd=2560, vocab=248320; fingerprint=2026111301.
**Status:** code and instructions prepared; real M2.10 numerical comparison remains UNTESTED.

## Implementation

The full-model HIP final verifier now supports the layer-zero embedding
path, so no prior activation handoff is necessary in a single HIP stage.
The host-only live_smoke supports --tokens (per-row exact token inputs)
and --capture-prefix, which records each fully committed STFR/v1 packet
with its complete FP32 head logits, input window, GGUF shard paths,
pack-index SHA, profile path, model identity, and a completion marker.
Only successful process shutdown writes PREFIX.complete.json.

tools/heterogeneous_reference_config.py creates a new hip:0:0-48 v2
config from the trusted CUDA stage-0 config. Its PLE source, original
GGUF shards, pack, profile, and safety reserve stay unchanged. The
12-layer arena_budget_bytes is replaced with zero (no artificial cap);
**the native loader still checks actual free VRAM and the configured
reserve**. The source config is never overwritten.

tools/heterogeneous_compare_parity.py checks source/input provenance,
STFR checksum and finite logits, then calculates KL(reference||hetero),
greedy top-1 and prediction agreement, max absolute FP32-logit difference,
and RMSE per row. Provisional acceptance is 100% predictions/top-1
agreement and each row KL <= 0.001. Report actual discrepancies even
when HIP's Q8 hyper-connection precision mode differs. This is a
same-engine unsplit native Strata reference, not independent llama.cpp parity.

## Constraints

Work only under /home/chris/workspace/Strata-heterogeneous and
/home/chris/tmp/strata-heterogeneous. Never touch the two Strata-dev
directories. No GitHub Actions, GPU resets, reboot, driver unload, or
disruptive recovery. Separate CUDA and HIP builds/processes; max 28
build jobs. User-initiated live GPU tests are authorized for this scope.

## Step 1: sync, CPU-only tests, and rebuild

~~~bash
cd /home/chris/workspace/Strata-heterogeneous
git switch feature/heterogeneous-cuda-window-admission
git pull --ff-only
git status --short
python3 -m unittest discover -s tools -p test_heterogeneous_compare_parity.py -v
cmake -S experimental/heterogeneous -B build-hetero-smoke-host
cmake --build build-hetero-smoke-host --parallel 28
ctest --test-dir build-hetero-smoke-host --output-on-failure

export ROCM_PATH=/opt/rocm/core-10.1 HIP_PLATFORM=amd
export PATH="$ROCM_PATH/bin:$ROCM_PATH/lib/llvm/bin:/usr/local/cuda/bin:$PATH"
cmake -S . -B build-hetero-smoke-hip \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DSTRATA_BUILD_HETERO_HIP_WORKER=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
  -DCMAKE_PREFIX_PATH="$ROCM_PATH" -DCMAKE_BUILD_RPATH=/opt/rocm/lib
cmake --build build-hetero-smoke-hip --parallel 28 --target strata_heterogeneous_hip_worker
cmake --build build-hetero-smoke-cuda --parallel 28 \
  --target strata_heterogeneous_cuda_worker strata_heterogeneous_live_smoke

python3 tools/heterogeneous_reference_config.py \
  --source "$PWD/build-q2-stage-configs/cuda-stage0.cfg" \
  --output "$PWD/build-q2-stage-configs/hip-reference-full-m210.cfg" \
  --ordinal 0
~~~

If the reference config already exists, verify rather than overwrite.
MMQ prefill remains unnecessary for verifier-window parity.

## Step 2: exact live two-window comparison

Each test uses fresh model state and exactly token 1 at positions 0 and 1.
Run reference and heterogeneous sequentially, never concurrently.

~~~bash
cd /home/chris/workspace/Strata-heterogeneous
set -o pipefail
mkdir -p /home/chris/tmp/strata-heterogeneous/m210
OUT=/home/chris/tmp/strata-heterogeneous/m210
SMOKE="$PWD/build-hetero-smoke-cuda/experimental/heterogeneous/strata_heterogeneous_live_smoke"
CUDA_WORKER="$PWD/build-hetero-smoke-cuda/experimental/heterogeneous/strata_heterogeneous_cuda_worker"
HIP_WORKER="$PWD/build-hetero-smoke-hip/experimental/heterogeneous/strata_heterogeneous_hip_worker"

common=(--layers 48 --hc 4 --n-embd 2560 --fingerprint 2026111301
        --rows 1 --windows 2 --tokens 1,1 --timeout-ms 120000)
ref=(--plan hip:0:0-48 --hip-worker "$HIP_WORKER"
     --config "$PWD/build-q2-stage-configs/hip-reference-full-m210.cfg")
het=(--plan cuda:0:0-12,hip:0:12-30,hip:1:30-48
     --cuda-worker "$CUDA_WORKER" --hip-worker "$HIP_WORKER"
     --config "$PWD/build-q2-stage-configs/cuda-stage0.cfg"
     --config "$PWD/build-q2-stage-configs/hip-stage1.cfg"
     --config "$PWD/build-q2-stage-configs/hip-stage2.cfg")

# CPU-only checks before any native initialization:
"$SMOKE" "${common[@]}" "${ref[@]}"
"$SMOKE" "${common[@]}" "${het[@]}"

# Actual GPU runs. Stop on the first failure, retain its log.
"$SMOKE" "${common[@]}" "${ref[@]}" --capture-prefix "$OUT/ref-two" --run \
  2>&1 | tee "$OUT/ref-two.log"
"$SMOKE" "${common[@]}" "${het[@]}" --capture-prefix "$OUT/het-two" --run \
  2>&1 | tee "$OUT/het-two.log"

python3 tools/heterogeneous_compare_parity.py \
  --reference-prefix "$OUT/ref-two" --heterogeneous-prefix "$OUT/het-two" \
  --max-kl 0.001 --report "$OUT/two-parity.json"
~~~

If the two-window test passes, repeat with fresh output prefixes,
--windows 4 --tokens 1,198,271,1 on both fresh sessions to check
more state transitions. Compare shared-memory versus socket too, but
never substitute cross-transport agreement for the unsplit reference.

## Outcome criteria

**PASS:** both live sessions complete and commit, exact same Q2_0 source,
GGUF shards and profile; identical token inputs, positions, vocab size,
predictions and logit top-1; finite complete logits; KL <= 0.001 per row.

**FAIL:** report first divergent position, predictions, top-1, KL, logit
RMSE/max delta, actual device IDs, and any HIP Q8 precision settings.
Do not waive a difference without investigation.

**REFERENCE UNAVAILABLE:** if full-stage HIP cannot allocate or initialize,
record the genuine error and stop. This is not a numerical mismatch.
Do not bypass memory, model identity or runtime safety checks.

Numerical parity with unsplit native Strata remains UNVERIFIED until
the operator actually runs these tests. Prompt prefill, long context,
external llama.cpp correctness, and throughput are separate milestones.

## Current validation evidence

Locally ran `python3 -m unittest -v test_heterogeneous_compare_parity.py`
against source files verified byte-for-byte identical to the GitHub
branch blobs: **14/14 CPU-only synthetic tests PASS**. These cover
full-logit identity, two successive positions, corrupt STFR checksums,
nonfinite logits, missing completion/packets, missing logits, incorrect
model source/input sequence, emitted-prediction mismatches, KL mismatch,
reference config generation, and rejection of trivial self-comparisons
or swapped reference arguments. No CUDA/HIP runtime was initialized.

C++ native builds and the live unsplit/heterogeneous runs have **not**
been executed in this ChatGPT environment; their exact execution
commands are provided above for Oberon. Actual numerical parity remains
**UNVERIFIED**, independent of the passing synthetic parser tests.
