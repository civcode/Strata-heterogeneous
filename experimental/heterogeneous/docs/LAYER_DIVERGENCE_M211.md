# M2.11 — Locate the first observable layer divergence

> **Historical cross-backend numerical diagnostic.** The active implementation
> and architecture gate is [CUDA → HIP contract (M2.18)](IMPLEMENTATION_CONTRACT.md).
> Preserve all original measurements and KL thresholds below as history;
> their failure alone does not establish a CUDA → HIP handoff defect.


**Branch:** feature/heterogeneous-cuda-window-admission
**Reference:** hip:0:0-48 on R9700 PCI 0000:03:00.0
**Heterogeneous:** cuda:0:0-12,hip:0:12-30,hip:1:30-48
**Model fingerprint:** 2026111301, Q2_0, 48 layers, hc=4, n_embd=2560.

## Confirmed M2.10 result: numerical parity FAIL

Both GPU runs completed two real token windows with positive-prefix commits,
and both predicted 198 then 271, but unsplit vs heterogeneous head-logit
KL(reference||heterogeneous) failed the 0.001 threshold:

| Position | Input | Prediction pair | KL | Max abs logit diff | Logit RMSE |
| --- | --- | --- | --- | --- | --- |
| 0 | 1 | 198 / 198 | 0.003376 | 0.471658 | 0.079767 |
| 1 | 1 | 271 / 271 | 0.039019 | 1.206786 | 0.223623 |

A separate STRATA_HC_Q8=0 diagnostic also failed (KL 0.005003,
0.037718), so Q8 read mode alone does not explain the divergence.
M2.10 tests: 14/14 CPU parity and 32/32 host CTest passed.
Live M2.10 artifacts are under /home/chris/tmp/strata-heterogeneous/m210/.
**Do not relax the numerical threshold or claim parity.**

## M2.11 method

The native Verifier already supports set_taps/gather_taps_host. M2.11
uses those existing GPU kernels, opt-in only, without changing native
layer math, model assets, head sampling, or commit policy.

With --layer-taps and --capture-prefix, each worker writes an
exclusive STLT/v1 packet per window:

  PREFIX.stage<stage-index>.w<window-id>.stlt

A 72-byte little-endian header includes version, stage/layer identity,
shape, request/window ID, position and plan fingerprint; a verified
FNV-1a checksum covers the header, token IDs and FP32 samples.
Samples are [row][layer][n_embd] float32, with the layer range
[stage.first_layer, min(stage.end_layer, total_layers-1)).
The full unsplit reference captures layers 0..46; the heterogeneous
stages capture 0..11, 12..29, 30..46. No final layer 47 tap is exposed
by this existing kernel; the M2.10 head-logit comparison remains the
observable final-layer output.

**Important interpretive limit:** These taps are post-layer
HC-mean residuals *with the pending write applied*. They are not a
lossless/full R|bo|inject handoff. A first divergence pinpoints the
earliest divergent observed layer output, not necessarily the exact
sub-operation or whether its origin is a GPU kernel versus a boundary.
Do not infer numerical parity from taps alone.

Comparison requires complete M2.10 STFR/JSON/commit markers for each run
and checks exact source fingerprints, input tokens, state positions,
per-stage identity, checksums, and finite tap values. It reports per-layer
relative L2, RMSE, max absolute difference, and the first tapped layer
where relative L2 exceeds a **diagnostic** threshold (default 0.001).
The tap threshold is not the M2.10 KL acceptance threshold. Boundaries
after layer 11 (CUDA->HIP) and 29 (HIP->HIP) are highlighted in the report.

## Oberon constraints

Use ONLY /home/chris/workspace/Strata-heterogeneous for the repository,
and /home/chris/tmp/strata-heterogeneous for output. Do not access
/home/chris/workspace/Strata-dev or /home/chris/tmp/strata-dev.
GPU runtime work is authorized by the user's requested M2.11 test.
No GitHub Actions; no GPU resets, reboots, driver unload, or disruptive
recovery. Build locally (<=28 parallel jobs), separate CUDA/HIP toolchains.
Do not enable STRATA_TAP_FILE simultaneously with M2.11 taps.

## CPU validation and native rebuild

~~~bash
cd /home/chris/workspace/Strata-heterogeneous
git switch feature/heterogeneous-cuda-window-admission
git pull --ff-only
git status --short

python3 -m unittest discover -s tools -p test_heterogeneous_layer_divergence.py -v
python3 -m unittest discover -s tools -p test_heterogeneous_compare_parity.py -v
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
~~~

Resolve any compiler or test failure locally first. The existing native
M2.10 HIP full-stage reference config is reused unchanged.

## Two-window scoped diagnostic GPU runs

Start **fresh sessions**; do not reuse earlier M2.10 outputs. Run the
reference first, close its workers, then the heterogeneous chain.
No concurrency. Use socket transport initially.

~~~bash
cd /home/chris/workspace/Strata-heterogeneous
set -o pipefail
mkdir -p /home/chris/tmp/strata-heterogeneous/m211
OUT=/home/chris/tmp/strata-heterogeneous/m211
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

# CPU-only preflight. Explicit tap flag does not launch any worker without --run.
"$SMOKE" "${common[@]}" "${ref[@]}" --capture-prefix "$OUT/ref-two" --layer-taps
"$SMOKE" "${common[@]}" "${het[@]}" --capture-prefix "$OUT/het-two" --layer-taps

# Live runs, in series; stop if either command exits nonzero.
"$SMOKE" "${common[@]}" "${ref[@]}" \
  --capture-prefix "$OUT/ref-two" --layer-taps --run 2>&1 | tee "$OUT/ref-two.log"
"$SMOKE" "${common[@]}" "${het[@]}" \
  --capture-prefix "$OUT/het-two" --layer-taps --run 2>&1 | tee "$OUT/het-two.log"

# CPU-only layer localization; requires successful completion markers.
python3 tools/heterogeneous_layer_divergence.py \
  --reference-prefix "$OUT/ref-two" \
  --heterogeneous-prefix "$OUT/het-two" \
  --relative-threshold 0.001 \
  --report "$OUT/layer-divergence.json"

# Existing M2.10 KL result, for corroboration only; do not change threshold.
python3 tools/heterogeneous_compare_parity.py \
  --reference-prefix "$OUT/ref-two" \
  --heterogeneous-prefix "$OUT/het-two" \
  --max-kl 0.001 --report "$OUT/head-parity.json"
~~~

Note: the final M2.10 comparator is **expected to exit 1** on the same
numerical discrepancy; do not confuse its FAIL with a tap-reader failure.
The M2.11 layer diagnostic exits 0 when valid data are fully compared,
which is not a numerical-parity PASS.

## Decision based on first observed divergence

- **Already at layer 0–10:** investigate CUDA vs HIP native kernel
  differences and exact weight/projection options within stage 0.
  Do not blame the interprocess handoff yet.
- **At layer 11/12:** inspect the layer-11 tap and actual
  [R|bo|inject] CUDA-to-HIP boundary; do not treat the HC-mean tap as
  proof of full handoff integrity.
- **At layer 29/30:** inspect HIP-to-HIP boundary and stage-local
  session/weight setup.
- **Only deep in stage 2 / final logits:** check final projection,
  head normalization/precision and state progression.
- **Gradual small differences from early layers:** record per-layer
  metric growth and compare kernel precision settings; do not call
  a gradual numerical drift an isolated transport failure.

If native tap recording fails, stop; preserve first worker error,
exact commands, build configuration, backend/PCI identity and logs.
Do not reboot/reset or artificially loosen the KL threshold.

Expected output is the first above-threshold layer per position,
per-layer metrics near layers 11/12 and 29/30, and a specific next
code-level hypothesis. M2.10 numerical parity remains FAIL until
a separately passing full-logit comparison.


## Development-side validation (CPU only)

- **13/13** STLT/v1 and layer-divergence synthetic Python tests passed
  locally, including an end-to-end generated M2.10 STFR completion marker
  plus all four STLT stages at the real 2560-element embedding width.
- **14/14** existing M2.10 Python parity tests passed again.
- Confirmed the local Python comparator, tests, and native STLT recorder
  header match the committed GitHub blob hashes exactly.
- Compiled the native STLT recorder header against a minimal fake
  Verifier/Plan using both GCC and Clang, with C++17 and
  `-Wall -Wextra -Wpedantic -Werror`. Ran a host-only writer, then
  decoded and checksummed its binary using the actual Python STLT reader.
  Verified exclusive-create rejects overwrites and the legacy
  STRATA_TAP_FILE conflict fails closed.

**This is not a CUDA/HIP build or live M2.11 run.** All actual
gpu-backed verifier tap allocations, native builds and numerical layer
measurements still require the user-directed Oberon hardware test above.
