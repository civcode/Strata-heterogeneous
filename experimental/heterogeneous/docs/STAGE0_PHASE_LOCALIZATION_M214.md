# M2.14 — Stage-zero phase localization (matched BF16)

> **Historical cross-backend numerical diagnostic.** The active implementation
> and architecture gate is [CUDA → HIP contract (M2.18)](IMPLEMENTATION_CONTRACT.md).
> Preserve all original measurements and KL thresholds below as history;
> their failure alone does not establish a CUDA → HIP handoff defect.


**Status:** Implementation committed; native M2.14 GPU runs NOT YET EXECUTED.
**Plan:** HIP reference hip:0:0-48 vs heterogeneous cuda:0:0-12,hip:0:12-30,hip:1:30-48.
**Model:** Q2_0, 48 layers, 2560 embedding, HC=4, fingerprint=2026111301.

## What M2.13 established

Under STRATA_HC_Q8=0 on both devices, layer-zero HC mixed
relative L2 fell from 0.006630 to 7.27e-8; embeddings and GDN QKV
were identical, as were raw Q8_1 inputs. The remaining post-layer
relative-L2>0.001 first occurred at layer 8, position 0 and
layer 6, position 1. M2.10 full-head KL still FAILED:
0.005003 and 0.037718 versus a 0.001 threshold.

The original HC default-precision mismatch is largely explained.
The later same-BF16 discrepancy is not. No GR/KL tolerances change.

## New native instrumentation

Manually enabled --stage0-phases captures these FP32 n_embd=2560
vectors after each of layers 0–11:

1. hc0_mixed — first hyper-connection read, input to attention/GDN
2. attention_gdn_bo — main branch output before second HC read
3. hc1_mixed — second HC read, input to FFN/MoE
4. moe_bo — final expert-combined output before residual write

Each first-stage worker writes an exclusive, bounded and checksummed
PREFIX.stage0.wN.stp0 file, containing a 72-byte header, token ID,
and 12×4×2560 FP32 values, in layer/phase/feature order. With the
flag absent, no new GPU allocation/copy is performed.

The CPU-only tools/heterogeneous_stage0_phases.py requires
complete STFR and manifest evidence and validates token, position,
model/pack/profile, wire shape, checksum and finite values.
It reports per-phase relative L2, RMSE, max absolute difference and
the first phase above diagnostic relative-L2=0.001.
The diagnostic threshold is not M2.10 head-logit KL acceptance.

A mismatch first measured at hc0_mixed may have originated at the
previous layer's residual; attention_gdn_bo implicates the main
branch; hc1_mixed implicates the second HC read or prior branch;
moe_bo implicates router, experts or combine. An earliest
threshold crossing is not necessarily the first nonzero error.

## Safety and build

Use only /home/chris/workspace/Strata-heterogeneous and
/home/chris/tmp/strata-heterogeneous. Do not touch either Strata-dev
directory. No GitHub Actions, GPU resets/reboots, driver unload,
or disruptive recovery. Max 28 build jobs. Preserve all original
model checks, VRAM reserve, finite checks and positive-prefix commits.
The user-requested narrow GPU run is authorized; no extra ritual.

~~~bash
cd /home/chris/workspace/Strata-heterogeneous
git switch feature/heterogeneous-cuda-window-admission
git pull --ff-only
git status --short
python3 -m unittest discover -s tools -p test_heterogeneous_stage0_phases.py -v
python3 -m unittest discover -s tools -p test_heterogeneous_hc_mode_compare.py -v
python3 -m unittest discover -s tools -p test_heterogeneous_layer0_arithmetic.py -v
python3 -m unittest discover -s tools -p test_heterogeneous_layer_divergence.py -v
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
cmake --build build-hetero-smoke-hip --parallel 28 \
  --target strata_heterogeneous_hip_worker
cmake --build build-hetero-smoke-cuda --parallel 28 \
  --target strata_heterogeneous_cuda_worker strata_heterogeneous_live_smoke
~~~

Native C++ compilation has NOT been performed in ChatGPT; the above
builds and host CTest must succeed before GPU launch.

## Fresh matched-BF16 native runs — sequential, socket only

Retain the validated full HIP reference and three existing stage configs.
Use exactly token 1 at positions 0 and 1. Do not run an additional
four-window, shared-memory or plain-BF16 control until the phase result
is reviewed.

~~~bash
cd /home/chris/workspace/Strata-heterogeneous
set -euo pipefail
unset STRATA_HC_Q8 STRATA_HC_SPLIT STRATA_GR_NORM_FAST STRATA_GR_V3
unset STRATA_GR_FAST STRATA_GR_UP_WIDE STRATA_TAP_FILE
unset STRATA_HETERO_STAGE0_PHASE_PREFIX STRATA_HETERO_L0_PROBE_PREFIX
unset STRATA_HETERO_LAYER_TAP_PREFIX

mkdir -p /home/chris/tmp/strata-heterogeneous/m214
OUT=/home/chris/tmp/strata-heterogeneous/m214
SMOKE="$PWD/build-hetero-smoke-cuda/experimental/heterogeneous/strata_heterogeneous_live_smoke"
CUDA_WORKER="$PWD/build-hetero-smoke-cuda/experimental/heterogeneous/strata_heterogeneous_cuda_worker"
HIP_WORKER="$PWD/build-hetero-smoke-hip/experimental/heterogeneous/strata_heterogeneous_hip_worker"

common=(--layers 48 --hc 4 --n-embd 2560 --fingerprint 2026111301
        --rows 1 --windows 2 --tokens 1,1 --timeout-ms 120000
        --stage0-phases --layer-taps --layer0-probe)
ref=(--plan hip:0:0-48 --hip-worker "$HIP_WORKER"
     --config "$PWD/build-q2-stage-configs/hip-reference-full-m210.cfg")
het=(--plan cuda:0:0-12,hip:0:12-30,hip:1:30-48
     --cuda-worker "$CUDA_WORKER" --hip-worker "$HIP_WORKER"
     --config "$PWD/build-q2-stage-configs/cuda-stage0.cfg"
     --config "$PWD/build-q2-stage-configs/hip-stage1.cfg"
     --config "$PWD/build-q2-stage-configs/hip-stage2.cfg")

env STRATA_HC_Q8=0 "$SMOKE" "${common[@]}" "${ref[@]}" --capture-prefix "$OUT/ref-two"
env STRATA_HC_Q8=0 "$SMOKE" "${common[@]}" "${het[@]}" --capture-prefix "$OUT/het-two"

env STRATA_HC_Q8=0 "$SMOKE" "${common[@]}" "${ref[@]}" \
  --capture-prefix "$OUT/ref-two" --run 2>&1 | tee "$OUT/ref-two.log"
env STRATA_HC_Q8=0 "$SMOKE" "${common[@]}" "${het[@]}" \
  --capture-prefix "$OUT/het-two" --run 2>&1 | tee "$OUT/het-two.log"

grep -F 'STRATA_GR0_PATH down_q8=0 up_q8=0' "$OUT/ref-two.log"
grep -F 'STRATA_GR0_PATH down_q8=0 up_q8=0' "$OUT/het-two.log"

python3 tools/heterogeneous_stage0_phases.py \
  --reference-prefix "$OUT/ref-two" --heterogeneous-prefix "$OUT/het-two" \
  --relative-threshold 0.001 --report "$OUT/stage0-phases.json"

python3 tools/heterogeneous_layer_divergence.py \
  --reference-prefix "$OUT/ref-two" --heterogeneous-prefix "$OUT/het-two" \
  --relative-threshold 0.001 --report "$OUT/layer-divergence.json"

# Keep independent original head KL acceptance; rc=1 is expected mismatch,
# rc=2 is an integrity error and must stop the investigation.
rc=0
python3 tools/heterogeneous_compare_parity.py \
  --reference-prefix "$OUT/ref-two" --heterogeneous-prefix "$OUT/het-two" \
  --max-kl 0.001 --report "$OUT/head-parity.json" || rc=$?
if [[ "$rc" -eq 1 ]]; then
  echo "Full-head numerical parity still FAIL"
elif [[ "$rc" -ne 0 ]]; then
  exit "$rc"
fi
~~~

Do not reuse earlier output prefixes: the capture rejects existing
files. Stop at the first real native build/runtime failure. Keep
pre-existing untracked probes untouched.

## Report

- For position 0 and 1: first above-threshold stage-zero
  layer/phase, relative L2, RMSE, max abs.
- Per-phase metrics surrounding layers 6–8, especially where
  previous phase relative L2 is small and next phase grows.
- Existing post-layer STLT crossings and full-logit KL.
- Backend PCI identities, test/build commands, successful commits,
  failures/fixes, commit SHA, raw STP0/STLT/STFR artifacts and logs.

**Validation in this development environment:** 14/14 new M2.14
CPU tests; 64/64 total available CPU synthetic/regression tests.
The committed Python comparator/test blobs match locally tested
files byte-for-byte. No real M2.14 CUDA/HIP test has yet run.
