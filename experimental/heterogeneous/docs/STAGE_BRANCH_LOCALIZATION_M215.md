# M2.15 — Localize layer-5 MoE and layer-6 GDN divergence

**Status:** implementation prepared; GPU M2.15 has NOT been run.
**Plan:** compare single-R9700 HIP reference hip:0:0-48 with CUDA→HIP→HIP
cuda:0:0-12,hip:0:12-30,hip:1:30-48, using Q2_0 model fingerprint
2026111301, 48 layers, n_embd 2560, 512 experts, top-k 10.

## Confirmed baseline

Under matching STRATA_HC_Q8=0, the original layer-0 HC-mixed
relative L2 difference fell to 7.27e-8 (previously 0.006630).
The remaining M2.14 *first above-threshold* phase crossed at:
- position 0: layer 5 moe_bo, relative L2 0.001073; preceding
  hc1_mixed relative L2 1.64e-7
- position 1: layer 6 attention_gdn_bo, relative L2 0.002753;
  preceding hc0_mixed relative L2 2.70e-7

These are amplification sites, not proof of a broken kernel.
Both pipelines predict 198 and 271 and commit. M2.10 full-head KL
still FAILS at 0.005003 and 0.037718, threshold <= 0.001.

## M2.15 opt-in snapshot contract

The native Verifier's new --branch-probe snapshots:
- Layer 5: all 512 router logits, 10 final selected expert IDs,
  10 route weights, shared expert 2560 FP32, 10*2560 expert
  part floats, and the final 2560 MoE-combined output.
- Layer 6 GDN: projected QKV, convolved/normalized intermediate,
  gate and beta, z projection, y recurrent/normalized output,
  and final 2560 projected GDN output.

These snapshots use the actual native GPU operations; the core model
arithmetic and acceptance policy are untouched. Disabled probes incur
no extra allocations or GPU copies.

The first-stage worker writes unique, 1-MiB-bounded, checksummed
STB0/v1 files PREFIX.stage0.w1.stb0 and PREFIX.stage0.w2.stb0.
The format includes exact token, request, window, position, plan
fingerprint and tensor dimensions. A CPU-only reader,
tools/heterogeneous_stage_branch.py, validates full STFR completion,
model/pack/profile input identity, checksum and finite values, then
compares each component separately. Its per-expert detail includes
the expert ID on both backends: different slot IDs make a direct
elementwise expert comparison inconclusive.

The probe fails closed at preparation if layer 6 is not GDN or
the loaded model does not have the required Q2_0 geometry.
The M2.14 phase taps can be captured in the same run.

## Rules

Repository only: /home/chris/workspace/Strata-heterogeneous.
Artifacts only: /home/chris/tmp/strata-heterogeneous.
Never touch either Strata-dev directory, pre-existing untracked probe,
or any GitHub Actions. Do not reset/reboot GPUs, unload drivers or
attempt disruptive recovery. Separate CUDA/HIP builds, <=28 jobs.
Original GGUF/pack, memory reserve, finite-value and positive-prefix
commit checks all remain enforced. These user-requested narrowly
scoped GPU diagnostics are authorized without new approval flags.

## Sync, CPU tests, rebuild

~~~bash
cd /home/chris/workspace/Strata-heterogeneous
git switch feature/heterogeneous-cuda-window-admission
git pull --ff-only
git status --short

python3 -m unittest discover -s tools -p test_heterogeneous_stage_branch.py -v
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
cmake --build build-hetero-smoke-hip --parallel 28 --target strata_heterogeneous_hip_worker
cmake --build build-hetero-smoke-cuda --parallel 28 \
  --target strata_heterogeneous_cuda_worker strata_heterogeneous_live_smoke
~~~

Fix native compiler/CTest/stub errors locally first. No Actions.

## Exactly two fresh sequential native runs

Keep matched BF16 HC projection weights STRATA_HC_Q8=0 on both
backends. Do not force extra GR variants, Q8, or a new kernel mode.
Use socket transport only and the same token sequence 1,1.

~~~bash
cd /home/chris/workspace/Strata-heterogeneous
set -euo pipefail
unset STRATA_HC_Q8 STRATA_HC_SPLIT STRATA_GR_NORM_FAST STRATA_GR_V3
unset STRATA_GR_FAST STRATA_GR_UP_WIDE STRATA_TAP_FILE
unset STRATA_HETERO_BRANCH_PROBE_PREFIX STRATA_HETERO_STAGE0_PHASE_PREFIX
unset STRATA_HETERO_L0_PROBE_PREFIX STRATA_HETERO_LAYER_TAP_PREFIX
mkdir -p /home/chris/tmp/strata-heterogeneous/m215
OUT=/home/chris/tmp/strata-heterogeneous/m215
SMOKE="$PWD/build-hetero-smoke-cuda/experimental/heterogeneous/strata_heterogeneous_live_smoke"
CUDA_WORKER="$PWD/build-hetero-smoke-cuda/experimental/heterogeneous/strata_heterogeneous_cuda_worker"
HIP_WORKER="$PWD/build-hetero-smoke-hip/experimental/heterogeneous/strata_heterogeneous_hip_worker"

common=(--layers 48 --hc 4 --n-embd 2560 --fingerprint 2026111301
        --rows 1 --windows 2 --tokens 1,1 --timeout-ms 120000
        --branch-probe --stage0-phases --layer-taps)
ref=(--plan hip:0:0-48 --hip-worker "$HIP_WORKER"
     --config "$PWD/build-q2-stage-configs/hip-reference-full-m210.cfg")
het=(--plan cuda:0:0-12,hip:0:12-30,hip:1:30-48
     --cuda-worker "$CUDA_WORKER" --hip-worker "$HIP_WORKER"
     --config "$PWD/build-q2-stage-configs/cuda-stage0.cfg"
     --config "$PWD/build-q2-stage-configs/hip-stage1.cfg"
     --config "$PWD/build-q2-stage-configs/hip-stage2.cfg")

env STRATA_HC_Q8=0 "$SMOKE" "${common[@]}" "${ref[@]}" \
  --capture-prefix "$OUT/ref-two"
env STRATA_HC_Q8=0 "$SMOKE" "${common[@]}" "${het[@]}" \
  --capture-prefix "$OUT/het-two"

env STRATA_HC_Q8=0 "$SMOKE" "${common[@]}" "${ref[@]}" \
  --capture-prefix "$OUT/ref-two" --run 2>&1 | tee "$OUT/ref-two.log"
env STRATA_HC_Q8=0 "$SMOKE" "${common[@]}" "${het[@]}" \
  --capture-prefix "$OUT/het-two" --run 2>&1 | tee "$OUT/het-two.log"

grep -F 'STRATA_GR0_PATH down_q8=0 up_q8=0' "$OUT/ref-two.log"
grep -F 'STRATA_GR0_PATH down_q8=0 up_q8=0' "$OUT/het-two.log"

python3 tools/heterogeneous_stage_branch.py \
  --reference-prefix "$OUT/ref-two" \
  --heterogeneous-prefix "$OUT/het-two" \
  --report "$OUT/branch-details.json"
python3 tools/heterogeneous_stage0_phases.py \
  --reference-prefix "$OUT/ref-two" \
  --heterogeneous-prefix "$OUT/het-two" \
  --relative-threshold 0.001 \
  --report "$OUT/stage0-phases.json"

rc=0
python3 tools/heterogeneous_compare_parity.py \
  --reference-prefix "$OUT/ref-two" \
  --heterogeneous-prefix "$OUT/het-two" \
  --max-kl 0.001 --report "$OUT/head-parity.json" || rc=$?
if [[ "$rc" -eq 1 ]]; then
  echo "M2.10 head numerical parity remains FAIL"
elif [[ "$rc" -ne 0 ]]; then
  exit "$rc"
fi
~~~

Use NEW prefixes; incomplete or pre-existing captures fail closed.

## How to interpret the results

At layer 5: if selected IDs differ, inspect close router scores,
top-k/order, route-prior/residency settings. If IDs match but weights
differ, isolate routing normalization. If IDs and weights match but
shared/individual expert parts differ, investigate their native/CPU
paths and weights. If all those match but the final moe_bo differs,
investigate combine/reduction order.

At layer 6: if QKV differs, investigate quantized input/MMVQ. If
QKV matches but convolution output differs, investigate conv/history.
Gate/beta/z mismatch implicates those projections; identical
inputs but different recurrent y implicates the state update/
normalization. If y matches but final GDN output differs, inspect
the last projection.

Compare field-specific relative L2, RMSE and max abs, not only a
threshold crossing. Earlier subthreshold differences can amplify.
Retain all logs, IDs/weights, STB0/STP0/STFR, PCI IDs, test and build
results, first worker error, and any fixes/commit SHA.

No four-window extension, stress test, alternate numerical
tolerance, or speculative kernel fix until evidence is reviewed.
M2.10 numerical parity remains FAIL until original full-head
KL<=0.001 and identical predictions/top1 pass.
