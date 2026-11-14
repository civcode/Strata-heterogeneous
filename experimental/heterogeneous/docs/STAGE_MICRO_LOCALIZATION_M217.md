# M2.17 — Exact shared-expert gate/down and GDN preconv-history probe

**State:** code committed; live M2.17 CUDA/HIP correctness UNVERIFIED.
**Branch:** feature/heterogeneous-cuda-window-admission
**Model:** Qwen3.8-Flash-Next, Q2_0, 48 layers, n_embd=2560,
hc=4, 512 experts, fingerprint=2026111301.
**Reference:** hip:0:0-48 (R9700 PCI 0000:03:00.0).
**Mixed:** cuda:0:0-12,hip:0:12-30,hip:1:30-48 (RTX at
0000:04:00.0 and R9700s at 0000:03:00.0 / 0000:07:00.0).

## Established evidence

M2.15 with both stage-zero workers on STRATA_HC_Q8=0:
- Position 0, layer 5: selected router expert IDs match, router
  logits relative-L2 4.51e-8, route weights 2.16e-7, routed
  expert parts 2.86e-7, but shared expert output **0.002823**.
  This drives the combined MoE difference **0.001073**.
- Position 1, layer 6: QKV and z match exactly; convolution
  intermediate **0.000267**, recurrent y **0.000567** and final
  GDN output **0.002753**.
- Both runs completed positive-prefix commits and predicted 198/271.
  Full-head KL **0.005003 / 0.037718**, FAIL vs 0.001.

M2.16 controlled runs, both sequential and valid:
- Arm A STRATA_SH_STREAM=0: layer5 shared output still 0.002823
  at position 0; position1 shared 2.06e-7.
- Arm B also STRATA_FUSED_SWIGLU_Q81=0: shared output still
  0.002823 / 2.06e-7. Head KL worsened at position 0
  to 0.010083 (position1 0.037814). These results do not
  establish which internal shared projection differed.
- Distinct RTX/R9700 GPU PCI identities and 83/83 Python,
  37/37 host CTest were reported.
- All heterogeneous adapters use set_always_publish(true);
  the CUDA/HIP one-token self-commit default is INACTIVE here.

## New instrumentation (opt-in)

The --micro-probe flag, with --capture-prefix and --rows 1,
enables a bounded real native-device capture before graph creation.
With this flag absent there are **no** extra buffers, copies or
files in inference.

The first-stage worker emits one unique STM0/v1 binary per completed
token, PREFIX.stage0.w1.stm0 and PREFIX.stage0.w2.stm0.
The checksum excludes only header bytes 64–67, size is <=4 MiB,
and the header includes token, stage, request/window/position,
plan fingerprint, FP32 shapes and conv history depth.

Fixed FP32 component order:
1. layer5 shared_input — second HC mixed input before shared expert
2. layer5 gate_raw — native gate projection before SwiGLU
3. layer5 up_raw — native up projection before SwiGLU
4. layer5 down_unscaled — down projection BEFORE scalar sigmoid gate
5. layer5 gate_scale — the shared-expert sigmoid scalar
6. layer5 shared_final — post-scaling contribution to MoE
7. layer6 QKV — input to GDN convolution
8. layer6 conv_history — full last (ssm_d_conv-1) history for
   each conv channel, copied before the native convolution reads it
9. layer6 conv_output — post-convolution/normalization intermediate

Two additional exact **raw Q8_1 byte images**:
- shared_input_q8 — native quantized shared gate/up input;
- swiglu_q8 — native quantized SwiGLU activation consumed by the
  shared down projection.

The native shared_expert_multi tracing is on its *producer stream*,
and copies gate/up before SwiGLU potentially modifies gate scratch,
the raw quantized image before reuse, and down BEFORE sigmoid
scaling. The verifier captures convolution history on its GDN
consumer stream before convolution, not after the commit.

The native worker prints STRATA_SHARED_MICRO_PATH with actual
native-BF16 flag, fused SwiGLU flag, Q8 dedup image reuse and
native gate/up/down tensor types, only when --micro-probe is enabled.
No inferential reliance on environment flags alone.

The new CPU-only tools/heterogeneous_stage_micro.py requires complete
STFR/JSON manifests and the *existing* STB0 layer5/6 captures.
It checks that shared_final, layer6 QKV and layer6 conv_output
are BITWISE equal to the corresponding STB0 values per window;
a mismatch invalidates the diagnostic rather than producing a
misleading cross-run comparison. Checksums, finite FP32, exact
model/pack/profile/token/position/source, shape and size are mandatory.

**Do not** infer that an earliest nonidentical scalar is the
first causal operation. Relative L2 is included separately;
tiny upstream differences may amplify. STM0 does not prove
the GPU-resident weight bytes match across architectures beyond
the existing canonical model provenance and native type logs.

## Safety and prerequisites

Repository: /home/chris/workspace/Strata-heterogeneous.
Scratch/logs: /home/chris/tmp/strata-heterogeneous/m217/.
Do NOT touch either Strata-dev directory or the pre-existing
untracked probe. Do not run GitHub Actions, enable auto triggers,
reset/reboot GPUs, unload drivers or do disruptive recovery.
Compile CUDA/HIP workers separately, <=28 parallel build jobs.
Preserve all VRAM reserve, GGUF/index/pack provenance, nonfinite
and positive-prefix commit gates. Run only the two requested
one-token windows per session. No extra precision controls.

## Sync and CPU/native build gates

~~~bash
cd /home/chris/workspace/Strata-heterogeneous
git switch feature/heterogeneous-cuda-window-admission
git pull --ff-only
git status --short

python3 -m unittest discover -s tools -p test_heterogeneous_stage_micro.py -v
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

Do not run native GPU workers until the new Verifier API and
shared_expert_multi signature build successfully on both backends,
the 21 new CPU fixtures pass, and host CTest passes. If compilation
fails, fix exactly the failure and rerun local checks. Do NOT use
Actions to compensate for a local error.

## EXACT matched-BF16 live runs (fresh, sequential, socket only)

This keeps M2.16 Arm A's STRATA_SH_STREAM=0 and fused SwiGLU ON
(the default), with STRATA_HC_Q8=0 on both. Do not repeat the
unfused Arm B or change transport. Enable both --branch-probe
and --micro-probe for the required cross-record consistency check.

~~~bash
cd /home/chris/workspace/Strata-heterogeneous
set -euo pipefail
unset STRATA_HC_Q8 STRATA_SH_STREAM STRATA_FUSED_SWIGLU_Q81
unset STRATA_HC_SPLIT STRATA_GR_NORM_FAST STRATA_GR_V3
unset STRATA_GR_FAST STRATA_GR_UP_WIDE STRATA_TAP_FILE
unset STRATA_HETERO_BRANCH_PROBE_PREFIX STRATA_HETERO_STAGE0_PHASE_PREFIX
unset STRATA_HETERO_L0_PROBE_PREFIX STRATA_HETERO_LAYER_TAP_PREFIX
unset STRATA_HETERO_MICRO_PROBE_PREFIX

mkdir -p /home/chris/tmp/strata-heterogeneous/m217
OUT=/home/chris/tmp/strata-heterogeneous/m217
SMOKE="$PWD/build-hetero-smoke-cuda/experimental/heterogeneous/strata_heterogeneous_live_smoke"
CUDA_WORKER="$PWD/build-hetero-smoke-cuda/experimental/heterogeneous/strata_heterogeneous_cuda_worker"
HIP_WORKER="$PWD/build-hetero-smoke-hip/experimental/heterogeneous/strata_heterogeneous_hip_worker"
common=(--layers 48 --hc 4 --n-embd 2560 --fingerprint 2026111301
        --rows 1 --windows 2 --tokens 1,1 --timeout-ms 120000
        --micro-probe --branch-probe --stage0-phases
        --layer-taps --layer0-probe)
ref=(--plan hip:0:0-48 --hip-worker "$HIP_WORKER"
     --config "$PWD/build-q2-stage-configs/hip-reference-full-m210.cfg")
het=(--plan cuda:0:0-12,hip:0:12-30,hip:1:30-48
     --cuda-worker "$CUDA_WORKER" --hip-worker "$HIP_WORKER"
     --config "$PWD/build-q2-stage-configs/cuda-stage0.cfg"
     --config "$PWD/build-q2-stage-configs/hip-stage1.cfg"
     --config "$PWD/build-q2-stage-configs/hip-stage2.cfg")

# CPU-only source/preflight, no workers:
env STRATA_HC_Q8=0 STRATA_SH_STREAM=0 "$SMOKE" "${common[@]}" "${ref[@]}" \
  --capture-prefix "$OUT/ref-two"
env STRATA_HC_Q8=0 STRATA_SH_STREAM=0 "$SMOKE" "${common[@]}" "${het[@]}" \
  --capture-prefix "$OUT/het-two"

# Real native stages, one at a time:
env STRATA_HC_Q8=0 STRATA_SH_STREAM=0 "$SMOKE" "${common[@]}" "${ref[@]}" \
  --capture-prefix "$OUT/ref-two" --run 2>&1 | tee "$OUT/ref-two.log"
env STRATA_HC_Q8=0 STRATA_SH_STREAM=0 "$SMOKE" "${common[@]}" "${het[@]}" \
  --capture-prefix "$OUT/het-two" --run 2>&1 | tee "$OUT/het-two.log"

grep -F 'STRATA_GR0_PATH down_q8=0 up_q8=0' "$OUT/ref-two.log"
grep -F 'STRATA_GR0_PATH down_q8=0 up_q8=0' "$OUT/het-two.log"
grep -F 'STRATA_SHARED_MICRO_PATH' "$OUT/ref-two.log"
grep -F 'STRATA_SHARED_MICRO_PATH' "$OUT/het-two.log"

# Reject absent STB0/STFR, misaligned windows or field-layout overlap:
python3 tools/heterogeneous_stage_micro.py \
  --reference-prefix "$OUT/ref-two" --heterogeneous-prefix "$OUT/het-two" \
  --relative-threshold 0.001 --report "$OUT/microtrace.json"

python3 tools/heterogeneous_stage0_phases.py \
  --reference-prefix "$OUT/ref-two" --heterogeneous-prefix "$OUT/het-two" \
  --relative-threshold 0.001 --report "$OUT/stage0-phases.json"

rc=0
python3 tools/heterogeneous_compare_parity.py \
  --reference-prefix "$OUT/ref-two" --heterogeneous-prefix "$OUT/het-two" \
  --max-kl 0.001 --report "$OUT/head-parity.json" || rc=$?
if [[ "$rc" -eq 1 ]]; then
  echo "Full-head numerical parity remains FAIL"
elif [[ "$rc" -ne 0 ]]; then
  exit "$rc"
fi
~~~

Abort immediately on native build, OOM, device identity, worker,
checksum, STB0 overlap, nonfinite or commit failures. No GPU resets
or automatic Actions. Keep all raw captures and logs.

## Decision table after M2.17

| Measurement | Interpretation / next experiment |
| --- | --- |
| layer5 shared_input or input Q8 differ materially | First control input/dequantization; do not blame gate/down math yet |
| Input/Q8 match; gate/up differ | Native shared gate/up MMVQ or loaded weight path is implicated; compare isolated identical quantized input/weight bytes |
| Gate/up match; swiglu Q8 differs | Fused SwiGLU/Q8 kernel difference candidate; inspect exact intermediate quantization |
| SwiGLU Q8 matches; down_unscaled differs | Down MMVQ / Q2_0 accumulation is candidate |
| Down and gate scalar match; shared_final differs | Shared scaling operation is candidate |
| layer6 QKV matches; conv_history differs | **Prior committed convolution state** differs; trace accepted-prefix history update before comparing kernel |
| QKV AND conv_history match; conv_output differs | Conv kernel arithmetic becomes a strong candidate; isolate direct one-step convolution with identical inputs and weights |
| All microtrace fields match but previous STB0 differs | Fail the cross-record check; investigate capture semantics, not parity |

Both positions must be reported independently. Report component-level
relative L2, RMSE, maximum absolute difference, exact Q8 byte counts,
the selected STRATA_SHARED_MICRO_PATH flags, source fingerprints and
GPU PCI functions. A small first nonzero difference is not a
proven first causal operation.

M2.10 full-head KL<=0.001 and exact prediction/top-1 agreement
remain the only existing numerical parity acceptance criterion;
the threshold is not relaxed. No M2.17 real GPU evidence exists yet.
