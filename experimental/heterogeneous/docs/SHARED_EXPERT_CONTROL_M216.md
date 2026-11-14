# M2.16 — Shared-expert stream control (no new native kernel code)

**Status:** hardware test PENDING. Branch feature/heterogeneous-cuda-window-admission.

## Evidence

M2.15, matched STRATA_HC_Q8=0:
- Position 0, layer 5: expert IDs match, router logits relL2=4.51e-8,
  route weights=2.16e-7, expert parts=2.86e-7. Shared expert relL2
  **0.002823** and MoE-combined output **0.001073**.
- Position 1, layer 6: QKV and z match exactly, convolution
  intermediate relL2 **0.000267**, recurrent y **0.000567**,
  output **0.002753**.
- Two commits and predictions 198/271 in both runs. Full-head KL
  **0.005003 / 0.037718**, FAIL against unchanged 0.001.

Source audit:
- src/core/verify.cpp::sh_stream_on() defaults ON on CUDA, OFF on HIP.
  STRATA_SH_STREAM=0 on both aligns shared-expert scheduling.
- src/kernels/cuda/shared_expert.cu::fused_swiglu_q81_enabled()
  defaults ON on both. STRATA_FUSED_SWIGLU_Q81=0 on both tests
  unfused SwiGLU followed by native Q8_1 quantization.
- CUDA/HIP one-token self-commit defaults DO differ, but **all four
  heterogeneous verifier adapters call set_always_publish(true)**,
  disabling that shortcut. Do not test STRATA_ONE_TOKEN_COMMIT.
- GDN convolution history was **not captured** by M2.15. Identical
  QKV does not establish identical prior convolution state. No
  GDN kernel defect has been identified.

Arm A (required): STRATA_HC_Q8=0 STRATA_SH_STREAM=0 on both pipelines.
Compare layer5 shared output vs original M2.15 0.002823 and layer6 conv.
Arm B (conditional only if Arm A still materially differs in layer5
shared): retain A and set STRATA_FUSED_SWIGLU_Q81=0 on both pipelines.

This is a two-variable *sequential* investigation, with only one
new control changed between the arms. Never change KL tolerance.

## Safety and prerequisites

Only use repository /home/chris/workspace/Strata-heterogeneous and
artifacts /home/chris/tmp/strata-heterogeneous/m216. Never access
either Strata-dev directory or disturb pre-existing untracked probes.
No GitHub Actions, GPU reset/reboot, driver unload, or disruptive
recovery. Reuse successfully validated separate CUDA and HIP M2.15
workers; if rebuilt, local jobs <=28. Keep model/pack/GGUF, memory
reserve, nonfinite and positive-prefix commit guards intact.

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
ctest --test-dir build-hetero-smoke-host --output-on-failure

set -euo pipefail
unset STRATA_HC_Q8 STRATA_SH_STREAM STRATA_FUSED_SWIGLU_Q81
unset STRATA_HC_SPLIT STRATA_GR_NORM_FAST STRATA_GR_V3
unset STRATA_GR_FAST STRATA_GR_UP_WIDE STRATA_TAP_FILE
unset STRATA_HETERO_BRANCH_PROBE_PREFIX STRATA_HETERO_STAGE0_PHASE_PREFIX
unset STRATA_HETERO_L0_PROBE_PREFIX STRATA_HETERO_LAYER_TAP_PREFIX
mkdir -p /home/chris/tmp/strata-heterogeneous/m216
OUT=/home/chris/tmp/strata-heterogeneous/m216
SMOKE="$PWD/build-hetero-smoke-cuda/experimental/heterogeneous/strata_heterogeneous_live_smoke"
CUDA_WORKER="$PWD/build-hetero-smoke-cuda/experimental/heterogeneous/strata_heterogeneous_cuda_worker"
HIP_WORKER="$PWD/build-hetero-smoke-hip/experimental/heterogeneous/strata_heterogeneous_hip_worker"

common=(--layers 48 --hc 4 --n-embd 2560 --fingerprint 2026111301
        --rows 1 --windows 2 --tokens 1,1 --timeout-ms 120000
        --branch-probe --stage0-phases --layer-taps --layer0-probe)
ref=(--plan hip:0:0-48 --hip-worker "$HIP_WORKER"
     --config "$PWD/build-q2-stage-configs/hip-reference-full-m210.cfg")
het=(--plan cuda:0:0-12,hip:0:12-30,hip:1:30-48
     --cuda-worker "$CUDA_WORKER" --hip-worker "$HIP_WORKER"
     --config "$PWD/build-q2-stage-configs/cuda-stage0.cfg"
     --config "$PWD/build-q2-stage-configs/hip-stage1.cfg"
     --config "$PWD/build-q2-stage-configs/hip-stage2.cfg")

run_arm() {
  local label="$1"; shift
  local rp="$OUT/$label-ref" hp="$OUT/$label-het"
  env STRATA_HC_Q8=0 "$@" "$SMOKE" "${common[@]}" "${ref[@]}" --capture-prefix "$rp"
  env STRATA_HC_Q8=0 "$@" "$SMOKE" "${common[@]}" "${het[@]}" --capture-prefix "$hp"
  env STRATA_HC_Q8=0 "$@" "$SMOKE" "${common[@]}" "${ref[@]}" \
    --capture-prefix "$rp" --run 2>&1 | tee "$rp.log"
  env STRATA_HC_Q8=0 "$@" "$SMOKE" "${common[@]}" "${het[@]}" \
    --capture-prefix "$hp" --run 2>&1 | tee "$hp.log"
  grep -F 'STRATA_GR0_PATH down_q8=0 up_q8=0' "$rp.log"
  grep -F 'STRATA_GR0_PATH down_q8=0 up_q8=0' "$hp.log"
  python3 tools/heterogeneous_stage_branch.py \
    --reference-prefix "$rp" --heterogeneous-prefix "$hp" \
    --report "$OUT/$label-branch.json"
  python3 tools/heterogeneous_stage0_phases.py \
    --reference-prefix "$rp" --heterogeneous-prefix "$hp" \
    --relative-threshold 0.001 --report "$OUT/$label-phases.json"
  local rc=0
  python3 tools/heterogeneous_compare_parity.py \
    --reference-prefix "$rp" --heterogeneous-prefix "$hp" \
    --max-kl 0.001 --report "$OUT/$label-parity.json" || rc=$?
  if [[ "$rc" -eq 1 ]]; then
    echo "Head numerical parity still FAIL for $label"
  elif [[ "$rc" -ne 0 ]]; then
    return "$rc"
  fi
}

# Required: align shared-expert stream scheduling only.
run_arm shared-inline STRATA_SH_STREAM=0

# Inspect this arm before running anything more:
python3 - "$OUT/shared-inline-branch.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1],encoding="utf-8"))
for p in r["positions"]:
    m=p["components"]
    print("position",p["position"],"ids_match",p["route_ids_match"],
          "shared_relL2",m["layer5_shared"]["relative_l2"],
          "expert_parts_relL2",m["layer5_expert_parts"]["relative_l2"],
          "conv_relL2",m["layer6_conv"]["relative_l2"],
          "gdn_bo_relL2",m["layer6_gdn_bo"]["relative_l2"])
PY
~~~

**STOP and review Arm A.** If layer5 shared output has converged
toward numerical noise, do not run another arm. If it still differs
materially, perform ONE conditional Arm B in the same shell:

~~~bash
run_arm shared-unfused STRATA_SH_STREAM=0 STRATA_FUSED_SWIGLU_Q81=0
~~~

All runs sequential, fresh output prefixes and socket transport.
Stop on compiler, identity, OOM, nonfinite, missing capture, or worker
errors. Do not reset GPUs or run four-window/stress tests.

## Interpret and report

1. Arm A dramatically lowers layer5 shared relL2: shared-stream
   scheduling / event ordering is implicated; not a proven race.
2. Arm A does not, but Arm B does: fused SwiGLU/Q8_1 behavior is
   implicated. Next capture exact gate/up and quantized activation
   inputs, not a speculative precision patch.
3. Neither lowers the shared difference: inspect shared gate/down
   arithmetic with identical weights/inputs.
4. If position1 QKV and z still match, but convolution differs,
   next add **pre-convolution history-state snapshots** before
   gdn_conv_l2_multi; STB0 alone cannot distinguish history drift
   from conv arithmetic. Do not call this a conv kernel bug.

Report per-position exact route IDs, shared/expert/conv/GDN metrics,
phase crossings, full-head KL, GPU PCI identities, test/build status,
logs and raw STB0/STP0/STFR captures. Numerical parity remains FAIL
until original KL <= 0.001 and top-1/prediction criteria pass.
