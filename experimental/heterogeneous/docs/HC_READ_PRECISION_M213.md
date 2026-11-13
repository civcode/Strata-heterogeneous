# M2.13 — Native HC read precision and dispatch isolation

**Baseline:** M2.12 showed exact embedding agreement and exact
HIP-split-versus-unsplit agreement. CUDA stage zero first differs
at `hc_mixed` (relative L2 0.006630, RMSE 0.004182, max
absolute 0.143806 at positions 0/1), before Q8_1 and QKV.
Full M2.10 head KL remains FAIL: 0.003376 / 0.039019 > 0.001.
Do not change the threshold or claim numerical parity.

## Source findings

- `src/core/native_dense.cpp::hc_q8_env()`: HIP gfx12 defaults
  `STRATA_HC_Q8` to **on**, loading Q8_0 HC projection copies.
  CUDA defaults to **off**, using BF16 HC projection rows.
- `src/kernels/cuda/fused_gr.cu::fused_gr_read_multi_slice`:
  Q8_0 selects `launch_q8`, with a different summation order from
  the BF16 read. Under BF16, the gfx12 fast norm is default-on
  (`STRATA_GR_NORM_FAST`); CUDA's is default-off. The selected
  split/staged GR variant may also differ.
- M2.10 tried `STRATA_HC_Q8=0` and **final-logit KL still failed**,
  but no layer-zero STL0 capture was made for that mode. We cannot
  conclude the original HC-read discrepancy survived that override.

M2.13 changes **no kernel arithmetic**. With manually enabled
`--layer0-probe` the verifier prints a `STRATA_GR0_PATH` line
showing actual Q8-down/up pointers, scale availability, qfuse and
HC/GR environment variables. A CPU-only two-run comparator
(`tools/heterogeneous_hc_mode_compare.py`) validates complete,
checksummed M2.12 STL0/STFR artifacts and compares the exact
embedding, HC-mixed, raw Q8_1 input and QKV projections. It rejects
misaligned model, pack, profile, token, stage or position identities.

## Scope

Only work under `/home/chris/workspace/Strata-heterogeneous` and
`/home/chris/tmp/strata-heterogeneous`. Do not touch either
Strata-dev directory. No GitHub Actions, GPU resets/reboots,
driver unload or disruptive recovery. Local builds <=28 jobs,
isolated CUDA and HIP worker processes. Keep all model,
memory reserve, finite-value and positive-prefix commit gates.

## 1. Sync, CPU tests, build

~~~bash
cd /home/chris/workspace/Strata-heterogeneous
git switch feature/heterogeneous-cuda-window-admission
git pull --ff-only
git status --short
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

## 2. Live control A — matching BF16 HC weights

Use *two fresh, sequential* two-window processes under the **same**
environment. The previous reference config and heterogeneous stage
configs are reused unchanged. Stage zero is R9700 HIP0 vs RTX CUDA0.

~~~bash
cd /home/chris/workspace/Strata-heterogeneous
set -euo pipefail
unset STRATA_HC_Q8 STRATA_HC_SPLIT STRATA_GR_NORM_FAST STRATA_GR_V3
unset STRATA_GR_FAST STRATA_GR_UP_WIDE STRATA_HETERO_L0_PROBE_PREFIX STRATA_TAP_FILE
mkdir -p /home/chris/tmp/strata-heterogeneous/m213
OUT=/home/chris/tmp/strata-heterogeneous/m213
SMOKE="$PWD/build-hetero-smoke-cuda/experimental/heterogeneous/strata_heterogeneous_live_smoke"
CUDA_WORKER="$PWD/build-hetero-smoke-cuda/experimental/heterogeneous/strata_heterogeneous_cuda_worker"
HIP_WORKER="$PWD/build-hetero-smoke-hip/experimental/heterogeneous/strata_heterogeneous_hip_worker"

common=(--layers 48 --hc 4 --n-embd 2560 --fingerprint 2026111301
        --rows 1 --windows 2 --tokens 1,1 --timeout-ms 120000
        --layer0-probe --layer-taps)
ref=(--plan hip:0:0-48 --hip-worker "$HIP_WORKER"
     --config "$PWD/build-q2-stage-configs/hip-reference-full-m210.cfg")
het=(--plan cuda:0:0-12,hip:0:12-30,hip:1:30-48
     --cuda-worker "$CUDA_WORKER" --hip-worker "$HIP_WORKER"
     --config "$PWD/build-q2-stage-configs/cuda-stage0.cfg"
     --config "$PWD/build-q2-stage-configs/hip-stage1.cfg"
     --config "$PWD/build-q2-stage-configs/hip-stage2.cfg")

# Reusable local function; prefix must be fresh for every arm.
run_arm() {
  local label="$1"; shift
  local ref_out="$OUT/$label-ref"
  local het_out="$OUT/$label-het"
  env "$@" "$SMOKE" "${common[@]}" "${ref[@]}" --capture-prefix "$ref_out"
  env "$@" "$SMOKE" "${common[@]}" "${het[@]}" --capture-prefix "$het_out"
  env "$@" "$SMOKE" "${common[@]}" "${ref[@]}" --capture-prefix "$ref_out" --run \
    2>&1 | tee "$ref_out.log"
  env "$@" "$SMOKE" "${common[@]}" "${het[@]}" --capture-prefix "$het_out" --run \
    2>&1 | tee "$het_out.log"

  # Verify actual model weight formats, not just requested env variables.
  grep -F 'STRATA_GR0_PATH down_q8=0 up_q8=0' "$ref_out.log"
  grep -F 'STRATA_GR0_PATH down_q8=0 up_q8=0' "$het_out.log"

  python3 tools/heterogeneous_hc_mode_compare.py \
    --reference-prefix "$ref_out" --heterogeneous-prefix "$het_out" \
    --arm "$label" --report "$OUT/$label-hc.json"

  # Original M2.10 full-logit parity remains independent; FAIL exit 1
  # is valid evidence; any integrity/error exit 2 must stop the test.
  local rc=0
  python3 tools/heterogeneous_compare_parity.py \
    --reference-prefix "$ref_out" --heterogeneous-prefix "$het_out" \
    --max-kl 0.001 --report "$OUT/$label-parity.json" || rc=$?
  if [[ "$rc" -eq 1 ]]; then
    echo "M2.10 full-head KL still fails for $label"
  elif [[ "$rc" -ne 0 ]]; then
    return "$rc"
  fi
}

run_arm bf16 STRATA_HC_Q8=0
~~~

**STOP and inspect** `$OUT/bf16-hc.json`: if the layer-zero
`hc_mixed` output matches exactly (or falls to near numerical noise)
with Q8 disabled, the original first discrepancy is attributable
primarily to **unequal Q8 vs BF16 projection paths**, not yet proof of
a bug. The full-head KL may still fail due to later divergence.
Record the STLT first divergent layer under the BF16 arm before
investigating further. Do not run the next control unnecessarily.

## 3. Live control B — only if Arm A HC still differs

Force the plain BF16 GR path and disable architecture-selected fast
normalization (and other GR optimizations) for both workers, while
keeping `STRATA_HC_Q8=0`. This differentiates kernel-variant effects
from a residual CUDA/HIP arithmetic discrepancy.

~~~bash
# In the SAME configured shell with run_arm defined above:
run_arm bf16-plain STRATA_HC_Q8=0 \
  STRATA_GR_NORM_FAST=0 STRATA_HC_SPLIT=0 STRATA_GR_V3=0 \
  STRATA_GR_FAST=0 STRATA_GR_UP_WIDE=0
~~~

Check `STRATA_GR0_PATH` and `strata hc:` logs for the actually
selected mode; the source's default-path details are architecture
dependent. If B converges while A differs, isolate the changing
GR flag in a subsequent one-variable-at-a-time test. If both still
differ, the next increment should capture HC normalization and
down/up subprojection tensors, not assume a particular kernel bug.

## Report

- First different STL0 component and HC relative L2, RMSE, max abs
  for each position and executed arm.
- Actual `STRATA_GR0_PATH` flags from HIP0 and CUDA0.
- Final-head KL results, including whether they remain >0.001.
- The first STLT layer divergence under the new matched modes if
  it can be computed without extra runs.
- Device PCI identities, build/toolchain identifiers, exact command
  exits, logs and raw STL0/STLT/STFR/manifest files.

The default M2.12 HC relative L2 0.006630 is the starting baseline.
The M2.10 KL<=0.001 acceptance threshold is **never relaxed**.
M2.13 local CPU evidence is 8/8 new tests plus 42/42 earlier
M2.10–12 fixtures, all passed. **M2.13 live hardware results
have not been executed in this ChatGPT environment.**
