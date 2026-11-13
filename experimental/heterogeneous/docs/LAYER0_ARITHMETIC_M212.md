# M2.12 — Exact layer-zero arithmetic: HIP control vs CUDA stage zero

**Status:** Implementation committed; native M2.12 GPU diagnosis NOT YET EXECUTED.
**Branch:** feature/heterogeneous-cuda-window-admission
**M2.11 known fact:** the first observed post-layer HC-mean residual divergence
is layer 0 (relative L2 0.011669 / 0.012537 at positions 0 / 1).
Both models still predict 198 / 271; M2.10 head KL stays
0.003376 / 0.039019 > 0.001. Turning STRATA_HC_Q8=0 on both
does not resolve it (KL 0.005003 / 0.037718).

The objective is to distinguish the first different **actual stage-zero
operation**, not to change the M2.10 criterion or patch unconfirmed kernels.

## Architecture of the control

Three **sequential, fresh-session** runs, identical model sources and input
tokens at positions 0 and 1:

1. Full HIP reference: hip:0:0-48 (R9700 at PCI 0000:03:00.0).
2. Same-backend HIP split: hip:0:0-12,hip:1:12-48
   (stage zero on that same first R9700; stage one on PCI 0000:07:00.0).
3. Current mixed pipeline: cuda:0:0-12,hip:0:12-30,hip:1:30-48
   (stage zero on RTX 4070 Ti SUPER at PCI 0000:04:00.0).

The canonical Q2_0 shards, pack index SHA, profile, model fingerprint and
token IDs must match across all three. Compare HIP control against full HIP
**before** attributing CUDA differences to arithmetic; an all-HIP split
may expose stage initialization or resource strategy differences.

Native core::Verifier now has an opt-in layer-zero diagnostic, enabled
ONLY when supervised manual launcher --layer0-probe is requested.
It captures, for T=1:

- **embedding:** decoded FP32 token embedding, before HC broadcasting
- **hc_mixed:** FP32 output of layer-0 hyper-connection read before GDN
  Q8_1 quantization
- **q8_input:** exact raw Q8_1 bytes consumed by the native GDN
  attn_qkv projection, including quantized activations and block metadata
- **gdn_qkv:** actual native quantized QKV projection FP32 output before
  GDN convolution, gate/recurrent state and MoE computations

The stage-0 worker writes one exclusive STL0/v1 binary record per window,
PREFIX.stage0.wN.stl0, with a 72-byte versioned header, window/plan
identity, checksum, input token, 2560 embedding floats, 2560 mixed floats,
the Q8_1 input buffer, and the complete layer-0 QKV projection.
No debug device buffers or work are added without the opt-in flag.
The original M2.11 --layer-taps remains independently optional, enabling
comparison against the completed post-layer residual (STLT/v1).

**Interpretation:** if embedding differs, investigate native GGUF embedding
decode; if embedding matches but HC-mixed differs, investigate the
native BF16 GR/norm/down/up/read and its input. If HC-mixed matches but
Q8 bytes differ, investigate native Q8_1 quantization; if Q8 bytes match
but QKV floats differ, isolate native quantized attn_qkv MMVQ/weight
and accumulation. If all four are identical but layer-0 STLT diverges,
the first mismatch lies **later** inside the GDN/FFN/MoE layer.
Exact component equality is reported separately from relative L2; do
not label a tiny one-ULP difference equivalent to the observed 1% drift.

## Scope and safety

Only /home/chris/workspace/Strata-heterogeneous for repository work and
/home/chris/tmp/strata-heterogeneous for logs. **Never touch** either
Strata-dev directory. Never reset/reboot GPUs or unload drivers. Never run
GitHub Actions. Local compilation and CTest only (maximum 28 jobs).
User-requested native GPU diagnostics here are authorized; no redundant
runtime approval environment flags. Do not run unrequested stress tests.

Native CUDA and HIP workers remain **separate processes/builds**.
Use the tested ROCm environment and build RPATH /opt/rocm/lib.
Do not disable model provenance, native GPU VRAM reserve, nonfinite
checks, STFR identity or accepted-prefix validation.

## Sync and CPU tests

~~~bash
cd /home/chris/workspace/Strata-heterogeneous
git switch feature/heterogeneous-cuda-window-admission
git pull --ff-only
git status --short

python3 -m unittest discover -s tools -p test_heterogeneous_layer0_arithmetic.py -v
python3 -m unittest discover -s tools -p test_heterogeneous_layer_divergence.py -v
python3 -m unittest discover -s tools -p test_heterogeneous_compare_parity.py -v
cmake --build build-hetero-smoke-host --parallel 28
ctest --test-dir build-hetero-smoke-host --output-on-failure
~~~

**Important:** the new source introduces a core Verifier method and the
worker-side recorder. Compile the native workers rather than treating
the Python fixtures as proof of GPU integration. The CPU fake-verifier
headers have fail-closed signatures. Fix any real compilation error
and rerun the tests before starting the GPUs.

## Native rebuild

~~~bash
cd /home/chris/workspace/Strata-heterogeneous
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

No MMQ prefill is required for these verifier windows.

## HIP-only control configs (CPU-only)

The generator reads the existing CUDA stage-0 and HIP stage-1 v2 configs
and validates canonical GGUF shards/pack/profile/model metadata.
It writes **new** stage0 [0,12) HIP0 and stage1 [12,48) HIP1 configs.
It resets stale artificial stage-specific arena budgets to zero, but
native GPU free-memory and reserve checks still apply.
Never overwrite original configs or modify them by hand.

~~~bash
python3 tools/heterogeneous_hip_control.py \
  --source-first "$PWD/build-q2-stage-configs/cuda-stage0.cfg" \
  --source-rest "$PWD/build-q2-stage-configs/hip-stage1.cfg" \
  --output-first "$PWD/build-q2-stage-configs/hip-control0-m212.cfg" \
  --output-rest "$PWD/build-q2-stage-configs/hip-control1-m212.cfg" \
  --hip-first-ordinal 0 --hip-rest-ordinal 1
~~~

If generated files already exist, verify against the current source;
do not overwrite. The model loader remains the canonical source of
truth for stage/PLE/weight validation.

## Three exact two-window GPU runs

Use one clean shell, no inherited Q8 precision override, with a fresh
capture directory. One set of workers at a time; abort subsequent runs
on a genuine hardware failure (set -e and pipefail enforce this).

~~~bash
cd /home/chris/workspace/Strata-heterogeneous
set -euo pipefail
unset STRATA_HC_Q8 STRATA_HETERO_L0_PROBE_PREFIX STRATA_TAP_FILE
mkdir -p /home/chris/tmp/strata-heterogeneous/m212
OUT=/home/chris/tmp/strata-heterogeneous/m212
SMOKE="$PWD/build-hetero-smoke-cuda/experimental/heterogeneous/strata_heterogeneous_live_smoke"
CUDA_WORKER="$PWD/build-hetero-smoke-cuda/experimental/heterogeneous/strata_heterogeneous_cuda_worker"
HIP_WORKER="$PWD/build-hetero-smoke-hip/experimental/heterogeneous/strata_heterogeneous_hip_worker"

common=(--layers 48 --hc 4 --n-embd 2560 --fingerprint 2026111301
        --rows 1 --windows 2 --tokens 1,1 --timeout-ms 120000
        --layer0-probe --layer-taps)
ref=(--plan hip:0:0-48 --hip-worker "$HIP_WORKER"
     --config "$PWD/build-q2-stage-configs/hip-reference-full-m210.cfg")
control=(--plan hip:0:0-12,hip:1:12-48 --hip-worker "$HIP_WORKER"
         --config "$PWD/build-q2-stage-configs/hip-control0-m212.cfg"
         --config "$PWD/build-q2-stage-configs/hip-control1-m212.cfg")
hetero=(--plan cuda:0:0-12,hip:0:12-30,hip:1:30-48
        --cuda-worker "$CUDA_WORKER" --hip-worker "$HIP_WORKER"
        --config "$PWD/build-q2-stage-configs/cuda-stage0.cfg"
        --config "$PWD/build-q2-stage-configs/hip-stage1.cfg"
        --config "$PWD/build-q2-stage-configs/hip-stage2.cfg")

# CPU-only plan, source and path validation (no workers/GPU runtime):
"$SMOKE" "${common[@]}" "${ref[@]}" --capture-prefix "$OUT/reference"
"$SMOKE" "${common[@]}" "${control[@]}" --capture-prefix "$OUT/hip-control"
"$SMOKE" "${common[@]}" "${hetero[@]}" --capture-prefix "$OUT/hetero"

# Deliberate, ordered native GPU runs; --run starts isolated workers:
"$SMOKE" "${common[@]}" "${ref[@]}" \
  --capture-prefix "$OUT/reference" --run 2>&1 | tee "$OUT/reference.log"
"$SMOKE" "${common[@]}" "${control[@]}" \
  --capture-prefix "$OUT/hip-control" --run 2>&1 | tee "$OUT/hip-control.log"
"$SMOKE" "${common[@]}" "${hetero[@]}" \
  --capture-prefix "$OUT/hetero" --run 2>&1 | tee "$OUT/hetero.log"

# Strict CPU-only full-data provenance and arithmetic comparison:
python3 tools/heterogeneous_layer0_arithmetic.py \
  --reference-prefix "$OUT/reference" \
  --hip-control-prefix "$OUT/hip-control" \
  --heterogeneous-prefix "$OUT/hetero" \
  --report "$OUT/layer0-arithmetic.json"
~~~

The optional M2.11 residual tap in these same runs can corroborate
the layer-0 *output*, but STL0 identifies earlier suboperations.
M2.10 KL remains a separate numerical criterion; do not loosen it.

## Report and next decision

Return each position's **first nonidentical component** in:
(a) HIP control vs unsplit HIP, (b) CUDA vs unsplit HIP,
(c) CUDA vs the HIP control. Include embedding/mixed/QKV relative L2,
RMSE and max abs, Q8 mismatched-byte count, worker GPU PCI identity,
the exact native kernel and compiler/runtime precision settings used.

If the HIP control differs materially from the unsplit HIP reference
already at embedding or HC read, investigate the first-stage worker
startup or stage-dependent kernel paths before blaming CUDA. If the
HIP control matches but CUDA first differs in HC mixed, focus GR BF16
kernels; if only Q8 differs, focus quantization; if only QKV differs,
focus Q2_0 MMVQ and accumulation. If all STL0 components match, inspect
GDN recurrence/gates and MoE within layer zero.

**Stop at first confirmed cause candidate**; don't change kernels,
precision defaults or acceptance thresholds speculatively. Retain raw
STL0, STLT, STFR, manifests and all logs for repeatable analysis.

Passing input/probe checks does NOT establish M2.10 numerical parity.
Numerical parity remains FAIL until the full head-logit test meets its
original KL<=0.001 criterion at every requested position.


## Development-side validation evidence

The actual committed M2.12 Python source files were copied into the
isolated local workspace and verified byte-for-byte by Git blob SHA:

- `tools/heterogeneous_hip_control.py` — `40f1f0c6681714b5c578c96aa74b38e6c7a2f83f`
- `tools/heterogeneous_layer0_arithmetic.py` — `15bd57c7d822c066b04cbce2b8a5d36398a34c63`
- `tools/test_heterogeneous_layer0_arithmetic.py` — `bc28f8123fbe75c78a02706230b70a1bf7ab5210`

**42/42 local CPU synthetic tests pass:** M2.10 14, M2.11 13,
M2.12 15. All four Python comparator/generator files passed py_compile.

The exact committed STL0/v1 C++ recorder header (Git blob
`d7794ef4eb366500a169cb7704ef0599bfed11bc`) compiled against
an interface-compatible CPU fake Verifier using **both GCC and Clang
C++17** under `-Wall -Wextra -Wpedantic -Werror`. Both host-only
executables emitted real STL0/v1 files; the exact M2.12 Python parser
verified their checksum, token, dimensions and FP32/Q8 samples.
A second exclusive-create attempt failed closed as intended.

These tests validate the diagnostic protocol and the CPU comparison
contract, NOT GPU compilation, actual QKV tensor values, or numerical
parity. Full native CUDA/HIP compilation and the sequential three-run
hardware comparison are still pending on Oberon.
