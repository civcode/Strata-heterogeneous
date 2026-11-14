# M2.9 — Native CUDA→HIP hardware-test handoff

> **Historical milestone snapshot.** Early CPU-only/unverified native status
> describes this milestone, not the later successful native GPU windows.
> Follow the [current CUDA → HIP implementation contract](IMPLEMENTATION_CONTRACT.md).


**Branch:** feature/heterogeneous-cuda-window-admission
**Target:** Oberon; RTX 4070 Ti SUPER 16 GiB and two Radeon AI PRO R9700 32 GiB.
**Stage plan:** cuda:0:0-12,hip:0:12-30,hip:1:30-48
**Model:** canonical 48-layer Q2_0, HC=4, embedding=2560, vocabulary=248320, fingerprint=2026111301.
**Index SHA-256:** c344c01f59f547ed8a59003857a82eb90499e5299154241dbca787acb54b5b7c
**Status:** code and manual run instructions prepared; **no live GPU test performed by this handoff**.

## Scope

The operator or agent explicitly invoking a live test authorizes CUDA/HIP runtime
initialization, enumeration, native Q2_0 model loading and inference for that test.
No additional I_APPROVE_* environment-variable ritual is required. The launcher
without --run performs CPU-only preflight; with --run, it starts native GPU workers.
Do not run Actions, reset or reboot GPUs, unload drivers or attempt recovery by default.
Use at most 28 parallel build jobs and do not access either
/home/chris/workspace/Strata-dev or /home/chris/tmp/strata-dev. Use only
/home/chris/workspace/Strata-heterogeneous and separate build directories.

The launcher is a CPU-only coordinator linking the host protocol library, **not**
CUDA and HIP simultaneously. CUDA, HIP0 and HIP1 workers are exec-isolated
processes. The first stage ignores the zero-filled *placeholder* activation and
embeds supplied token IDs using the actual model; later stages receive real
output activations from previous workers. The final stage returns STFR/v1
predictions and all workers must acknowledge a positive-prefix commit.

## Step 1: sync and test CPU-only source

~~~bash
cd /home/chris/workspace/Strata-heterogeneous
git switch feature/heterogeneous-cuda-window-admission
git pull --ff-only
git status --short
python3 -m unittest discover -s tools -p 'test_heterogeneous_model_gate.py' -v
python3 -m unittest discover -s tools -p 'test_heterogeneous_hardware_readiness.py' -v
python3 -m unittest discover -s tools -p 'test_heterogeneous_ordinal_reconciliation.py' -v
cmake -S experimental/heterogeneous -B build-hetero-smoke-host
cmake --build build-hetero-smoke-host --parallel 28
ctest --test-dir build-hetero-smoke-host --output-on-failure
~~~

Check real Q2_0 assets with the M2.6 model checker. Its extra
--preflight executable is optional now:

~~~bash
python3 tools/heterogeneous_model_gate.py \
  --plan cuda:0:0-12,hip:0:12-30,hip:1:30-48 \
  --layers 48 --hc 4 --n-embd 2560 --fingerprint 2026111301 \
  --config "$PWD/build-q2-stage-configs/cuda-stage0.cfg" \
  --config "$PWD/build-q2-stage-configs/hip-stage1.cfg" \
  --config "$PWD/build-q2-stage-configs/hip-stage2.cfg"
~~~

Missing optional PCI declarations and missing offline M2.8 ordinal-identity
evidence are NOT GPU hardware failures. Runtime device identity is reported
directly by the native worker at startup.

## Step 2: native builds (no GPU execution)

Reuse existing native build trees only if they match the current feature SHA;
otherwise configure separate CUDA-only and HIP-only trees:

~~~bash
cmake -S . -B build-hetero-smoke-cuda \
  -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_BUILD_HETERO_CUDA_WORKER=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-hetero-smoke-cuda --parallel 28 \
  --target strata_heterogeneous_cuda_worker strata_heterogeneous_live_smoke

cmake -S . -B build-hetero-smoke-hip \
  -DSTRATA_ENABLE_HIP=ON \
  -DSTRATA_BUILD_HETERO_HIP_WORKER=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201
cmake --build build-hetero-smoke-hip --parallel 28 \
  --target strata_heterogeneous_hip_worker
~~~

If the existing Oberon toolchain needs different architecture flags, record
the exact flags used and validate locally; never dispatch GitHub Actions.

## Step 3: manually initiated hardware tests

~~~bash
cd /home/chris/workspace/Strata-heterogeneous
CUDA_WORKER="$PWD/build-hetero-smoke-cuda/experimental/heterogeneous/strata_heterogeneous_cuda_worker"
HIP_WORKER="$PWD/build-hetero-smoke-hip/experimental/heterogeneous/strata_heterogeneous_hip_worker"
SMOKE="$PWD/build-hetero-smoke-cuda/experimental/heterogeneous/strata_heterogeneous_live_smoke"

smoke_args=(
  --plan cuda:0:0-12,hip:0:12-30,hip:1:30-48
  --layers 48 --hc 4 --n-embd 2560 --fingerprint 2026111301
  --cuda-worker "$CUDA_WORKER" --hip-worker "$HIP_WORKER"
  --config "$PWD/build-q2-stage-configs/cuda-stage0.cfg"
  --config "$PWD/build-q2-stage-configs/hip-stage1.cfg"
  --config "$PWD/build-q2-stage-configs/hip-stage2.cfg"
  --token 1 --rows 1 --timeout-ms 120000
)

# CPU-only stage config preflight. Never initializes a GPU.
"$SMOKE" "${smoke_args[@]}"

# Live test A: one actual token, CUDA→HIP→HIP socket handoff and STFR commit.
"$SMOKE" "${smoke_args[@]}" --run

# Live test B: new workers, same request using shared-memory activation transport.
"$SMOKE" "${smoke_args[@]}" --shared --run

# Live test C: new workers, two successive token windows at positions 0 and 1.
"$SMOKE" "${smoke_args[@]}" --windows 2 --run
~~~

Run A first; do NOT continue to B if A fails or to C if B fails. Each --run
is manually invoked. There are no automatic Actions, CTest hardware jobs,
or scheduled probes.

## Required runtime inspection and stop conditions

Every native stage-resource owner logs one stderr line in this form:

~~~text
STRATA_DEVICE backend=CUDA|HIP stage=<index> ordinal=<ordinal> runtime_visible=<count> name=<runtime name> pci=<BDF or unknown> free_mib=<n> total_mib=<n>
~~~

Confirm CUDA stage 0 selects the NVIDIA GPU and HIP stages 1 and 2 select
**two distinct R9700 PCI functions**. Expected physical addresses:
NVIDIA 0000:04:00.0; intended AMD 0000:03:00.0 and 0000:07:00.0.
The extra AMD 0000:7e:00.0 is not an intended R9700. Actual runtime
ordinals can be reordered or filtered by device visibility variables.
If the wrong device is selected, stop and correct GPU visibility / plan
ordinals before any further model load.

Immediately stop and preserve diagnostics on any driver/runtime error,
wrong model/GPU, missing or mismatched GGUF/PLE/STRP/index, free-memory
or reserve failure, nonfinite activation or head output, worker timeout,
lost stage identity, missing prefix ACK, or unexpected process termination.
Do not turn errors into warnings or reboot/reset the machine. Capture exact
build/launch commands, git SHA, STRATA_DEVICE lines, first failure and
worker stderr, and any relevant driver fault markers.

A PASS proves only one- or two-window native startup, stage handoff,
STFR generation and acknowledged positive-prefix commits. It does not
prove numerical parity to unsplit Strata, long-context stability, prompt
prefill, zero-prefix rollback, sampler integration, recovery or throughput.
After A–C pass, compare outputs to the same unsplit model on matching
tokens and positions before claiming numerical correctness.
