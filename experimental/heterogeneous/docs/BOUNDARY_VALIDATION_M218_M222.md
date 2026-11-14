# M2.18–M2.22: real CUDA → HIP architecture validation

**Status: prepared, native hardware UNVERIFIED on this commit.** The [active contract](IMPLEMENTATION_CONTRACT.md) is authoritative. No new inference engine, mathematical reference or fixture data is introduced.

## M2.18: audit of existing native production code

| Requirement | Existing code path | Evidence |
| --- | --- | --- |
| Native inference | CUDA and HIP non-final drivers and HIP final worker invoke existing `core::Verifier` | SOURCE AUDITED |
| Handoff layout | `R | bo | inject` float32, `Geometry::floats_per_row()` | SOURCE AUDITED |
| CUDA producer completion | `enqueue_download` → event fence → `wait_download` → stage reply | SOURCE AUDITED |
| Real transport | `FinalPipelineSession` forwards checked socket or memfd activation frames | SOURCE AUDITED |
| HIP consumer completion | `enqueue_upload` → event fence → `wait_upload` → `Verifier::run` | SOURCE AUDITED |
| State and commit | Stage-local state, `WindowAdmission`, positive-prefix ACK from each worker | SOURCE AUDITED |
| Whole pipeline | Existing CUDA 0–11 → HIP 12–29 → HIP 30–47 native workers, final STFR/v1 head | PAST SHORT RUN REPORTED; NEW RUN UNVERIFIED |

These audit results are **not GPU PASS claims**.

## M2.19: real-device evidence with minimal code

The existing `strata_heterogeneous_live_smoke` accepts `--boundary-capture` together with `--capture-prefix`. The actual native worker, **after existing stream fences**, reads its device output/input buffer back, compares each byte to the pinned transfer buffer, and writes the raw bytes to an exclusive 0600 file. Readback is **before** receiver arithmetic or **after** producer arithmetic. A mismatch fails the window. Disabled: no additional copies or files. No inference arithmetic, transport format, kernel or standalone test executable changed.

With request 1 and window 1, the real device files are `PREFIX.stage0.r1.w1.out.f32`, `PREFIX.stage1.r1.w1.in.f32`, `PREFIX.stage1.r1.w1.out.f32`, and `PREFIX.stage2.r1.w1.in.f32`. For window 2 substitute `w2`. With one row and the target geometry each contains **51,216 bytes**. Use `cmp` of raw producer/consumer device images, not checksums or floating-point tolerances. Require valid final STFR files, completed manifests and stage commits.

## Manual Oberon run (do not run GitHub Actions)

Use only `/home/chris/workspace/Strata-heterogeneous` and `/home/chris/tmp/strata-heterogeneous`; never touch either Strata-dev tree. Keep separate CUDA/HIP builds, ≤28 jobs, model provenance, PCI checks, VRAM reserves, positive-prefix gates and existing nonfinite guards. No GPU reset/reboot/driver unload.

~~~bash
set -euo pipefail
cd /home/chris/workspace/Strata-heterogeneous
git switch feature/heterogeneous-cuda-window-admission
git pull --ff-only
git status --short
cmake --build build-hetero-smoke-host --parallel 28
ctest --test-dir build-hetero-smoke-host --output-on-failure
cmake --build build-hetero-smoke-cuda --parallel 28 --target strata_heterogeneous_cuda_worker strata_heterogeneous_live_smoke
cmake --build build-hetero-smoke-hip --parallel 28 --target strata_heterogeneous_hip_worker
OUT=/home/chris/tmp/strata-heterogeneous/m218
mkdir -p "$OUT"
SMOKE="$PWD/build-hetero-smoke-cuda/experimental/heterogeneous/strata_heterogeneous_live_smoke"
CUDA="$PWD/build-hetero-smoke-cuda/experimental/heterogeneous/strata_heterogeneous_cuda_worker"
HIP="$PWD/build-hetero-smoke-hip/experimental/heterogeneous/strata_heterogeneous_hip_worker"
common=(--layers 48 --hc 4 --n-embd 2560 --fingerprint 2026111301 --rows 1 --windows 2 --tokens 1,1 --timeout-ms 120000)
mixed=(--plan cuda:0:0-12,hip:0:12-30,hip:1:30-48 --cuda-worker "$CUDA" --hip-worker "$HIP" --config "$PWD/build-q2-stage-configs/cuda-stage0.cfg" --config "$PWD/build-q2-stage-configs/hip-stage1.cfg" --config "$PWD/build-q2-stage-configs/hip-stage2.cfg")

# CPU-only preflight, followed by real native execution.
"$SMOKE" "${common[@]}" "${mixed[@]}" --boundary-capture --capture-prefix "$OUT/socket"
env STRATA_HETERO_WINDOW_TRACE=1 "$SMOKE" "${common[@]}" "${mixed[@]}" --boundary-capture --capture-prefix "$OUT/socket" --run 2>&1 | tee "$OUT/socket.log"
cmp "$OUT/socket.stage0.r1.w1.out.f32" "$OUT/socket.stage1.r1.w1.in.f32"
cmp "$OUT/socket.stage1.r1.w1.out.f32" "$OUT/socket.stage2.r1.w1.in.f32"
cmp "$OUT/socket.stage0.r1.w2.out.f32" "$OUT/socket.stage1.r1.w2.in.f32"
cmp "$OUT/socket.stage1.r1.w2.out.f32" "$OUT/socket.stage2.r1.w2.in.f32"
for w in 1 2; do
  for name in stage0.r1.w${w}.out stage1.r1.w${w}.in stage1.r1.w${w}.out stage2.r1.w${w}.in; do
    test "$(wc -c < "$OUT/socket.${name}.f32")" -eq 51216
  done
done
test -s "$OUT/socket.complete.json"
~~~

Stop on any failure, and never overwrite a used output prefix. The existing
`STRATA_HETERO_WINDOW_TRACE=1` switch logs stage admission, provisional
execution and committed prefix metadata from the real workers; compare its
per-stage request/window/position/commit events to the final STFR record. After confirming socket success, repeat the **same native run** with `--shared` and a fresh `$OUT/shared` prefix; compare both producer/consumer pairs for both windows. Retain logs, device PCI identifiers, model/index/GGUF/profile provenance, raw bytes and completion/ACK evidence.

## M2.20–M2.22: remaining gates

**M2.20 (state/commits):** verify real window positions, state ownership, every worker's accepted-prefix ACK and fail-stop behavior. Keep existing CPU protocol error tests; no numeric fixtures.

**M2.21 (end-to-end):** verify both transports, finite final head logits, valid predictions, correct GPU identities and consistent repeated fresh runs. Two windows do not establish prefill/long-context or general production readiness.

**M2.22 (consolidation):** keep the reusable opt-in boundary observation if it proves valuable; avoid new engine code, disposable test executables or speculative kernel cleanup. Report **PASS / FAIL / UNVERIFIED** per requirement.

The historical all-HIP KL ≤0.001 remains a **non-gating numerical parity diagnostic**. Preserve previously observed FAIL values.
