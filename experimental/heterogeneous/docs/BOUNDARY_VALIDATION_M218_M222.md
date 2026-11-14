# M2.18–M2.22: real CUDA → HIP architecture validation

**Status:** M2.18 source audit PASS; M2.19 device-boundary PASS and M2.21 bounded end-to-end PASS on `5e1f868` (operator-reported Oberon runs). M2.20 partial-prefix state/live failure-path qualification and M2.22 final consolidation UNVERIFIED. The [active contract](IMPLEMENTATION_CONTRACT.md) is authoritative. No new inference engine, mathematical reference or fixture data is introduced.

## M2.18: audit of existing native production code

| Requirement | Existing code path | Evidence |
| --- | --- | --- |
| Native inference | CUDA and HIP non-final drivers and HIP final worker invoke existing `core::Verifier` | SOURCE AUDITED |
| Handoff layout | `R | bo | inject` float32, `Geometry::floats_per_row()` | SOURCE AUDITED |
| CUDA producer completion | `enqueue_download` → event fence → `wait_download` → stage reply | SOURCE AUDITED |
| Real transport | `FinalPipelineSession` forwards checked socket or memfd activation frames | SOURCE AUDITED |
| HIP consumer completion | `enqueue_upload` → event fence → `wait_upload` → `Verifier::run` | SOURCE AUDITED |
| State and commit | Stage-local state, `WindowAdmission`, positive-prefix ACK from each worker | SOURCE AUDITED |
| Whole pipeline | Existing CUDA 0–11 → HIP 12–29 → HIP 30–47 native workers, final STFR/v1 head | BOUNDED HARDWARE PASS REPORTED BELOW |

These audit results are **not GPU PASS claims**.

### First Oberon build attempt and correction

At `0f757ab`, the host build stopped before CTest, native builds or GPU execution: the four CUDA/HIP verifier contract-test targets could not compile the new synchronous `cudaMemcpy` call because the existing CPU runtime stub exposes only `cudaMemcpyAsync`. The worker capture now uses the same asynchronous D2H API, owned pinned staging buffer, verifier stream and bounded event wait as the production path. It snapshots the original host bytes and invalidates the staging bytes before readback so a skipped or partial copy cannot masquerade as equality. Publication requires a full byte match. No stub extension, new inference code, or new test fixture is needed. The corrected `5e1f868` build subsequently passed on Oberon as recorded below.

## Reported Oberon results: 5e1f868002296b06b335b50febc46fed4746942e

Host build, **38/38 CTests**, separate native CUDA/HIP builds, real socket and shared-memory runs and a fresh sequential socket repeat passed, per the operator report. The raw logs reside on Oberon and were not independently retrieved.

| Milestone | Result and scope |
| --- | --- |
| M2.18 | PASS, source audit |
| M2.19 | PASS, exact CUDA → HIP and HIP → HIP device endpoint matches for every tested window in all three runs |
| M2.20 | Full-window positive-prefix ACK and position advancement PASS; independent persistent-state/partial-prefix and native failure-path qualification UNVERIFIED |
| M2.21 | PASS, bounded two-window real model inference and fresh repeat; no prefill, long-context or general production-readiness claim |
| M2.22 | UNVERIFIED pending remaining qualification/consolidation |

- Plan: `cuda:0:0-12,hip:0:12-30,hip:1:30-48`, request 1, tokens 1 and 1 at positions 0 and 1. All workers admitted, executed and acknowledged keep=1. All three runs predicted 198 then 271.
- Every corresponding device readback pair matched bytewise (51,216 bytes per image; 12 pairs across three runs). Device-vs-pinned checks, STFR identity/checksum/manifests and 248,320 finite logits per window passed.
- Devices: RTX 4070 Ti SUPER `0000:04:00.0`; two R9700s `0000:03:00.0` and `0000:07:00.0`.
- Model/plan fingerprints 2026111301 / 14342964352252375683; pack-index SHA-256 c344c01f59f547ed8a59003857a82eb90499e5299154241dbca787acb54b5b7c. Source paths in manifests; model fingerprint does not attest GPU weight bytes.
- Artifacts: `/home/chris/tmp/strata-heterogeneous/m218-run-xUxChBl2/`. Host/CPU/native build logs, socket/shared/repeat logs and comparisons, STFR validation, *.stfr, *.json, *.f32 retained.
- No new KL run, source changes, GPU reset or driver changes. Existing pool probe untouched.

**Consequence:** the real CUDA → HIP boundary has passed exact device endpoint validation for this configuration. Do not reopen cross-backend kernel localization on this evidence.

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

## M2.20 follow-up: positive partial-prefix qualification

The existing native smoke runner now accepts optional `--keep N`, defaulting to `--rows`. It calls the unchanged commit interface, advances position by accepted count and records that count in each manifest. STFR retains every evaluated row; the live log publishes accepted predictions only. Existing parity comparisons still require full commits. For partial-prefix captures, explicitly use `read_run(prefix, allow_partial=True)`.

Rebuild/test the existing host runner, verify separate native builds and use the same approved worker configs, source, PCI identities and precision settings as the successful socket run. If preflight rejects two rows, stop; do not bypass limits or silently change configs. Use fresh sequential sessions. No new GPU probes, synthetic numeric fixtures or native fault injection.

The following three real executions evaluate two rows at position 0, accept just the first, then evaluate the identical two-row window at position 1. A/B vary only the discarded first-window token. A/C vary the accepted first-window token, providing a history-sensitivity control.

~~~bash
# Use the SMOKE and mixed array from the established runbook. Do not
# reuse common (it contains a different --tokens argument).
set -euo pipefail
OUT=$(mktemp -d /home/chris/tmp/strata-heterogeneous/m220-XXXXXX)
state_args=(--layers 48 --hc 4 --n-embd 2560 --fingerprint 2026111301
            --rows 2 --keep 1 --windows 2 --timeout-ms 120000)
for arm in a b c; do
  case "$arm" in
    a) tokens=1,271,198,271 ;;
    b) tokens=1,1,198,271 ;;
    c) tokens=198,271,198,271 ;;
  esac
  "$SMOKE" "${state_args[@]}" "${mixed[@]}" --tokens "$tokens" \
    --boundary-capture --capture-prefix "$OUT/$arm"
  env STRATA_HETERO_WINDOW_TRACE=1 "$SMOKE" "${state_args[@]}" "${mixed[@]}" \
    --tokens "$tokens" --boundary-capture --capture-prefix "$OUT/$arm" --run \
    2>&1 | tee "$OUT/$arm.log"
  for w in 1 2; do
    for name in stage0.r1.w${w}.out stage1.r1.w${w}.in \
                stage1.r1.w${w}.out stage2.r1.w${w}.in; do
      test "$(wc -c < "$OUT/$arm.$name.f32")" -eq 102432
    done
    cmp "$OUT/$arm.stage0.r1.w${w}.out.f32" "$OUT/$arm.stage1.r1.w${w}.in.f32"
    cmp "$OUT/$arm.stage1.r1.w${w}.out.f32" "$OUT/$arm.stage2.r1.w${w}.in.f32"
  done
  test -s "$OUT/$arm.complete.json"
done
~~~

Check all workers' admitted/provisional/committed traces and positions 0 → 1 → 2, keep=1, rows=2, and matching ACKs. Use the existing STFR parser (no new reference arithmetic):

~~~bash
python3 - "$OUT" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "tools")
from heterogeneous_compare_parity import read_run, require
root = Path(sys.argv[1])
runs = {name: read_run(root / name, allow_partial=True) for name in ("a", "b", "c")}
wanted = {"a": [[1, 271], [198, 271]], "b": [[1, 1], [198, 271]],
          "c": [[198, 271], [198, 271]]}
identity = ("plan", "plan_fingerprint", "model_fingerprint", "pack_index_sha256",
            "gguf_shards", "profile_path")
reference = runs["a"][0][0]
for name, run in runs.items():
    require(len(run) == 2, "wrong number of windows")
    for pos, (meta, packet) in enumerate(run):
        require(meta["pos0"] == pos and meta["n_keep"] == 1 and packet["rows"] == 2
                and meta["token_ids"] == wanted[name][pos], "wrong prefix metadata")
        require(packet["vocab"] == runs["a"][0][1]["vocab"] and
                all(meta[k] == reference[k] for k in identity), "source or geometry changed")
        print(name, pos, "accepted prediction", packet["predictions"][0])
a, b, c = (runs[name][1][1] for name in ("a", "b", "c"))
print("A/B continuation predictions equal:", a["predictions"] == b["predictions"])
print("A/B continuation logits bytes equal:", a["logits"].tobytes() == b["logits"].tobytes())
print("A/C continuation logits bytes differ:", a["logits"].tobytes() != c["logits"].tobytes())
for stage in (0, 1):
    suffix = f".stage{stage}.r1.w2.out.f32"
    x, y, z = ((root / (name + suffix)).read_bytes() for name in ("a", "b", "c"))
    print("stage", stage, "A/B equal", x == y, "A/C differ", x != z)
PY
~~~

Exact byte equality within each transport remains mandatory. Across independent computations, record observed differences without imposing a new KL or bitwise-arithmetic threshold. A/B equality supports rejected-suffix independence; A/C difference supports accepted-history sensitivity. If outputs vary, compare an identical fresh A repeat once before assigning cause. Report inconclusive where appropriate. These are bounded observable state checks, not independent proof of every stage's K/V or recurrent tensor. Existing host protocol tests cover fail-stop logic; intentional native GPU fault injection is not authorized and remains UNVERIFIED. No broad architecture acceptance or production-readiness claim until limitations are reviewed.
