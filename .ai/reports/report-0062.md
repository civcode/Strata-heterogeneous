# Report 0062 — Oberon current-branch compatibility validation

Iteration: 0062 · Branch: feature/heterogeneous-hip-only · Start commit: ef419d9e (synced, fast-forward from 32d3ff45)
Status: **PARTIAL — CPU/build/launcher gates PASS; GPU smoke (task 3) BLOCKED: no GPU devices visible in this session.**

## Blocker
This session exposes no GPU device nodes (`/dev/nvidia*`, `/dev/kfd`, `/dev/dri` absent; `nvidia-smi` cannot reach the driver; `rocm-smi` absent). Device-requiring CTests fail with "no CUDA/ROCm-capable device is detected". No HIP-only/mixed HTTP smoke, cancel, UI or timing/residency observations were made; no GPU budget used. Not bypassed.
`/home/chris/tmp` is mounted read-only in this session; launcher output (`Path.home()/tmp/strata-heterogeneous/launchers`) was redirected with `HOME=/tmp/h0062` plus symlinks to the real private qualification dir. Private evidence is under /tmp/h0062 (not persisted in repo).

## Defect corrected
`experimental/heterogeneous/tests/cuda_stubs/strata/core/verify.hpp` lacked `Verifier::wait_commit`, which `src/cuda_verifier.cpp:315` (native MTP completion, f8002c5b) calls. This broke `strata_heterogeneous_cuda_verifier_contract_tests` in build-256k-host/cuda/hip (-Werror). Added a test-double `wait_commit` returning true. All three builds now succeed (28 jobs).

## Builds / CTest (28 jobs, existing build-256k-{host,cuda,hip}, existing .venv; `pip check` clean)
- host: 107/107 pass.
- cuda: 194 tests; hip: 339 tests. Failures are all device-requiring native kernel/parity tests (no device present, fail in ~0.02 s); the CPU-side heterogeneous contract tests pass. Not a regression indicator; rerun on a GPU-visible session.
- Artifact sha256 (main binaries): cuda `strata` b9c2141d…ff858; hip `strata` 4121058a…cc98 (hip). Worker/coordinator paths: build-256k-hip/…/strata_heterogeneous_hip_worker, build-256k-cuda/…/strata_heterogeneous_cuda_worker, build-256k-host/strata_heterogeneous_live_smoke.

## Python tests
71 tests OK: serve.test_heterogeneous_{sampling,resident,launch_compat,http_disconnect,retirement}, serve.test_context_overflow, strata_control.test_mixed_placement, tools.test_mixed_runconfig, tools.test_heterogeneous_oberon_launch, tools.test_strata_server_launch. Shell syntax checks for run-*.sh and tools/*.sh pass. These cover legacy nested-plan and duplicate-placement rejection and default-sampling refusal (pre-GPU).

## Launchers (`--prepare-only --no-open`, both PASS CPU preflight, no workers started)
- HIP-only: `--max-context 262144 --kv int8 --kv-resident 0 --prefill 128 --layer-split 24 --hetero-devices hip:0,hip:1`; stages HIP0 [0,24), HIP1 [24,48); max_new_tokens 2048 (existing config); stage cfg free_reserve_bytes=1073741824 (1 GiB).
- Mixed: same context/KV, `--mtp …/mtp/rt --spec 4 --spec-min-p 0.5 --prefill 32 --layer-split 15,33 --hetero-devices cuda:0,hip:0,hip:1`; stages CUDA0 [0,15), HIP0 [15,33), HIP1 [33,48); max_new_tokens 256.
- Worker binaries come from the explicit build-256k-{hip,cuda} dirs; no latest-config scanning observed.

## Not done / remaining
- Task 3 entirely (HTTP/UI smoke, cancel/ACK/MTP BATCH ordering, shutdown, timings) — needs a GPU-visible session.
- Source-gate (0061) / pinned-source (0060) audit of 8db7ddc4 remains incomplete; no native sampler/algorithm was changed. Pinned commit object was not inspected beyond existence check. No performance qualification was attempted.
- Native `--serve` wire parity not claimed.

Untracked `.env` and `tools/heterogeneous_pool_probe.cpp` pre-existed and were left uncommitted.
