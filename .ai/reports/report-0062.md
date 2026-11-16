# Report 0062 — Oberon current-branch compatibility validation

Iteration: 0062 · Branch: feature/heterogeneous-hip-only · Start commit: ef419d9e (synced, fast-forward from 32d3ff45)
Status: **COMPLETE** — builds, CPU tests, launchers and bounded GPU smoke (HIP-only, mixed) pass. A first pass in a sandbox without GPUs was redone with GPUs visible.

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

## GPU CTest (GPUs visible)
CUDA 193/194, HIP 334/339. Remaining failures, all unrelated to this batch's changes: `ple_parity` (cuda, hip) and `expert_parity`/`pool_test` (hip) need data files at relative paths (`../../Q2_0/*.gguf`, `pack/full/experts.bin`) absent from the build dir; `hip_flag_order_test` (3 host-visibility ordering checks on HIP) and `conversation_copy_streams_test` (first copy stream+event 151552 KiB > 16 MiB limit on this ROCm) are native HIP-runtime behaviors in code not touched here. Not investigated further; low priority.

## GPU smoke (public launchers, `--no-open`, loopback :8081, sequential; ~156 s wall total of the 5 min budget)
Common: ready ~22 s after start; /health ok (max_context 262144), /v1/models loaded, web page 200 (23 KB; chat UI not exercised, HTTP/assets only). Non-streaming and streaming short completions return finish_reason and usage; greedy (temperature 0, top_p 1, top_k 0, min_p 0, seed 1) accepted; 200-token budget within cap completes (stop); `temperature 0.8` is refused with 400 invalid_request_error ("not supported by the experimental greedy resident engine") before GPU work; a 2 s client disconnect during streaming produced `finish=cancelled` (native drain, `next` advanced consistently, capacity re-checked at prefill_drained) and the next fresh request answered correctly ("OK"); /health stayed ok.
- HIP-only (HIP0 [0,24), HIP1 [24,48)): short prompt prefill 146–164 ms for 15–20 tokens (~137 tok/s, includes first step; first request 830 ms cold); decode ~45–53 tok/s on 85–104-token completions (no MTP). VRAM in use ~23 GB per GPU while loaded (free >9.6 GB each, reserve 1 GiB, fits=1).
- Mixed (CUDA0 [0,15), HIP0 [15,33), HIP1 [33,48), MTP spec 4): prefill ~170 ms for 15–20 tokens (~117 tok/s); decode 53–54 tok/s on 94–99 tokens, MTP drafts accepted 45/83 and 51/93 (one stream: 83 tok/s over 23 tokens, `host_commit_progress` timing). VRAM ~14.2 GB CUDA, 18.1/17.4 GB HIP; free ≥1.8 GB CUDA, ≥15 GB HIP at early_commit (reserve 1 GiB respected). No error/fault/mismatch lines in the log (matches are "no error" verify lines).
- Timings are per-request from the server (`timings`, `prefill_ms` / `decode_ms` in HETERO_REQUEST_TIMING); tiny unmatched samples, not a speedup claim and not comparable to 0044/0045.
- Shutdown: SIGTERM to the server (SIGINT is ignored for a backgrounded process in this harness) closed in ~2 s; log ends `HETERO_CLOSED`; no stray workers; HIP VRAM back to ~57 MB and CUDA to 15 MiB. No force kills.

## Remaining
- Chat UI interaction not exercised (no GUI); MTP BATCH ordering only inferred from cancel/next-position consistency in logs.
- Source-gate (0061) / pinned-source (0060) audit of `8db7ddc4` remains incomplete (commit object exists in /home/chris/workspace/Strata; not audited here); no native sampler/algorithm changed. No performance qualification attempted. Native `--serve` wire parity not claimed.
- Environment note: untracked `.env` and `tools/heterogeneous_pool_probe.cpp` pre-existed and were left uncommitted.
