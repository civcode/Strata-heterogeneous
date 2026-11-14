# M3.5 — resident native workers and existing Strata server

**Status: bounded M3.5a–e PASS on Oberon**, as reported in [report 0002](../../../.ai/reports/report-0002.md) for tested source revision `992fc289634e228f57cd062eb8a35fcc01fbc2cb`. This is experimental opt-in resident serving, not general production or native failure-path qualification.

## Reported Oberon qualification

The coding agent tested the corrected source revision above; report commit `580c97f8ef38bbe76cd116ab1bae8ce6714a482a` adds only the report. Python 3.12 checkout-local venv passed pip check; all three CMake caches used it. Host build, 38/38 CTests, and independent CUDA worker/coordinator and HIP worker builds passed.

| Gate | Bounded result |
| --- | --- |
| M3.5a | PASS: source audit and CPU reset/replay/poison coverage |
| M3.5b | PASS: fresh A/B vs resident A/B/A token and termination equivalence in socket and shared-memory transports |
| M3.5c | PASS: active client disconnect, cooperative cancellation, subsequent successful request and orderly owned-process shutdown |
| M3.5d | PASS: existing loopback HTTP chat completions, sequential A/B/A and post-cancel A |
| M3.5e | PASS: bounded end-to-end CLI/server operation on the recorded plan and devices |

The tested Q2_0 plan was `cuda:0:0-12,hip:0:12-30,hip:1:30-48` (RTX 4070 Ti SUPER and two R9700s). CLI A returned token IDs `[11751,13]` (stop); B returned `[44745,11,864,369,1048,279,7526,3177]` (length). Resident traces checked request IDs, position-zero resets and 7/12/7 committed windows per stage in both transports. The server retained the same coordinator and three worker identities across requests and cancellation; one device startup announcement per stage and no reload were logged (not an independent measurement of every allocation). Active cancellation was observed, the next A succeeded, and a normal owner SIGTERM produced `HETERO_CLOSED` and all owned children exited.

Raw artifacts are on Oberon at `/home/chris/tmp/strata-heterogeneous/m35-0002-PF0NDL1z/`; the committed report is the evidence available here, not an independent rerun or inspection of raw logs. Longer contexts, concurrent/multiturn requests, non-greedy sampling, incremental token streaming, native failure recovery, abrupt parent death/hangs, broader numerical parity, packaging and production readiness remain unqualified.

## Architecture and scope

The existing native stage resources retain weights, experts, session allocations and verifier graphs. At each independent request boundary all stages acknowledge a fenced reset using the existing core::session_zero (GDN, QSA/KV, residual and PLE history). The loader supplies the native reset callback; no new GPU kernels or numerical path. The opt-in resident handshake and request IDs preserve the legacy one-shot mode. Requests start at position zero, advance only on commit ACKs, and are serialized. Reset or protocol failure poisons the worker set; do not reuse it. Cancellation is cooperative between fully committed windows; no speculative rollback. The next request resets all session state.

The CLI and server share the existing greedy coordinator and source-bound tokenizer/template. The existing serve.server Engine adapter accepts one text request at a time, with bounded context (at most 4096 total tokens) and configured completion cap (1..256). It uses ordinary Service HTTP/auth/readiness handling, with a dedicated loopback-only configuration. Actual completion tokens are buffered until the coordinator completes the request; SSE keepalives are not incremental token streaming. Other sampling modes, embeddings, tools and advanced features are out of scope. Do not claim packaging/profile-launcher support or concurrent inference.

The coordinator owns its native children and uses stdin EOF for graceful cleanup at safe checkpoints; its separate process session prevents foreground server Ctrl+C from also interrupting native workers. Normal shutdown is through the server owner. No new forced termination or automatic restart of a poisoned set is introduced. Abrupt parent death, hanging native work, guaranteed orphan prevention, and native failure recovery are not qualified.

## Oberon handoff

Work only in /home/chris/workspace/Strata-heterogeneous and /home/chris/tmp/strata-heterogeneous. Preserve existing untracked probes, configs, model files and GPU assignments. No GitHub Actions, GPU resets/reboots, driver unload, intentional worker kills or Strata-dev access. Stop on first unexpected error; do not bypass guards.

Use the checkout-local venv procedure in [GENERATION_M31_M35.md](GENERATION_M31_M35.md); Python 3.10+; create only if absent, reuse if isolated, install existing root pinned requirements if needed, pip check and use that interpreter for ALL Python steps. Verify CMake's discovered Python interpreter; preserve existing build options.

~~~bash
set -euo pipefail
cd /home/chris/workspace/Strata-heterogeneous
git switch feature/heterogeneous-cuda-window-admission
git pull --ff-only
git rev-parse HEAD
unset PYTHONHOME PYTHONPATH
if [ ! -e .venv ] && [ ! -L .venv ]; then
    python3 -m venv .venv
fi
PY="$PWD/.venv/bin/python"
"$PY" -c 'import sys,site,pathlib; assert sys.version_info >= (3,10) and sys.prefix != sys.base_prefix and pathlib.Path(sys.prefix).resolve() == pathlib.Path(".venv").resolve() and not site.ENABLE_USER_SITE'
"$PY" -m pip install -r requirements.txt
"$PY" -m pip check
OUT=$(mktemp -d /home/chris/tmp/strata-heterogeneous/m35-XXXXXX)
# Inspect Python3_EXECUTABLE/_Python3_EXECUTABLE in the existing CMake cache.
# If needed reconfigure with the original options and -DPython3_EXECUTABLE="$PY".
cmake --build build-hetero-smoke-host --parallel 28
ctest --test-dir build-hetero-smoke-host --output-on-failure
cmake --build build-hetero-smoke-cuda --parallel 28 --target strata_heterogeneous_cuda_worker strata_heterogeneous_live_smoke
cmake --build build-hetero-smoke-hip --parallel 28 --target strata_heterogeneous_hip_worker
SMOKE="$PWD/build-hetero-smoke-cuda/experimental/heterogeneous/strata_heterogeneous_live_smoke"
CUDA="$PWD/build-hetero-smoke-cuda/experimental/heterogeneous/strata_heterogeneous_cuda_worker"
HIP="$PWD/build-hetero-smoke-hip/experimental/heterogeneous/strata_heterogeneous_hip_worker"
export STRATA_NO_HANDOVER=1 STRATA_HETERO_WINDOW_TRACE=1
gen=("$PY" strata.py heterogeneous-generate --coordinator "$SMOKE" --cuda-worker "$CUDA" --hip-worker "$HIP" --plan cuda:0:0-12,hip:0:12-30,hip:1:30-48 --fingerprint 2026111301 --config "$PWD/build-q2-stage-configs/cuda-stage0.cfg" --config "$PWD/build-q2-stage-configs/hip-stage1.cfg" --config "$PWD/build-q2-stage-configs/hip-stage2.cfg" --output json --max-new-tokens 8)
A="The capital of France is"
B="The capital of Austria is"
"${gen[@]}" --prompt "$A" --run >"$OUT/fresh-a.json" 2>"$OUT/fresh-a.log"
"${gen[@]}" --prompt "$B" --run >"$OUT/fresh-b.json" 2>"$OUT/fresh-b.log"
"${gen[@]}" --resident --prompt "$A" --additional-prompt "$B" --additional-prompt "$A" >"$OUT/preflight.json"
"${gen[@]}" --resident --prompt "$A" --additional-prompt "$B" --additional-prompt "$A" --run >"$OUT/resident.json" 2>"$OUT/resident.log"
~~~

Verify the actual output and token IDs for fresh A/B vs resident A/B/A, distinct request IDs, position zero and monotonically increasing committed positions per request, one model load per stage, and clean shutdown. If tokens differ, repeat fresh A once to establish reproducibility and report differences rather than changing kernels or silently accepting contamination. Keep evidence of reset ACKs, commits and physical GPU identity. Socket first; shared-memory mode only after socket success.

## Existing HTTP server

After the resident CLI gate passes, create a new dedicated JSON config (no old profile/shared settings) within OUT. It must point to the same three worker configs and executables. For example:

~~~bash
"$PY" - "$OUT/server.json" "$SMOKE" "$CUDA" "$HIP" "$PWD" <<'PY'
import json,sys
from pathlib import Path
out,smoke,cuda,hip,root=sys.argv[1:]
cfg={"model_name":"qwen3.8-flash-next","max_queue":0,"reasoning_effort":"none",
     "heterogeneous":{"coordinator":smoke,"cuda_worker":cuda,"hip_worker":hip,
       "plan":"cuda:0:0-12,hip:0:12-30,hip:1:30-48","fingerprint":2026111301,
       "config":[str(Path(root)/"build-q2-stage-configs"/name) for name in
                 ("cuda-stage0.cfg","hip-stage1.cfg","hip-stage2.cfg")],
       "max_new_tokens":16,"prompt_chunk":1,"shared":False}}
with open(out,"x") as f: json.dump(cfg,f,indent=2)
PY
"$PY" -m serve.server --engine heterogeneous --config "$OUT/server.json" --host 127.0.0.1 --port 0 --offline --ready-file "$OUT/ready.json" --shutdown-timeout 30 >"$OUT/server.log" 2>&1
~~~

Run the server in a separate foreground terminal; use the actual port from the ready record. From another terminal, use the existing API client or curl to POST /v1/chat/completions with model qwen3.8-flash-next, temperature 0, max_tokens 8, stream false and one user message. Execute A/B/A sequentially with different questions, checking actual responses and finish reasons. Check coordinator PID and direct worker child PIDs/start times before and after; no model reload between requests. Test ordinary client disconnect at an acknowledged window using a streaming request within the bounded context: close the client, check HETERO_REQUEST_DONE finish=cancelled, and confirm another A succeeds with unchanged worker IDs. If cancellation cannot be observed during active generation, report UNVERIFIED, not PASS. Do not kill any GPU worker. End the owning server with a single normal Ctrl+C or SIGTERM and verify HETERO_CLOSED and owned process exit. Never force-kill a stuck worker as a test.

## Evidence

Report revision, venv interpreter/dependency checks, host CTest and separate native builds; M3.5a audit, b resident reset/isolation, c cooperative cancellation/cleanup, d API integration, e end-to-end results individually. Include output token IDs, worker identity and load/reset/commit traces, responses and shutdown evidence, artifact directory, any failures and remaining limits. Record native-failure handling as UNVERIFIED; do not add numerical fixtures or claim broader readiness.

Historical local evidence before handoff 0002: coordinator translation unit compiled using GCC C++17 with strict warnings and exact repository header/source blob identities; isolated Python adapter syntax and metadata validation checks passed. Full checkout retrieval failed due to DNS resolution in the local environment, so full linking, CTest, native builds, and live HTTP/GPU verification remain pending. Existing CPU protocol tests were extended for resident reset/replay/idle/poison behavior, not numerical inference. The subsequent bounded Oberon results above supersede these pre-handoff validation limits for the tested revision only.
