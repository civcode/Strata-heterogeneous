# M3.1–M3.5 — native text completion and CLI integration

**Status:** M3.1–M3.4 bounded validation PASS, as reported by the Oberon operator for the `4a976a0` implementation. M3.5, long-context behavior, and native GPU failure paths remain UNVERIFIED or unimplemented as appropriate. M2's bounded native boundary and state results remain documented in [BOUNDARY_VALIDATION_M218_M222.md](BOUNDARY_VALIDATION_M218_M222.md). The [implementation contract](IMPLEMENTATION_CONTRACT.md) remains authoritative.

## Scope and code

| Milestone | Implemented in this increment | Qualification status / remaining work |
| --- | --- | --- |
| M3.1 Tokenization | Existing Strata tokenizer loads the primary original GGUF identified by the native, CPU-only validated worker configuration. Detokenization uses the same tokenizer. Optional single-turn chat uses the exact model template and existing frontend literal-handling helpers. | PASS, reported bounded real source/tokenizer/template validation |
| M3.2 Autoregressive generation | Existing `strata_heterogeneous_live_smoke` coordinator supports a fresh greedy request and feeds each previously predicted token back into its single acknowledged session. Stop IDs come from the loaded model. | PASS, reported bounded real decode and corrected trace checks |
| M3.3 Prompt processing/context | Sequential ordinary verifier windows, optional prompt chunks up to the already configured limit (1..8); positions and full accepted counts advance after all worker ACKs. | PASS, reported bounded prompt/chunk runs; larger contexts and optimized bulk prefill remain unverified |
| M3.4 Strata CLI | Explicit `strata.py heterogeneous-generate` command with human or JSON output; CPU preflight by default, GPU use only with `--run`. | PASS, reported bounded checkout CLI; packaging/launcher integration remains pending |
| M3.5 Strata server | Audited integration seam: existing `serve.server.Engine.generate` protocol, existing frontend and service. | Resident worker/request lifecycle, safe session reset, cancellation, failure/reaping, readiness, existing server integration. **Not enabled or implemented as HTTP serving yet.** |

The new native coordinator code reuses `FinalPipelineSession::execute` and `commit` and the already-qualified transfer mechanism. No new GPU kernels, sampler, engine, worker executable, model format or transport protocol. Prompt ingestion commits each real input window using `core::Verifier`. The last prompt row predicts the first completion token; subsequent predictions are fed back one at a time. The last returned token need not itself be evaluated, since this is a fresh one-shot request with no resumable session. EOS/EOT predictions terminate without appearing in the completion. The existing greedy head's default behavior is unchanged.

The CLI passes prompt IDs on stdin (bounded to 4096 tokens / 64 KiB of serialized input), not as fake activations. All stages use the existing source/plan/weight configuration guards. The coordinator's `--describe-source` response provides the primary GGUF, pack directory, pack index digest and minimum worker capacities; a second native preflight requires the same primary GGUF path and pack index identity before executing. The existing GGUF metadata reader is used; no independent tokenizer or external tokenizer download. Chat mode requires the extracted model template to equal its original GGUF metadata and uses Strata's existing frontend. It returns raw completion text, not a full chat-service response envelope.

Prompt plus requested completion must fit the smallest configured `max_cells`. The requested prompt chunk must fit every worker's `max_t` and activation/STFR caps. Completion is bounded to 1..256 tokens, default 16; default chunk is 1. These are initial conservative bounds, not evidence of long-context performance. Generation does not enable numerical capture or probes; old diagnostic environment switches used by the worker are cleared. Machine-readable results are returned only after all executed windows commit and all workers exit successfully. No streaming or persistent conversation cache in this increment.

## Local evidence

The modified coordinator translation unit was compiled to an object using GCC C++17 with `-O2 -Wall -Wextra -Wpedantic -Werror` using the six real repository headers, with source and header Git blob identities checked. The Python CLI adapter was syntax compiled and its actual parser exercised for default/explicit options and four invalid generation/chunk budgets; those refusals occurred before launching any subprocess. The container could not resolve github.com for a complete checkout, and no GPU/model is available here. Full host linking, CTest, CUDA/HIP builds, actual tokenizer/template integration, and end-to-end generation were pending at the time of that local check. No GitHub Actions were used.

## Reported Oberon validation

The operator reports completion of the bounded M3.1–M3.4 handoff for implementation commit `4a976a010ec40b0482050a83b4061e66992d850c`: all 38 CPU CTests and the separate CUDA/HIP builds passed. Plain, chunked, chat and shared-memory text generation completed. An initial trace-check command failed because the operator misinterpreted a window counter; corrected trace checks passed for all four runs. No source changes were reported. This is not a native inference failure, nor an unqualified claim that every possible test passed.

The full operator handoff is `/home/chris/tmp/strata-heterogeneous/m31-run-cmEMYwfE/HANDOFF.md`. It contains the exact tested revision, output texts/token IDs, environment isolation, logs and limitations. That file and raw logs were not independently retrieved here; do not invent their contents. M3.5 is still unimplemented. Long-context, native failure-path behavior and broad production readiness remain unverified.

## Python environment on Oberon (one-time setup, reuse thereafter)

Use the **existing repository-root `requirements.txt`** (Strata's pinned runtime dependencies, Python 3.10+). The upstream setup already uses a local `.venv/`, which is ignored by Git. Do not install dependencies into the OS Python, use `--break-system-packages`, copy another checkout's environment or run the complete model setup just to get Python packages.

~~~bash
cd /home/chris/workspace/Strata-heterogeneous
unset PYTHONHOME PYTHONPATH
# Use an installed Python 3.10+; only create the venv if absent.
if [ ! -e .venv ] && [ ! -L .venv ]; then
    python3 -m venv .venv
fi
# Refuse a non-isolated or broken environment rather than modify system Python.
.venv/bin/python -c 'import sys, site, pathlib; p = pathlib.Path(sys.prefix).absolute(); expected = pathlib.Path(".venv").absolute(); cfg = (p / "pyvenv.cfg").read_text().lower(); assert sys.version_info >= (3, 10) and p == expected and sys.prefix != sys.base_prefix and "include-system-site-packages = false" in cfg and not site.ENABLE_USER_SITE, "local isolated Python 3.10+ venv required"'
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python -c 'import regex, jinja2, numpy, yaml, psutil, requests, PIL; print("Strata runtime imports: OK")'
~~~

Execute in order and stop if any command fails (use `set -euo pipefail` in a runbook shell). If `venv`/`ensurepip` is missing, ask the workstation maintainer to provide the appropriate Python venv support; do not install OS packages or alter another project automatically. If an existing `.venv` is incompatible, report it and arrange a clean recreation rather than deleting it silently. If a pinned wheel cannot be installed, retain the failing pip output and resolve it explicitly; do not relax upstream pins or silently use a different requirements set.

Reuse the same environment for future runs. Repeat the pinned pip installation and `pip check` after changes to `requirements.txt` or the interpreter. Ordinary invocations need neither activation nor another install: `.venv/bin/python strata.py ...` or `.venv/bin/python -m unittest ...`. Optional activation (`source .venv/bin/activate`) is only shell convenience. Use `STRATA_NO_HANDOVER=1` when invoking the checkout CLI so an installed Control version cannot take over validation.

Other existing requirements files have narrower purposes: `requirements-build.txt` supplies CMake/Ninja only if needed; `requirements-monitor.txt` is for the optional terminal monitor; `requirements-nvfp4-local.txt` is for NVIDIA FP4 conversion and is not needed for this Q2_0 run. Preserve the existing verified native build tools and configurations. For CMake-discovered Python tests, check the interpreter in the existing CMake cache; if it is not the local venv, reconfigure using the original source directory and existing build options with `-DPython3_EXECUTABLE="$PWD/.venv/bin/python"`, then verify the registered test commands (e.g. `ctest --test-dir build-hetero-smoke-host -N -V`). Do not change GPU flags, compiler paths or other cached settings. Record the venv interpreter/version and `pip check` output with validation artifacts.

## Oberon procedure

Use the existing authorized Q2_0 stage configs; do not change precision, budgets, model or GPU assignments to make tests pass. Preserve the successful M2 environment. Do not access either Strata-dev tree, change drivers, reset GPUs, intentionally kill live workers or dispatch GitHub Actions. Do not touch the pre-existing untracked pool probe. Use at most 28 parallel build jobs. Keep each GPU session sequential and stop on unexpected failure.

~~~bash
set -euo pipefail
cd /home/chris/workspace/Strata-heterogeneous
git switch feature/heterogeneous-cuda-window-admission
git pull --ff-only
git status --short
git rev-parse HEAD

OUT=$(mktemp -d /home/chris/tmp/strata-heterogeneous/m31-XXXXXX)
cmake --build build-hetero-smoke-host --parallel 28 2>&1 | tee "$OUT/host-build.log"
ctest --test-dir build-hetero-smoke-host --output-on-failure 2>&1 | tee "$OUT/ctest.log"
cmake --build build-hetero-smoke-cuda --parallel 28 --target strata_heterogeneous_cuda_worker strata_heterogeneous_live_smoke 2>&1 | tee "$OUT/cuda-build.log"
cmake --build build-hetero-smoke-hip --parallel 28 --target strata_heterogeneous_hip_worker 2>&1 | tee "$OUT/hip-build.log"

SMOKE="$PWD/build-hetero-smoke-cuda/experimental/heterogeneous/strata_heterogeneous_live_smoke"
CUDA="$PWD/build-hetero-smoke-cuda/experimental/heterogeneous/strata_heterogeneous_cuda_worker"
HIP="$PWD/build-hetero-smoke-hip/experimental/heterogeneous/strata_heterogeneous_hip_worker"
gen=("$PWD/.venv/bin/python" strata.py heterogeneous-generate
  --coordinator "$SMOKE" --cuda-worker "$CUDA" --hip-worker "$HIP"
  --plan cuda:0:0-12,hip:0:12-30,hip:1:30-48 --fingerprint 2026111301
  --config "$PWD/build-q2-stage-configs/cuda-stage0.cfg"
  --config "$PWD/build-q2-stage-configs/hip-stage1.cfg"
  --config "$PWD/build-q2-stage-configs/hip-stage2.cfg"
  --output json)
export STRATA_NO_HANDOVER=1
export STRATA_HETERO_WINDOW_TRACE=1

# No GPU launched by this command: real metadata + prompt + admission preflight.
"${gen[@]}" --prompt "The capital of France is" --max-new-tokens 8 \
  > "$OUT/plain-preflight.json" 2> "$OUT/plain-preflight.log"
# Verify source, tokenizer IDs, context limits and plan before adding --run.
cat "$OUT/plain-preflight.json"

"${gen[@]}" --prompt "The capital of France is" --max-new-tokens 8 --run \
  > "$OUT/plain.json" 2> "$OUT/plain.log"
cat "$OUT/plain.json"

# If the first real run completed, test bounded prompt chunks.
"${gen[@]}" --prompt "The capital of France is" --max-new-tokens 8 --prompt-chunk 2 \
  > "$OUT/chunk-preflight.json" 2> "$OUT/chunk-preflight.log"
"${gen[@]}" --prompt "The capital of France is" --max-new-tokens 8 --prompt-chunk 2 --run \
  > "$OUT/chunk.json" 2> "$OUT/chunk.log"

# Single-turn model-template chat, only if the existing context budget permits.
"${gen[@]}" --chat --prompt "What is the capital of France?" --max-new-tokens 16 \
  > "$OUT/chat-preflight.json" 2> "$OUT/chat-preflight.log"
"${gen[@]}" --chat --prompt "What is the capital of France?" --max-new-tokens 16 --run \
  > "$OUT/chat.json" 2> "$OUT/chat.log"
~~~

Prepare the local `.venv/` using the pinned runtime requirements above before executing this procedure. Never download model files or use a fallback tokenizer for these runs. If chat or chunk preflight rejects the configured context or window limits, report that limitation and stop that arm; do not increase worker budgets automatically. Keep output and diagnostics separate; review JSON records and actual logs. With prior confirmed success, repeat the identical plain request once using `--shared` and a fresh artifact prefix if the memory budget allows. Do not require identical logits/predictions between different prompt chunk shapes, and do not make historical cross-backend KL a generation gate.

## Acceptance and report

- Verify all existing host tests/builds and explicit text/tokenizer preflight. The result must be valid source-bound, nonempty prompt IDs inside the model vocabulary. Check chat formatting against the model-source template when enabled.
- For real generation, verify `status=complete`, valid completion token IDs, text from the same tokenizer, bounded count and `finish_reason`. A length stop has exactly the requested output count; EOS/EOT stop can be early. At least two output tokens on a successful real prompt are needed to exercise feedback decode; immediate EOS alone is not evidence of autoregression. Do not dictate a specific word as an architectural gate.
- Check the existing worker trace: each request/window is matched across all stages; positions advance by full prompt-chunk size and then by one; positive ACKs arrive for every evaluated window; never publish a failed request. For a length stop producing N tokens, the total evaluated input count is prompt length plus N-1 (no need to evaluate the final returned token). A sample with EOS after N output tokens evaluates prompt length plus N.
- Generation uses the normal capture-disabled path, so the completed run must not create probe output files. Native failure-path behavior remains unverified: do not inject GPU faults.
- If useful, compare actual prompt/feedback IDs from the completion with an explicit short sequence through the existing smoke runner (within its already bounded window count), with the same native model/plan; no new fixtures or reference arithmetic. Report observed results, including any variability.
- Report PASS / FAIL / UNVERIFIED for each milestone, tested commit, model source/pack fingerprint, device placement, native logs, output JSON, and all remaining limitations. Preserve previous M2 results; do not claim M3.5, production readiness, optimized prefill or long context based on this run.

## Next: existing server, not a new server

The current stage worker admission is deliberately fresh-session/single-request, and the coordinator launches and reaps workers per CLI invocation. A server adapter that reloads the full model per HTTP request, or pretends an existing worker can safely accept another request, would be an unsuitable substitute for the existing resident Strata server. After Oberon generation succeeds, identify the smallest reusable model-versus-session lifetime separation using existing Strata session initialization/reset, explicit multi-stage lifecycle acknowledgments and safe cancellation/poisoned-session recovery. Then integrate with the existing server `Engine.generate(ids, max_new, sampling, cancel)` interface and normal readiness/ownership controls; do not create a parallel HTTP service. Sampling beyond the existing greedy head and concurrent/multi-turn requests need their own explicit qualification.

### M3.5 implementation roadmap

M3.1–M3.4 are bounded hardware PASS as reported above. Work in this order;
reuse the successful one-shot CLI for regression and extract only what is
required for the existing resident server.

| Step | Implementation | Acceptance gate |
| --- | --- | --- |
| M3.5a — Lifetime audit | Separate loaded native weights/expert resources from request-local verifier/session state; identify existing reset/initialization and server engine lifecycle APIs. | Document reset coverage, ownership and any actual gaps before adding code. |
| M3.5b — Resident workers | Retain the same independently loaded CUDA/HIP workers; add only required request begin/end/reset coordination with matching acknowledgments across stages. | Two different requests and a repeated first request on the same workers, with no model reload, no stale state and fresh-session control comparisons. |
| M3.5c — Safe request lifecycle | Reuse existing admission, commit, timeout, poison and shutdown paths; support cooperative cancellation at safe window boundaries and reject invalid/replayed requests. | CPU protocol/lifecycle checks; real sequential requests and ordinary cancellation/recovery on Oberon. Never inject GPU faults. A poisoned worker set is not reused; no claim of rollback. |
| M3.5d — Existing server adapter | Reuse the committed generation loop behind the existing Engine interface; expose an explicit experimental heterogeneous configuration in the existing Strata server. Start with one active text request, greedy sampling and bounded context. Reject unsupported options instead of silently ignoring them. | Existing server authentication/readiness/ownership and request validation remain in force; ordinary single-backend modes are unchanged. |
| M3.5e — End-to-end qualification | Exercise the existing API and CLI client against multiple sequential prompts, normal stop/length limits, cancellation and orderly shutdown; verify resident process identities and no model reload. | Actual responses, token/commit traces, isolation and lifecycle evidence on Oberon; existing CPU and native builds pass. |

Do not implement another inference engine, sampler, HTTP server, or temporary
numerical fixture. Prefer no source changes where existing code suffices.
Streaming, concurrency, persistent conversational caches, wider sampling,
long-context performance and native fault handling are separate qualification
items; do not advertise them before supported and verified. Report PASS /
FAIL / UNVERIFIED separately for host and hardware.

Every Oberon handoff must include the local Python venv preflight explicitly:
create the ignored checkout-local `.venv` only if absent, install the existing
pinned `requirements.txt` when first created or changed, run
`.venv/bin/python -m pip check`, and use `.venv/bin/python` for every Python
command. Verify CMake's Python test interpreter follows the same convention.
See the one-time environment procedure above. No system Python or other
checkout's venv, no GPU resets, no GitHub Actions.
