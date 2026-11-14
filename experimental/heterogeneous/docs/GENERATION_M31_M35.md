# M3.1–M3.5 — native text completion and CLI integration

**Status: implementation increment prepared; model and GPU execution UNVERIFIED on this increment.** M2's bounded native boundary and state results remain documented in [BOUNDARY_VALIDATION_M218_M222.md](BOUNDARY_VALIDATION_M218_M222.md). The [implementation contract](IMPLEMENTATION_CONTRACT.md) remains authoritative.

## Scope and code

| Milestone | Implemented in this increment | Still required |
| --- | --- | --- |
| M3.1 Tokenization | Existing Strata tokenizer loads the primary original GGUF identified by the native, CPU-only validated worker configuration. Detokenization uses the same tokenizer. Optional single-turn chat uses the exact model template and existing frontend literal-handling helpers. | Real source/tokenizer/template verification on Oberon |
| M3.2 Autoregressive generation | Existing `strata_heterogeneous_live_smoke` coordinator supports a fresh greedy request and feeds each previously predicted token back into its single acknowledged session. Stop IDs come from the loaded model. | Real text generation and feedback validation |
| M3.3 Prompt processing/context | Sequential ordinary verifier windows, optional prompt chunks up to the already configured limit (1..8); positions and full accepted counts advance after all worker ACKs. | Real prompt/chunk tests, larger approved context budgets; not optimized bulk prefill |
| M3.4 Strata CLI | Explicit `strata.py heterogeneous-generate` command with human or JSON output; CPU preflight by default, GPU use only with `--run`. | End-to-end CLI verification and later packaging/launcher integration |
| M3.5 Strata server | Audited integration seam: existing `serve.server.Engine.generate` protocol, existing frontend and service. | Resident worker/request lifecycle, safe session reset, cancellation, failure/reaping, readiness, existing server integration. **Not enabled or implemented as HTTP serving yet.** |

The new native coordinator code reuses `FinalPipelineSession::execute` and `commit` and the already-qualified transfer mechanism. No new GPU kernels, sampler, engine, worker executable, model format or transport protocol. Prompt ingestion commits each real input window using `core::Verifier`. The last prompt row predicts the first completion token; subsequent predictions are fed back one at a time. The last returned token need not itself be evaluated, since this is a fresh one-shot request with no resumable session. EOS/EOT predictions terminate without appearing in the completion. The existing greedy head's default behavior is unchanged.

The CLI passes prompt IDs on stdin (bounded to 4096 tokens / 64 KiB of serialized input), not as fake activations. All stages use the existing source/plan/weight configuration guards. The coordinator's `--describe-source` response provides the primary GGUF, pack directory, pack index digest and minimum worker capacities; a second native preflight requires the same primary GGUF path and pack index identity before executing. The existing GGUF metadata reader is used; no independent tokenizer or external tokenizer download. Chat mode requires the extracted model template to equal its original GGUF metadata and uses Strata's existing frontend. It returns raw completion text, not a full chat-service response envelope.

Prompt plus requested completion must fit the smallest configured `max_cells`. The requested prompt chunk must fit every worker's `max_t` and activation/STFR caps. Completion is bounded to 1..256 tokens, default 16; default chunk is 1. These are initial conservative bounds, not evidence of long-context performance. Generation does not enable numerical capture or probes; old diagnostic environment switches used by the worker are cleared. Machine-readable results are returned only after all executed windows commit and all workers exit successfully. No streaming or persistent conversation cache in this increment.

## Local evidence

The modified coordinator translation unit was compiled to an object using GCC C++17 with `-O2 -Wall -Wextra -Wpedantic -Werror` using the six real repository headers, with source and header Git blob identities checked. The Python CLI adapter was syntax compiled and its actual parser exercised for default/explicit options and four invalid generation/chunk budgets; those refusals occurred before launching any subprocess. The container could not resolve github.com for a complete checkout, and no GPU/model is available here. Full host linking, CTest, CUDA/HIP builds, actual tokenizer/template integration, and end-to-end generation remain pending on Oberon. No GitHub Actions were used.

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
gen=(python strata.py heterogeneous-generate
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

Use the normal Strata Python environment (existing regex/Jinja2 dependencies), without downloading model files or a fallback tokenizer. If chat or chunk preflight rejects the configured context or window limits, report that limitation and stop that arm; do not increase worker budgets automatically. Keep output and diagnostics separate; review JSON records and actual logs. With prior confirmed success, repeat the identical plain request once using `--shared` and a fresh artifact prefix if the memory budget allows. Do not require identical logits/predictions between different prompt chunk shapes, and do not make historical cross-backend KL a generation gate.

## Acceptance and report

- Verify all existing host tests/builds and explicit text/tokenizer preflight. The result must be valid source-bound, nonempty prompt IDs inside the model vocabulary. Check chat formatting against the model-source template when enabled.
- For real generation, verify `status=complete`, valid completion token IDs, text from the same tokenizer, bounded count and `finish_reason`. A length stop has exactly the requested output count; EOS/EOT stop can be early. At least two output tokens on a successful real prompt are needed to exercise feedback decode; immediate EOS alone is not evidence of autoregression. Do not dictate a specific word as an architectural gate.
- Check the existing worker trace: each request/window is matched across all stages; positions advance by full prompt-chunk size and then by one; positive ACKs arrive for every evaluated window; never publish a failed request. For a length stop producing N tokens, the total evaluated input count is prompt length plus N-1 (no need to evaluate the final returned token). A sample with EOS after N output tokens evaluates prompt length plus N.
- Generation uses the normal capture-disabled path, so the completed run must not create probe output files. Native failure-path behavior remains unverified: do not inject GPU faults.
- If useful, compare actual prompt/feedback IDs from the completion with an explicit short sequence through the existing smoke runner (within its already bounded window count), with the same native model/plan; no new fixtures or reference arithmetic. Report observed results, including any variability.
- Report PASS / FAIL / UNVERIFIED for each milestone, tested commit, model source/pack fingerprint, device placement, native logs, output JSON, and all remaining limitations. Preserve previous M2 results; do not claim M3.5, production readiness, optimized prefill or long context based on this run.

## Next: existing server, not a new server

The current stage worker admission is deliberately fresh-session/single-request, and the coordinator launches and reaps workers per CLI invocation. A server adapter that reloads the full model per HTTP request, or pretends an existing worker can safely accept another request, would be an unsuitable substitute for the existing resident Strata server. After Oberon generation succeeds, identify the smallest reusable model-versus-session lifetime separation using existing Strata session initialization/reset, explicit multi-stage lifecycle acknowledgments and safe cancellation/poisoned-session recovery. Then integrate with the existing server `Engine.generate(ids, max_new, sampling, cancel)` interface and normal readiness/ownership controls; do not create a parallel HTTP service. Sampling beyond the existing greedy head and concurrent/multi-turn requests need their own explicit qualification.
