# M1.12 — CUDA final-stage verifier adapter (CPU-contract tested)

This is an **opt-in source adapter**, not a model-loading executable and
not proof of correct GPU inference. The NVIDIA RTX is currently unavailable
on Oberon (Xid 79/154); **do not reset or reboot it for this task**.

## Integration contract

`CudaFinalVerifierExecutor` implements the existing `FinalStageExecutor`
interface from `final_pipeline.hpp`. It is compiled into the
`strata_heterogeneous_cuda_verifier` archive when
`STRATA_HETERO_CUDA_VERIFIER=ON`. Run it only in an independently spawned
CUDA-only worker, through `serve_final_stage_worker()`. The existing
activation-only CUDA adapter and socket/memfd protocol are unchanged.

The model loader must provide exclusive live `CudaVerifierBindings`:
`WeightTable`, `ModelGeometry`, stage-carved `SessionState`, residency hits,
CPU expert-pool callback with its context, matching model fingerprint,
position after prefill, and an actual CUDA device ordinal. Keep every borrowed
object alive until this executor has been shut down. This increment does not
load models, perform prompt prefill, or infer fingerprints from weights.

Supported stages are final **CUDA** layer ranges whose start is greater
than zero, with an input activation from a prior stage. The stage owns an
input device allocation, pinned host staging, CUDA completion event, and a
`core::Verifier`. The verifier receives `set_stage(first,end,input,nullptr)`,
`set_handoff_rows(max_t)`, `set_always_publish(true)`, and
`set_head_sampling(true)` before initialization. This prevents T=1
self-commit and keeps stage-local state provisional until `commit_prefix`.

A `WindowSpec` carries the exact token IDs and `pos0` to `Verifier::run`.
The final worker uploads the incoming `R | bo | inject` rows, waits for a
stream event, then runs the actual verifier. It transports `out[t]` as
`predicted_tokens[t]`. `Verifier::nonfinite_in(t)` produces nonfinite-row
markers; optional `copy_logits(t,...)` exports the real float32 head logits.
No synthetic predictions or fake activation responses are permitted.

The default is `CudaFinalVerifierOptions{include_logits=true}`: this
makes unmarked NaN/Inf head logits detectable by the result gate. If logits
are disabled, `Verifier::nan_guard_on()` must be enabled, otherwise
`prepare()` rejects the configuration. Optional logits are refused before
GPU execution if worst-case STFR/v1 packet size exceeds 16 MiB.

`commit_prefix` requires exactly matching request/window IDs and **positive**
`n_keep <= T`; it calls `Verifier::commit(n_keep)`, `wait_commit()`, and a
completion fence. `n_keep=0` is deliberately rejected until the real
verifier has a proven indexer/KV/GDN/PLE restoration operation. Nonfinite
results cannot be committed even if a caller bypasses the coordinator gate.
On any worker error, abort and destroy the stage; never reuse its state.

## Limitations and GPU gates

- The verifier's default greedy sampling settings are used. Configurable
  sampling, token history/penalties, bias and speculative-q rows are **not**
  sent in the current host protocol and must be integrated separately.
- No real model-loading CUDA final-worker executable, head parity result,
  prefill, acceptance decision, restart/rollback, HIP worker, or mixed-vendor
  inference is implemented by this increment.
- Runtime calls in `Verifier::init/run/commit` may block; the existing
  supervising process must enforce a hard lifetime deadline, kill/reap a
  stuck worker, and rebuild stage-local state. Event polling is bounded but
  does not provide a deadline for blocking CUDA/engine API calls.
- `CudaFinalVerifierBindings` is exclusive: CUDA runtime objects never cross
  into HIP workers or through the AF_UNIX IPC protocol.
- Because of Xid 79/154, no GPU numerical validation was performed.
  Preserve the completed private NVIDIA bug report before any owner-approved
  maintenance recovery.

## Validation handoff (Oberon agent)

Use `/home/chris/workspace/Strata-heterogeneous` only, staging logs and
patches in `/home/chris/tmp/strata-heterogeneous`; completely ignore
`/home/chris/workspace/Strata-dev` and `/home/chris/tmp/strata-dev`.

Run `cmake --build "$build" --parallel 28` after configuring the standalone
module; run the full CTest suite with GCC, Clang, ASan and UBSan. Also enable
`STRATA_HETERO_CUDA_VERIFIER=ON` to compile the real source with native
CUDA/Strata headers, and link it against the native CUDA engine. No GitHub
Actions execution. Report the exact build logs and any API/ABI mismatch.

The `strata_heterogeneous_cuda_final_verifier_contract_tests` compile the
**same adapter source** against CPU doubles with fault-injected allocation,
H2D, kernel, fence, logits, commit and wait failures. They validate the
interface and cleanup only; they are not GPU kernel or numerical evidence.
