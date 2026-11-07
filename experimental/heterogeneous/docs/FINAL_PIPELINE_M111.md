# M1.11 — Opt-in final-stage pipeline (CPU-testable)

Status: **protocol integration only**. No actual final CUDA/HIP verifier, model
loader, multi-vendor kernel execution or numerical GPU parity is implemented.
The ordinary activation-only PipelineSession and CPU mock demo remain unchanged.

## Contract and negotiation

This increment adds FinalPipelineSession and serve_final_stage_worker, not
an automatic substitution for the existing PipelineSession. The final stage
must use FinalStageExecutor::run_final_window to produce actual model head
predictions; StageExecutor::run_window's activation reply is *not* a valid
final-stage result.

The control-frame ABI remains **v4**. Two new v4 hello kinds, hello_final
and hello_shared_final (numeric IDs 12 and 13), explicitly negotiate
a final-stage worker. Intermediate workers retain hello/hello_shared and
send v4 activation replies. The last worker sends the existing **STFR/v1**
packet instead of a v4 activation reply. STFR is sent over the *same*
trusted AF_UNIX control socket only after negotiation; it is never
interleaved with an activation frame or read using receive_frame.
Therefore an old worker fails the capability handshake rather than being
misinterpreted. No global conversion of v4 into a fictitious v5 is needed.

The session/worker handshake still passes the optional shared memfd with
SCM_RIGHTS. Intermediate workers advance one shared slot sequentially.
The final stage borrows the validated input slot (or a socket payload
view) and returns predictions and optional logits via STFR/v1. Shared
buffers are host-visible, **not** CUDA/HIP device pointers.

## Coordinator state machine

1. FinalPipelineSession::initialize verifies every worker's exact identity,
   plan fingerprint, stage geometry and negotiated transport/result mode.
2. execute(input, WindowSpec) sends token IDs, position and activation
   rows through all stages. Earlier workers reply with activations; the
   final worker replies with an STFR/v1 result. The coordinator validates
   its final stage, request/window IDs, rows, plan fingerprint, vocabulary
   bounds and checksum and stores it in a FinalResultGate.
3. execute returns a **provisional** FinalStageResult. Caller determines
   accepted prefix externally. The provisional token IDs/logits are not
   proof of committed model state.
4. commit(n_keep) rejects n_keep greater than rows without contacting
   workers. Before contacting workers it rejects any nonfinite head
   markers. It then sends commit_prefix to all stages and verifies every
   committed_prefix acknowledgement, including the exact n_keep value.
5. The coordinator calls mark_committed and take_committed only after all
   ACKs match, returning publishable predictions. On any error, including
   partial distributed commit, it poisons the gate and shuts down every
   worker control socket. There is **no cross-process rollback**.

A CPU mock final executor is implemented only inside the test program:
it fabricates deterministic predicted IDs and optional logits to test
the contract. The library itself has **no fake final-stage inference**.
Real Verifier::run on the final worker will need to populate STFR from
actual head output and handle model weights, pool callbacks, KV/PLE/GDN
state and token positions.

The wire protocol permits n_keep=0, but the existing real
core::Verifier::commit() requires n_keep >= 1. A real final-stage executor
must reject a zero prefix until correct restoration is implemented.
Do not acknowledge a no-op as restoration.

## Validation and remaining work

Tests in tests/final_pipeline_tests.cpp use genuine forked and isolated
CPU stage processes with both socket and shared-memfd modes. Coverage:
three-stage activation-to-head transport, repeated windows, zero/partial/
full prefixes, optional logits including signed zero, invalid prefix,
nonfinite failure, wrong result identity, failed final commit, wrong
final/activation handshake and mismatched prefix ACK. No GPU involved.

Local compilation and execution used GCC, Clang, ASan and UBSan against a
local CPU protocol shim; the exact repository source must additionally be
compiled and tested on Oberon against the real v4 protocol implementation.
The existing full CTest suite, root engine, CUDA and HIP hardware have not
been exercised in the ChatGPT container for this increment.

## Oberon agent handoff

Worktree: /home/chris/workspace/Strata-heterogeneous
Patch/log staging: /home/chris/tmp/strata-heterogeneous

Do NOT inspect, search, build or modify /home/chris/workspace/Strata-dev
or /home/chris/tmp/strata-dev. These are unrelated tasks. Do not run
GitHub Actions. Keep main untouched.

Start by building/running the experimental CTest suite. Then implement
a real loaded-model final-stage verifier driver in an independently
supervised CUDA worker (without changing the default engine code).
Run split/unsplit numerical parity only after owner-approved NVIDIA GPU
recovery; preserve the already collected nvidia-bug-report.log.gz.

Return actual build/test logs, discovered defects, and the new integration
commit SHA. Do not claim CUDA/HIP inference until hardware tests pass.
