# M1.18 — Bounded supervised CUDA stage windows (GPU-free integration)


> **M2.9 update:** The historical I_APPROVE_* runtime environment-variable
> gate described below was removed. An explicitly launched native worker is
> the runtime opt-in. See [LIVE_GPU_HANDOFF_M29.md](LIVE_GPU_HANDOFF_M29.md)
> for the current manual live GPU test sequence. Earlier milestone notes
> below are retained for historical context.

**Baseline:** M1.17 `9b33ee648a827d17f4cf55ff4a1570deb57f1d24`.
**Scope:** CPU-only window admission, provisional output and positive-prefix
commit checks around both existing CUDA verifier servers. No new CUDA kernels,
model writes, GGUF search, profile inference, HIP inference or GPU startup.

## Why

The native `CudaVerifierDriver` and `CudaFinalVerifierExecutor` already
implement ordered H2D/execute/D2H/wait/commit operations, but the worker
protocol previously accepted any valid v4 `window_spec` without checking
stage-specific context, vocabulary, replay state, positive accepted-prefix
support or non-finite activations **before** invoking the verifier.
A malformed window can poison GPU state or an omitted PLE path can yield
plausible bad values. A failed worker must never return a committed reply.

## Integration

`WindowAdmission` is a host-only, one-stage, fail-closed state machine.
It is configured from the trusted, already-validated `CudaWorkerConfig`:
`initial_position=0`, `max_cells`, `n_vocab` and `max_t`.
It checks, before calling `begin_window` on either actual verifier:

- The stage's plan fingerprint, backend-local index, frame geometry,
  nonzero stable request ID, strictly increasing window ID and contiguous
  token positions since the last **completed** commit.
- `WindowSpec` row count, per-token signed IDs in `[0, n_vocab)`, context
  bounds and the 16 MiB activation cap. No row above `max_t` is admitted.
- Socket payload dimensions and absence of bogus metadata; for memfd,
  payload is empty and the shared slot's checksum is validated by the
  existing transport. After checksum verification the gate checks all shared
  activation bytes for finite IEEE-754 float32 values using `memcpy` bits.
- Before *publishing* any intermediate activation reply, the exact
  completed output byte length and every float32 value are checked.
- The final stage's STFR result remains subject to the preexisting
  `validate_final_result` and result-identity rules; only then does the
  protocol stage mark the window provisional.
- Only `commit_prefix` with `1 <= n_keep <= T` is permitted. The gate
  advances `pos0` only after the CUDA executor reports completed commit.
  No `commit(0)`, rollback, skipped positions, replayed windows, full-window
  legacy commits or unverified prefix acknowledgments are allowed.

On any invalid request, GPU error, nonfinite output, timeout, or commit
failure, the server abandons the window, poisons the gate, shuts down the
executor, sends one best-effort failure frame and returns. The coordinator
then poisons its session; the process supervisor owns kill/reap. Do not
reuse a poisoned session. A partially mutated memfd slot is never published
as a successful result.

The guard is passed **only by the real supervised CUDA worker**.
Existing mock and test `serve_worker_stateful`/`serve_final_stage_worker`
callers continue to omit the optional limits argument and retain the
original protocol behavior. This avoids changing the default `strata`
executable or any HIP worker.

## Safe diagnostics

`STRATA_HETERO_WINDOW_TRACE=1` logs concise stage/request/window/row,
position and phase metadata on `stderr` during a gated worker session.
It never prints token IDs, activations, GGUF paths, expert IDs or logits.
Disabled by default. The normal worker opt-in gate
`STRATA_HETERO_CUDA_LIVE=I_APPROVE_CUDA_RUNTIME` is **unchanged**.
Neither setting should be enabled to perform CPU-only build validation.

## Tests

The standalone experimental CTest suite gains two CPU-only tests,
**20 total expected**:

- `strata_heterogeneous_window_admission_tests` compiles the same pure
  production gate and exercises bounds, sequencing, shared/socket shapes,
  nonfinite bit patterns, zero/oversized commits, double commits, replay,
  poisoning and redacted diagnostics.
- `strata_heterogeneous_window_admission_process_tests` runs real forked
  AF_UNIX v4 stage servers with CPU-only executors. Both socket and shared
  handoffs cover positive prefix ACKs, zero-keep refusal, wrong next
  position and a nonfinite output that must never be published.

Run GCC, Clang, ASan, UBSan and CUDA-enabled CTest on Oberon using
`cmake --build "$build" --parallel 28`; also native compile/link the optional
CUDA worker and whole-archive adapter with actual `strata_engine`. Do not
use GitHub Actions. Preserve the untracked pool probe and the owner's stash.

## Remaining validation and limitations

**This milestone does not claim a successful live CUDA inference window.**
After separate owner approval, an operator can validate an authorized
original GGUF + Q2_0 pack, the live NativeDense QKV path, layer-1 PLE,
finite stage output, positive-prefix commit, position progression and
numerical parity against the *same* unsplit model. Further investigation
is needed for prompt prefill, GPU watchdog/driver-fault teardown,
zero-prefix rollback and cross-CUDA/HIP numerical parity. Never attempt
NVIDIA reset or reboot as part of this CPU-only handoff.

Only explicitly authorized model paths may be read; do not visit or modify
any `Strata-dev` directories.
