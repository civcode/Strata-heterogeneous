# Verifier request/commit handoff — protocol v4 (M1.9)

Status: **CPU-only protocol scaffolding**, not CUDA/HIP model execution.
No model weights, GPU transfers, expert pooling, logits, KV, GDN, PLE,
prefill or model-state restoration are integrated here.

## Why two phases?

Strata's `Verifier::run(T, tokens, pos0, pool, user, out, err)` needs token
IDs and a starting position. The accepted-token count is determined
**after** execution, then the owner calls `Verifier::commit(n_keep)`.
M1.9 adds `WindowSpec {pos0, token_ids}` plus `commit_pending(n_keep)`.
It does not change the float32 `R | bo | inject` activation layout.

Wire protocol **version 4** (not compatible with v3) retains the 56-byte
header and introduces:

- `window_spec`: sent before the activation frame to every stage. Payload
  is little-endian pos0 (int64, 8 bytes), token count (uint32, 4 bytes),
  then token IDs (int32, 4 bytes each).
- `commit_prefix`: sent only after all stages finish provisional execution,
  carrying n_keep (uint32) in the range [0,window rows].
- `committed_prefix`: each stage must echo the exact accepted-token count.
  A mismatch poisons the session.

Windows require one nonnegative token ID per row, nonnegative position,
and overflow-safe position arithmetic. Protocol ceiling: 8192 rows.
Real GPU workers must also enforce vocabulary and `Verifier::max_t()`.

## Host-side use

`PipelineSession::initialize()` must succeed before any execution:

```cpp
Frame input = /* T valid activation rows */;
WindowSpec request{pos0, token_ids}; // T == token_ids.size()
Frame provisional = session.execute_window(input, request);
// Inspect acceptance results from the final real model stage.
uint32_t n_keep = /* number of tokens accepted, 0 through T */;
Frame committed = session.commit_pending(n_keep);
```

Never publish `provisional` as committed model output. The old
`run_window(Frame)` API remains for CPU mock compatibility and immediately
commits the entire window; it is not sufficient for speculative inference.

Workers call `StageExecutor::begin_window` before execution and
`StageExecutor::commit_prefix` only on commit. `DeviceStageExecutor`
forwards to `DeviceDriver::begin_window` and
`enqueue_commit_prefix`/`wait_commit`. The default virtual methods
**reject** metadata rather than silently dropping it or turning a partial
commit into a full commit.

## Fail-stop properties

- Only one provisional window may be in flight.
- All IDs, geometry, stage identity and plan fingerprint must match between
  the spec, data frame, commit and acknowledgement.
- An invalid n_keep greater than T is refused before any commit is sent;
  a corrected count may be retried.
- On partial commit, worker failure or mismatched ACK, abort all stage
  sessions. The protocol does not implement rollback across GPU workers.
- Socket-data and shared-memory mode both work; trusted local IPC only.

## Oberon coding-agent handoff

Use only `/home/chris/workspace/Strata-heterogeneous` for the working
tree and `/home/chris/tmp/strata-heterogeneous` for handoff logs/patches.
**Do not access the two unrelated Strata-dev directories.**

Build and test this experimental module locally, then inspect the actual
`include/strata/core/verify.hpp` and engine implementation. Verify
token IDs, positions, accepted prefixes (including zero), GPU state
ownership, expert pool callbacks, and final-stage logits. Return real
CUDA/HIP test logs, the integration commit and any API differences.
CPU-only tests are not evidence of GPU inference correctness.
