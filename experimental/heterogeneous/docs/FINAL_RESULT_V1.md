# M1.10 — Final-stage result framing (host-only)

Status: a CPU-tested, **opt-in standalone codec**. Not yet integrated with
`PipelineSession`, `serve_worker_stateful`, `Verifier`, or a model-loading
CUDA/HIP worker. The existing activation/commit protocol remains **v4**.

## Problem and contract

A non-final stage sends the next stage the float32 `R | bo | inject`
activation handoff. The **final** stage instead needs to return actual head
predictions; it has no subsequent activation consumer.

`FinalStageResult` in `include/strata_heterogeneous/final_result.hpp`
contains:

- Exact final stage index, request ID, window ID and plan fingerprint.
- One nonnegative, vocabulary-bounded predicted token ID per window row.
- A sorted, deduplicated list of rows whose head reported nonfinite logits.
- An **optional** row-major float32 logits matrix (rows × vocabulary) for
  diagnostics, with each float's IEEE754 bit pattern preserved.

The result is **provisional** until every execution stage confirms the
accepted-prefix commit. Nonfinite diagnostic markers must block acceptance
in the eventual coordinator; this standalone codec does not implement
sampling or validation of the model's actual head outputs.

## Independent wire packet v1

This is a **different** magic and version from the existing v4 frame.
It cannot be read by `receive_frame`. Never interleave it on a v4 control
socket until the state machine is updated to negotiate and expect it.
The helpers currently require a dedicated local AF_UNIX stream.

64-byte little-endian header:

| Byte offset | Field |
| --- | --- |
| 0 | magic `STFR` (uint32) |
| 4 | result-format version = 1 (uint16) |
| 6 | flags (uint16; bit 0: logits included) |
| 8 | final-stage index (uint32) |
| 12 | row count (uint32) |
| 16 | vocabulary size (uint32) |
| 20 | reserved zero (uint32) |
| 24 | request ID (uint64) |
| 32 | window ID (uint64) |
| 40 | plan fingerprint (uint64) |
| 48 | payload size (uint32) |
| 52 | FNV-1a checksum (uint32) |
| 56 | reserved zero (uint64) |

Body: `rows` int32 predictions; uint32 nonfinite-row count; that many
uint32 row indices; optionally `rows × vocab_size` float32 logits.

The checksum covers header + body excluding its own four bytes. Size is
capped at 16 MiB of payload, and rows at 8192. Declared length and shape
are validated before memory allocation. FNV-1a detects only accidental
corruption; the socket must be local and trusted.

The `FinalResultGate` holds a provisional result and prevents its
`take_committed()` publication until the coordinator explicitly calls
`mark_committed(request_id, window_id)` **after confirming all workers'
matching commit ACKs**. A mismatch or abort permanently poisons the gate.
This helper is not a distributed transaction manager and cannot roll back
any already committed GPU stage state.

## CPU validation

The additional standalone CTest target
`strata_heterogeneous_final_result_tests` exercises identity checks,
bounds, overflow, optional logits and bit preservation (including NaN
payload/signed zero), corrupt/truncated frames, socket fragmentation,
deadline timeouts, and provisional/commit/poison state transitions.
It uses no GPU drivers or model weights.

## Next increment for Oberon

1. Merge this feature branch from its exact base commit, run the entire
   CTest suite and inspect the new CMake target.
2. Agree a v5 handshake/state machine for an **opt-in final-stage worker**
   that replaces its activation reply with a final result. Stage 0 through
   N-2 must still send activation handoffs as before.
3. Obtain predictions from the real final verifier and enforce its
   actual `n_vocab`, `max_t`, sampling/history, nonfinite guard and
   accepted-prefix semantics. Do not send fake activations as logits.
4. Call `mark_committed` only after every stage has acknowledged commit.
   Reject failed stages, timeouts, stale IDs and nonfinite results.
5. Run CUDA-only split/unsplit numerical parity once NVIDIA hardware is
   healthy, then implement HIP with independent worker executables.

Work only in `/home/chris/workspace/Strata-heterogeneous`; stage
patches/logs in `/home/chris/tmp/strata-heterogeneous`. Neither unrelated
`Strata-dev` directory is part of this task. Preserve any completed
`nvidia-bug-report.log.gz` privately before owner-approved maintenance;
this change does not touch GPU state, drivers or hardware.
