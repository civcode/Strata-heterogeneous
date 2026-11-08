# Source changes and lifetime rationale

## R4 — Prefill

Paths: src/prefill/prefill.cpp, include/strata/prefill/host_group_readback.hpp,
include/strata/prefill/expert_stager.hpp, src/program/generate.cpp,
include/strata/core/transfer_terminal.hpp.

Before: KvDrain and HelpScope discarded wait errors. Final compute/copy/KV waits could return false without setting a status checked by terminal_transfer_failure(). Cleanup could fail after the body had chosen a true return. Early cancellation did not consistently settle all borrowed streams.

After:
- Impl owns one sticky completion status included in the existing transfer_failed() predicate.
- Existing waits in chunk-major and layer-major runs observe status. A failed explicit wait makes one error-path drain attempt before throwing the existing StagerFailure; if that drain also fails, the process exits immediately, before run-local readback vectors can unwind.
- Optional KV/PLE and helper cleanup waits latch status without throwing. Every helper stream is attempted.
- An early-return guard settles compute/copy and existing optional KV/PLE streams. Clean cancellation stays non-terminal if these waits succeed.
- drain_pipeline() joins stages and then checks terminal_transfer_failure(), including destructor-time failures after a successful body.
- release() stops CPU workers and retains Impl after completion failure. lm_release_chain() continues to suppress release on terminal status.
- Peer timing device selection is RAII-managed so a new failure exception cannot bypass device restoration.
- The existing serve/generate process policy is factored into transfer_terminal(); it flushes ERR and uses _Exit, not stack unwinding.

Owners: Prefill retains its private staging owners; cache/session VRAM loans remain caller-owned. A terminal library return never authorizes their release. The known generation boundaries terminate before refill/ordinary owner teardown.

Tests: prefill_completion_test (cleanup, all optional cancellation streams, later-success stickiness, null compute, healthy guard disarm); HIP real-model one-shot final-copy failure and persistent reported failure. The former reaches the actual serve boundary despite later successful drains. Neither emits T or DONE. KV/helper fault injection is component-level, not native multi-GPU engine coverage.

## R5 — synchronous completion and failed async issue

Paths: src/core/conversation_copy.hpp, conversation_readback.hpp,
conversation_completion.hpp; shared transfer_terminal.hpp.

Before: generic H2D/D2D and CUDA readback skipped the fence on issue error. A completion error returned through restore/capture owners. The restore gate prevented decode but did not itself retain operands.

After:
- finish_copy() drains nonqueued transfers and every reported issue error, including queued issue errors.
- A successful drain permits safe operand cleanup, but preserves the original issue error. The restore-attempt gate still rejects the transaction and suppresses subsequent operations.
- A failed completion fence terminates in the copy layer, before automatic snapshot/session/vector owners can unwind.
- HIP staged readback records unknown completion and its production wrapper immediately takes the same terminal boundary. Its internal landing arena is still detached on unknown completion.
- order_after_queued() drains on event acquisition/record/dependency failure; inability to recover the ordering stream is terminal. Successful queued copies/order retain their previous overlap and progress-beat behavior.
- Resource admission/preflight and whole/borrow/conversion gate behavior are unchanged.

Operand map:
| Operation | Source | Destination | Failure policy |
|---|---|---|---|
| Whole/prefix H2D restore | Incoming image or guarded donor vectors | Session K/V/GDN/PLE storage | Positive drain or immediate no-unwind exit |
| Cached D2D restore | VRAM slot/donor | Session storage | Same |
| CUDA direct D2H capture | Session/device source | Candidate image storage | Same; failed candidate is not published |
| HIP staged D2H capture | Session or mapped source | TLS pinned landing, then image | Landing only publishes after issue + wait success; unknown completion exits before source owner release |
| Queued ordering error | Earlier queued-copy owners | Their destinations | Copy-stream drain before returning failure |

Tests: 14 modes for each compiled CUDA/HIP fake path, including production copy headers and the real cache/restore gate for whole, borrowed and converted attempts. Eight Linux native modes verify exact H2D/D2D/D2H bytes, reported accepted-work errors, ordering failure and no-unwind subprocesses. Updated native readback tests exercise actual snapshot capture/restore terminal exits.

Red evidence: corrected fixture compiled against checkpoint headers fails issue-drain with exit 134 (r5-red-fixed-*). The first red attempt had a cache-fixture mismatch and is not valid defect evidence. Corrected tests pass.

## Directly related R6 closure

Paths: src/core/conversation_state.cpp, conversation_snapshot.cpp and conversation_completion.hpp.

Before: checkpoint/park joins logged worker error codes and returned void, allowing a failed image to look usable. Worker error exits could destroy events while DMA completion was unestablished.

After: device selection is checked; worker error paths drain before event destruction; an unknown drain exits immediately. A returned worker error causes the existing void join to terminate, rather than admitting a failed image. Park staging abandon/launch also require completion before slot release/publication.

CPU async-future join success/failure subprocesses cover the shared production join policy. This is a fail-closed correction, not a per-image recovery API. CUDA hardware async worker execution, worker C++ exceptions and broader queued-resource admission/owner interactions still need dedicated review and runtime coverage.

## CUDA 12.0 compatibility

Paths: CMakeLists.txt, include/strata/prefill/moe_fused.hpp.

Reproduced native_kernel<...,NT=256,...> formal parameters: 4216 bytes; CUDA 12.0 maximum: 4096. CMake defines STRATA_CUDA_SMALL_KERNEL_PARAMS for all host/device translation units with compiler <12.1. WideBatch then holds 128 pointers, and existing batch selection clamps to that room. No kernel is removed and arithmetic is unchanged. CUDA 12.1+ capacity is unchanged. Full CUDA application compilation/link now passes; numerical/performance validation on NVIDIA remains blocked.

## Costs and scope

No new healthy-path device-wide fence, allocation or CPU payload copy. New drains are error/cancellation-only; existing healthy completion waits remain. There is a small sticky-status observation at waits. CUDA <12.1 opt-in wide launches can require more batches; no performance claim is made. Tests/interposers are test-only. No unrelated R1/R2/R3 redesign or default async enablement was added.
