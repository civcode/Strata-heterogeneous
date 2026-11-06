# Conversation transfer audit

Local audit on `debug/isa-guard-4349`, R9700/gfx1201, ROCm core-10.1. Routine validation uses
`HSA_ENABLE_SDMA=0`. This audit does not establish the cause of the November 5 SDMA fault.

## Scope and findings

The HIP build disables `STRATA_CKPT_ASYNC`, `STRATA_PARK_ASYNC`, checkpoint gathering and the park VRAM
slot path, regardless of those environment settings. The two background D2H loops therefore do not run on
this build. Normal HIP snapshot saves used `cudaMemcpyAsync` directly into pageable vectors and segmented
buffers, followed by checked stream synchronization. That is a documented API usage, not proof of a caller
lifetime violation. The QSA reproducer nevertheless returned wrong bytes through the pageable readback path;
pinned rereads of unchanged device results were correct. The exact runtime/driver failure remains unknown.

This patch avoids that exposed path for HIP snapshot capture, disk-source readback and verification. It does
not change H2D restoration or inference transfers elsewhere. It does not fix restore resource preflight.

A separate static-analysis finding: even with background parking disabled, `conversation_park_async_begin`
wrote the process-wide `items`, `active` and `used` fields on every save. Independent concurrent saves could
race on that state. It now returns before those writes when the feature is disabled. Concurrent captures
have exact-byte regression coverage; there is no GPU ThreadSanitizer result.

## Capture and restore sequence

```text
park_current / park_source
  estimate/admit host memory
  move optional reusable storage and outgoing checkpoints into local capture inputs
  choose owning device (primary, then each later stage)
  synchronize producer work on that device
  snapshot_save builds a separate SavedConversation
    checkpoint: device running state -> pinned piece -> final vectors (HIP)
    K/V: authoritative device or mapped-host pools -> pinned piece -> host segments (HIP)
    each piece: submit -> checked stream completion -> CPU memcpy
  return image only after every synchronous transfer succeeds
  repeat for later stages
  put complete composite image into ConversationCache

restore selection (whole or borrowed prefix)
  validate images / targets
  existing whole-image 64 MiB/readiness heuristic (not a reservation)
  preserve outgoing branch; may move checkpoints and evict/replace cache entries
  whole restore: kvg_ensure may change K/V allocation before restoring
  synchronize producers on the owning device
  host segments (or a matching CUDA VRAM slot) -> session buffers
  checked private-stream synchronization before reusing a source
  residency refresh; restore running state; update CPU positional metadata
  later stages in order
  any restore transfer failure: inference operation returns an error, not permission to decode partial state
```

`ConversationBuffer` payloads are ordinary host allocations, in segments of at most 16 MiB. Growing a buffer
never relocates an existing payload. Directory movement is not payload movement. In synchronous HIP captures,
no transfer still references final destinations when save returns. Move, eviction and cancellation after return
therefore cannot race an outstanding capture. Sources remain owned by the quiesced session during the call.
The code does not support concurrent mutation of the same session while saving it.

`park_source` inserts an image only after primary and later-stage saves succeed. Failure on a later GPU returns
before `conversations.put`. This was inspected in source, not tested with a complete serving model. Reusable
storage and moved checkpoints can be consumed on a failed save; this is not rollback of outgoing ownership.

## Priority 1 transfer inventory

All ordinary copies use the calling thread's device-specific private nonblocking stream. `OnDevice` at the
program layer selects the stage device; core snapshot APIs operate on the current device. Streams and events
remain thread-owned until thread exit, when they return to the per-device pool. Buffers are not owned by the pool.

| Location / operation | Direction and extent | Ownership / memory | Ordering, completion, errors | Assessment |
|---|---|---|---|---|
| `checkpoint_save`: GDN, PLE, tails, dead row, block positions | D2H; checked session sizes and per-layer offsets | Session device arrays -> owned pageable checkpoint vectors | Caller quiesces producers; completed private-stream copy per part. HIP now stages in pinned pieces; failures propagate | Exposed pageable assumption mitigated on HIP; no success-path lifetime defect demonstrated |
| `ckpt_gather_small` (CUDA opt-in) | D2D gather then D2H; sum of validated component sizes | Mutex-protected GPU stage and pinned host buffer -> final vectors | Same private stream; final nonqueued copy synchronizes before CPU reads | Normal completion ordered; early failure after queued gathers needs separate failure-path review |
| Checkpoint async gather (CUDA opt-in) | D2D GDN then D2H in 4 MiB pieces | Global GPU stage/VRAM slot and pinned host buffer; raw destination pointer into checkpoint vector | D2D completes before worker launch; worker records/waits event before CPU memcpy; future joined before staging reuse | Disabled on HIP. CUDA error drain, checked device selection and failure propagation remain open |
| `conversation_kv_save`, ordinary path | D2H or mapped-host read; checked layout, visit ranges and retained prefix | Authoritative pool -> owned `ConversationBuffer` segments | Snapshot entry synchronizes producers; each transfer completes before visitor returns. HIP pinned staging now covers these reads | No final-host-buffer in-flight reference after HIP return |
| K/V async stage/slot gather (CUDA opt-in) | D2D, queued; part/segment offsets | Session pools -> global GPU stage or tagged per-device slot | `async_launch` synchronizes private stream before worker reads stage; abandon attempts a drain | Disabled on HIP; abandon ignores synchronization failure |
| Park async worker (CUDA opt-in) | D2H in alternating 4 MiB pieces | GPU stage/slot -> two pinned buffers -> raw final segment pointers | Event per piece; waits before CPU memcpy and pinned-buffer reuse | Normal ordering is explicit. Worker error exits and publication status need more work |
| `conversation_kv_restore` | H2D, host/mapped-host copy, or D2D slot hit; validated full layout | Image remains alive -> pre-existing session pools | Host fallback joins pending park; each nonqueued copy has checked stream synchronization; residency kernels follow | H2D lifetime correct on successful return; later failure can leave partial state |
| `conversation_kv_restore_prefix` | Same, limited to validated prefix sizes | Donor retained -> live session | Same completion contract; donor remains cached | No rollback on transfer failure; caller returns error |
| `conversation_checkpoint_restore` | H2D/D2D; validates all component sizes | Joined checkpoint or matching same-device slot -> live running-state arrays | Device sync before copies, each copy stream sync; final device sync checked | CPU positional fields updated after copies; error is not a valid restore |
| `conversation_checkpoint_copy` | D2D, checked matching session carves | Two existing sessions; no transient host storage | Nonqueued copies complete; queued caller must synchronize before use/release | Explicit queued contract; partial failure is not transactional |
| `conversation_kv_copy` | D2D or mapped pools; matching layouts | Existing source/destination sessions | Nonqueued copy completes; queued streamed refresh uses record/wait on private/legacy streams | Event creation/record/wait checked; capture and null-stream dependency hazard remains separate |
| `conversation_kv_source` disk callbacks | D2H/mapped host; part/offset/count bounded | Captures non-owning pool pointers; destination belongs to file writer | Read callback completes before return; caller must keep source session quiesced and alive for file write | HIP readback staged; source lifetime is an explicit API requirement |
| `conversation_kv_verify` | D2H/mapped host in 64 KiB comparisons | Existing session -> local scratch array | Pending park joined, validated ranges, checked completed transfer before comparison | HIP readback staged; verification failure propagates |

Byte arithmetic uses `product`, `add`, checked geometry and `ConversationBuffer::visit` bounds. Stage/slot
round-up arithmetic and assumptions about caller device selection were inspected, but no overflow/device-fault
injection proof was produced for every optional CUDA path.

## Readback implementation and failure policy

`conversation_readback.hpp` owns one portable pinned piece per copying thread, not a second full snapshot.
Successful calls reuse it; thread exit releases it. Maximum retained pinned payload is 4 MiB per thread that
has performed a HIP readback. Each piece adds a CPU memcpy and one stream wait; throughput was not benchmarked.
No device-wide synchronization was added by this patch. Existing producer synchronization remains necessary.

Allocation failure returns an error before submission. There is no unsafe pageable fallback. Submission errors
are followed by a stream drain because an error is not evidence that no work was queued. A failed completion
never copies that piece into final storage. If completion is unknown, staging is quarantined (not freed or reused)
until process exit: one 4 MiB allocation can remain per such failure. This favors memory safety over reclaiming
storage after a GPU error. Earlier completed chunks can have landed in a *temporary* image; save does not publish
that image. This mechanism does not prove that an asynchronously faulted GPU context can safely continue inference.

CUDA's normal snapshot transfer path is unchanged; the helper also compiles against CUDA API names, but no CUDA
toolchain/device is available here. HIP `cudaMallocHost` compatibility maps to `hipHostMalloc`; production staging
uses portable host allocation and checked `cudaStreamSynchronize`, not an event-only host-visibility assumption.

## CUDA-only async risks still open

The following are static-analysis concerns, not an explanation of the R9700 incident:

- `conversation_checkpoint_join` and `conversation_park_join` log a worker error and discard its future result.
  Readers do not receive that error. A failed background copy can therefore leave an image with incomplete bytes.
- Worker error exits do not reliably drain the stream after a failed event record. Device selection is unchecked.
- Last-member `pending` tokens protect ordinary destruction only when the deleter actually runs. Default memberwise
  assignment replaces payloads before assigning the token. Deep-copying an in-flight image also reads payloads
  before completion; sharing a token is not the same as sharing payload ownership.
- Async launch followed by host allocation failure while constructing lifetime tokens needs an exception-path review.

These paths need an image-specific, error-bearing completion owner, wait-before-copy/replacement semantics, and
fault-injection tests before they can be called safe. This patch does not redesign or enable them. Their lack of
HIP execution does not validate them for CUDA.

## Priority 2 / 3 sampling

- `generate.cpp` sampled-token D2H to stack storage checks submission and stream completion before reading/return.
  Adaptive gate landing storage has a separate pinned lifetime. No stack-use-after-return was demonstrated.
- `prefill.cpp` KVA residual capture uses a pageable vector, same producer stream, checked stream synchronization,
  then file output. It retains pageable-readback exposure; optional parity/dump reads are not blanket-replaced.
- `q36_engine.cu` hybrid routing destinations (`h_ids`, `h_h`, `h_cnt`) are pinned engine-owned allocations. GPU
  writes and event recording share the inference stream. The CPU event-query loop treats errors other than
  not-ready as completion: this is an unchecked-error concern, not validated by this patch.
- `ds4_model.cu::snap_into` resizes checkpoint vectors, submits on each owning device's producer stream, then waits
  each stream before returning. There is no normal-success local-vector lifetime violation. Submission, device
  selection and synchronization errors are ignored; `c.pos` is set before success. That path needs error-bearing
  checkpoint publication and targeted DS4 fixtures, separately from the Qwen conversation subsystem.
- DS4 diagnostic dumps use local pageable vectors and stream waits; other expert exchanges use long-lived pinned
  staging and stream/event handoffs. This is sampling, not a complete DS4/Q36/expert-exchange audit.

## Regression coverage and limits

`conversation_readback_test` exercises the production helper and snapshot API with small model-independent buffers:

- Exact pageable and pinned readbacks, 4 MiB boundary plus a tail, offset destination and guard bytes.
- Concurrent captures on two devices (or two workers on the only visible device); independent patterns.
- Pageable H2D restoration followed by exact D2H verification.
- Whole GDN snapshot capture, ownership move/replacement, restore and exact bytes.
- Injectable fake operations hold submitted data until synchronization; assertions reject staging reuse/free while
  in flight. Allocation, submission and completion failures are tested without device faults.
- ELF/HIP link wrappers inject pin, copy and stream-sync failures in the *actual* capture call, both before and after
  the first chunk. A previous image remains unchanged. Restore failure is checked after live GPU bytes changed:
  it returns `transfer_failed`, not `restored`. Wrappers drain real work before injecting a reported sync error.

`conversation_capture_formats_test` reuses the existing standalone snapshot fixtures without changing the build's
`STRATA_BUILD_CONVERSATION_TESTS=OFF` setting. It covers K/V formats, streamed/ring authoritative pools, prefix
restores, retained segments, session files and invalid images. It is not an async CUDA validation run on HIP.
The fixture is serial because it also compares device-wide free memory across restores. External GPU workloads
can still interfere with that diagnostic assertion. Concurrent readback correctness uses the separate readback test.

### Local validation

All runs below used `HSA_ENABLE_SDMA=0` and the existing optimized `build-r9700-debug` configuration.

| Check | Result |
|---|---|
| Full configured build | Passed |
| Requested conversation/snapshot/copy/cache/QSA filter | 7 passed, 0 skipped, 0 failed |
| Both new tests, separately visible physical GPU 0 and GPU 1 | 5 repetitions per test per GPU passed |
| Readback test with concurrent QSA register-forced test | 5 repetitions per test passed; readback also has its own simultaneous capture workers |
| Final broad run excluding `ple_parity` | 214 passed, 12 skipped, 0 failed (226 selected) |
| Initial broad run | 212 passed, 12 skipped, 2 failed: device-wide memory assertion in the new format fixture, plus `kv_q8_parity` |
| Isolated `kv_q8_parity` after that failure | 5 repetitions passed; parallel mismatch remains unexplained |

The format fixture's scheduling was corrected before the final broad run. The INT8 failure reported
`INT8 vs FP16 error 127.059 steps`; it was not diagnosed or fixed by this patch. A clean later run does not
establish that it cannot recur. The new failure tests were rerun after strengthening the prior-image validity check.

Broad skips: `io_priority_test`, `expert_cache_segmented_test`, `hip_pf_cover_test`,
`hip_prefill_wmma_gemm_parity`, `hip_prefill_hcd_exact_parity`, `hip_prefill_hipblaslt_gemm`,
`hip_prefill_rocblas_gemm`, `pinned_alias_test`, `hip_prefill_fused_moe`, `native_hadamard_test`,
`tq_cpu_parity`, `tq_cpu_parity_avx2`. `ple_parity` was excluded for missing reference captures, not passed.
Logs are under `/tmp/phase5-{build-final,targeted-final,broad-final,repeat-final-0,repeat-final-1,concurrent-final}.log`.

Unverified: concurrent diagnostic trace counters (`STRATA_PARK_TRACE`); full-model cancellation/eviction during
active CUDA background copies; actual GPU fault recovery;
composite serving restore failure on a later GPU; CUDA runtime; long-context throughput. No deliberate VRAM
exhaustion, SDMA re-enablement or driver-reset experiment was performed.

## Restore preflight (implemented)

Pinned D2H staging is not resource preflight; that hazard is now closed separately.

`conversation_copy_prepare(devices, count, error)` (`include/strata/core/conversation_copy_resources.hpp`,
implemented in `conversation_copy.hpp` / `conversation_state.cpp`) acquires this thread's copy stream **and**
ordering event on every participating device, reusing what it already owns or what the pool holds, creating a
new handle only when needed, and restoring the original device. Handles land in the existing thread-local set,
so no other thread can take them; they return to the pool when the thread ends. It submits no copy, records no
event and never touches the legacy stream, so it cannot conflict with another thread's graph capture. It returns
`ready`, `unavailable` (an allocation failure with the device restored: recoverable) or `fatal` (a poisoned
context, a failed device switch, or host allocation failure: not recoverable by rereading).

`ConversationRestoreAttempt` (`include/strata/program/conversation_restore.hpp`) is the serve loop's gate. It
validates nothing itself; `generate.cpp` validates the image, then calls `prepare()` **before** the snapshot
leaves the cache, before `park_current()`, before checkpoint ownership moves and before `kvg_ensure()`. The
borrowed-prefix path prepares the same conservative device set (primary plus every later stage), so the
blocked-borrow conversion to a whole restore cannot bypass preparation - `take_donor()` is only legal after the
boundary and takes the donor the same way.

Failure classes, encoded in the gate's phase rather than in comments:

| Stage | Outcome |
|---|---|
| prepare refused (`unavailable`) | phase `declined`: snapshot still cached, donor still parked, outgoing state untouched, behaves exactly like a request with no match, prompt is read |
| prepare `fatal` | phase `failed`: request aborted, `apply()` runs nothing |
| after `begin_mutation()` | any failed step sets `failed`; `apply()` then refuses to run further steps and `finish()` aborts the request, so no decode follows a partial restore |
| all steps ok | `finish()` moves `mutating` to `complete` |

The 64 MiB `kRestoreFirstUseMib` / `restore_fits` heuristic is **removed**: a cold HIP stream measured ~146 MiB
(plus ~2 MiB for the first legacy-stream event wait), so the budget was too small; `copy_stream_ready` was a
snapshot another thread could invalidate; a failed `cudaMemGetInfo` admitted the restore; and borrowed restores
bypassed it entirely. Actual acquisition with checked errors replaces the prediction, and a failed memory query
is now irrelevant because nothing is predicted from it.

What preflight does **not** guarantee: later ordering (`order_after_queued`), transfer, residency-kernel or
synchronization failures remain possible and remain fatal after the boundary. Preflight makes the *predictable,
recoverable* resource failures happen before anything is mutated.

Tests: `conversation_prepare_*` (8 modes, real GPU, link-wrapped injection of stream/event/device/memory-query
failures), the gate's state machine in `conv_cache_test`, and the orchestration walk below.

## Restore orchestration (verified)

`conversation_restore_orchestration_test` (`src/program/`) drives the serve loop's restore half in its own order
against the real `ConversationCache`, the real `ConversationRestoreAttempt` and the real `resume_gate`, with
injected preparation and per-step failures. The serve loop itself is one very large function around a live model
and cannot be linked into a test, so the harness transcribes its call order and asserts the observable state at
every step. CPU-only: no GPU, no model, no VRAM.

Verified, with the `generate.cpp` line each step mirrors:

- **Order.** `prepare` (16013) → `drop-slot`/`begin_mutation` (16029) → `park` (16078/16100) → `kvg` (16195) →
  `restore` (16234) → `stage-restore` (16246) → `stage-sync` (16255) → `finish` (16413) → decode. Preparation
  asserts on entry that nothing is parked, no slot was copied, the K/V has not grown and the session bytes are
  untouched, so a mutation before preflight fails the test rather than passing silently.
- **Ownership.** A whole restore empties the cache; a borrowed prefix leaves the donor parked and findable by its
  serial with its own checkpoints; a blocked borrow converts through `take_donor()` and still prepares first.
  A failure of the borrowed restore itself leaves the donor parked and whole, with none of its bytes applied.
- **Refused preparation.** On the first device *and* on a later one, the snapshot stays cached, no restore
  mutation runs, the outgoing conversation and its K/V are intact, and the request reads its prompt. A `fatal`
  preparation aborts with a reason instead of falling back.
- **After the boundary.** `park`, `kvg`, `restore`, `stage` and `stage_sync` failures each abort the request:
  nothing later runs, `finish()` is never reached, no decode follows, and a half-written session is left not
  marked valid (`from_live` false, `resume` unchanged) rather than published as a good state.
- **Threads.** Every step records its thread id; preparation and every gated step share one. This matches the
  implementation: `prepare_copy_resources` fills the same `thread_local` set that `copy_nonblocking` and
  `order_after_queued` read, and the only `std::async` in these files is the opt-in `STRATA_PARK_ASYNC` host
  staging, which captures its stream handle explicitly and creates and destroys its own events - it never calls
  `private_stream`/`private_event`, so it cannot take a prepared handle.

### The declined fallback

Phase 6 stopped forcing `slot_source = -1` when preparation is declined, so the fallback needed proving rather
than assuming. `resume_gate()` (`include/strata/program/conversation_restore.hpp`) now computes `pin_sibling`
and `park_outgoing` - the two decisions at 16043 and 16055 - as one pure function, and `generate.cpp` calls it
instead of inlining them. Its inputs carry no restore-attempt state: a declined restore has no incoming image,
no borrow and an untouched `slot_source`, which are exactly the inputs of a request that never matched a parked
conversation. The test proves the equivalence end to end rather than by inspection: it runs the declined request
and a reference request whose prompt matches nothing, over `slot_source ∈ {-1, 2}` × `pin ∈ {0, resume}`, and
requires identical slot copy, parking, checkpoint mounting, `resume`, session bytes and decode outcome. So the
fallback is the batch slot when one was chosen, else the checkpoint chain (and `pin=N` still keeps the outgoing
conversation unparked); it is never a half-applied restore.

One correction this made visible: on a declined restore the outgoing conversation *does* still park. That is the
ordinary request's own decision, not a restore mutation, and the reference request does the same.

Still unverified: the live-model serve loop (whole and borrowed restores end to end, a real blocked-borrow
conversion, a real preflight refusal under VRAM pressure, a real post-mutation copy failure) and the CUDA path,
for which no local toolchain or device is available. Live-model fault injection was deliberately not attempted.
This work does not claim to fix the historical SDMA page fault or the CPU SIGSEGV.

## Sources

- [HIP memory management](https://rocmdocs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html)
- [HIP coherence control](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/coherence_control.html)
- [Related pageable-host report, not proof of the same cause](https://github.com/ROCm/clr/issues/291)
- [Historical Strata report](https://github.com/Niko1221/Strata/issues/4591)
