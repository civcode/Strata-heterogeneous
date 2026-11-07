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
Successful calls reuse it; thread exit releases it. Successful steady-state readback retains 4 MiB per thread. Each piece adds a CPU memcpy and one stream wait; throughput was not benchmarked.
No device-wide synchronization was added by this patch. Existing producer synchronization remains necessary.

Allocation failure returns an error before submission. There is no unsafe pageable fallback. Submission errors
are followed by a stream drain because an error is not evidence that no work was queued. A failed completion
never copies that piece into final storage. If completion is unknown, staging is quarantined (not freed or reused)
until process exit: one 4 MiB allocation can remain per such failure. Repeated failures or new copying threads
can accumulate quarantined allocations; there is no process-wide cap. The 4 MiB steady-state figure is not a
failure-path memory bound. This favors memory safety over reclaiming storage after a GPU error. Earlier completed chunks can have landed in a *temporary* image; save does not publish
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
  writes and event recording share the inference stream. The route query now distinguishes success, not-ready
  and error. Errors take the existing fatal `CK` path before CPU reads. Focused fake/HIP checks and HIP/CUDA
  compilation cover the gate, not Q36 model inference.
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
event and never touches the legacy stream, avoiding new legacy-stream dependencies during graph capture. It
returns `ready`, `unavailable` (a device allocation failure or host handle-storage `bad_alloc`, with the original
device restored) or `fatal` (other runtime errors, or failure to restore the original device). A failed device
selection is recoverable only when it reports allocation failure and the original device remains selected.

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

## Restore orchestration: modeled tests and real inference

`conversation_restore_orchestration_test` uses the real cache, restore gate and `resume_gate`, but transcribes
`generate.cpp`'s call order. It does not execute the serve loop. Its injected preparation, parking, K/V growth,
copy and synchronization failures test the modeled boundary and ownership rules. Moving a production call
outside the gate would not necessarily fail this test.

In the model, preparation precedes cache take and outgoing parking. A declined attempt preserves the selected
snapshot; an ordinary request may still park its outgoing state. Whole restore takes the image; borrow keeps
the donor; conversion takes the donor after preparation. Failed modeled steps prevent later modeled decode.
The actual serve loop returns an error and exits the engine after post-mutation failure, rather than promising
that the partially written session can be reused in place. Live-model failure injection remains untested.

`resume_gate()` computes `pin_sibling` and `park_outgoing`. A declined attempt has no incoming image or borrow
and keeps the batch-slot selection. The test compares those decisions with a no-match request, including pin=N
and checkpoint/slot choices. Its fake byte arrays and decode flag do not prove actual prompt-reread equivalence.

Source inspection finds no thread migration between preparation and restoration. Copy streams/events are held
in the serving thread's TLS. Optional async parking captures its own stream and owns its events; this is not a
reservation for arbitrary work migrated to another thread.

### Real-model evidence and numerical limits

Local tests on two R9700s with Q2_0, int8 K/V and `HSA_ENABLE_SDMA=0` exercised whole restores and repeated
switches. Separate single-stage tests exercised a 1,280-token borrow within a 3,656-token prompt, donor reuse,
and host-cache-budget-driven conversion. The earlier 2,899-token attempt did not borrow. These success paths
do not validate live-model preflight refusal or fatal request-boundary behavior.

A controlled replay used exactly the same 19,523 archived prompt IDs. Live resume and restoration matched all
48 generated IDs and reported log probabilities. Ordinary full reread differed at output index 32, but its log
probabilities already differed at index 0. Disabling the drafter did not change either result. All experts were
resident and no expert work was offloaded in this reproduction.

A full reread with zero reused tokens and an explicit prefix boundary at token 19,473 matched live resume's
48 IDs and reported probabilities. A separate one-token probe also matched the primary stage's GDN/K/V hashes
between live resume, restoration and that boundary-matched reread; ordinary reread differed. This identifies
prefill segmentation as a numerical variable, not a demonstrated snapshot-copy defect. It is not a proof of
second-stage state equality or universal bitwise equality across prefill schedules. The smaller 889-token
short-read difference was also reproduced and disappeared with `--short-read 0`; no production default changed.

CUDA execution remains unvalidated. Kernel MES timeout diagnostics and the historical SDMA page fault and CPU
SIGSEGV remain separate open investigations. Successful requests and a green CTest run do not establish that
these faults are fixed.

## Phase 10D: expert failure paths

| Path | Classification | Boundary / remaining limit |
|---|---|---|
| `RemoteExperts::copy_cached_blobs` | Demonstrated defect — corrected | GPU writes now target an owned landing arena. Failed batches publish no hits and cannot fall back into an active caller destination. |
| Remote cache teardown after read failure | Missing guarantee — corrected | Checked compute/read stream fences; failed cleanup detaches the cache owner and retains landing/raw allocations. Poison blocks retry, reopen and adaptive replacement. |
| `Stager::work` / `issued_one` | Demonstrated defect — corrected | Checked waits, submission and recording; worker errors propagate before completion publication. Failed independent cleanup retains pinned and pageable sources. |
| Prefill borrowed owners after staging failure | Missing guarantee — corrected in engine callers | Terminal status exits the engine without unwinding external cache/session owners, before cancellation/refill. Other library callers must honor that raw-loan lifetime contract. Final caller change compiled but GPU rerun stopped below. |
| Q36 route query | Demonstrated defect — corrected | Only successful completion permits CPU reads; errors use the existing process-fatal policy. No Q36-model runtime claim. |
| `PeerExperts::finish` | Safe consumption gate; failed cleanup unresolved | Query errors are rejected before CPU reads. `close()` ignores fences; direct input/output are caller-owned mapped buffers, so leaking the peer alone would not fix them. Needs coordinated caller retention. |
| `RemoteExpertOpt` teardown / generic cache startup | Missing guarantees — unresolved | Optimizer teardown ignores stream errors. Cache startup uses default-stream copies, and `verify_slot` has a local D2H vector on error. Dedicated owner/error tests still needed. |

See [the focused tests](../tests/prefill/TRANSFER_FAILURES.md). This is not a completed audit of the 133 syntactic
production D2H/Default candidates. Healthy helper-cache reads add a pinned landing copy; the default-disabled
helper-source path stays opt-in. No global device fence was added.

**Validation stop:** a broad SDMA-off run produced a real MES `INVALIDATE_TLBS` timeout. Further GPU tests were
stopped. Focused earlier HIP tests passed, but broad completion, final engine-exit integration, performance,
Q36 inference and native CUDA runtime validation remain outstanding. Keep `HSA_ENABLE_SDMA=0` for unattended
operation; it does not replace the ownership fixes or guarantee freedom from MES faults.

## Sources

- [HIP memory management](https://rocmdocs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html)
- [HIP coherence control](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/coherence_control.html)
- [Related pageable-host report, not proof of the same cause](https://github.com/ROCm/clr/issues/291)
- [Historical Strata report](https://github.com/Niko1221/Strata/issues/4591)
