# Request-local resident MTP

Report 0048 qualified the bounded mixed Q2_0/Q8 resident buffered candidate
on Oberon: identical 32-token outputs to its short non-MTP reference,
33 proposed/18 accepted drafts per request, and variable-length A/B/A
reset. This is not a robust speed comparison or production admission.
The reference and MTP had different native overlap settings. No long
contexts or HTTP/cancellation qualification in 0048.

## Current candidate (256K configured short-request smoke in 0053)

- Exact Q2_0 pack/runtime identity admitted by the pinned MTP validator;
  target conversion checkpoint provenance remains unresolved.
- Mixed CUDA0/HIP0/HIP1 15/18/15 or HIP-only 24/24, grouped HIP overlap,
  shared transport, full target and draft experts, max_t=8, 1 GiB reserve
  per owner, exact aligned 2048..262144-cell worker capacity (256K configured capacity smoke passed; actual positions above 92 remain unverified), Q8 target and native Q8
  draft cache. Native drafter runs on terminal HIP1.
- The drafter now borrows its own device-local/mapped native embedding.
  It does not install HIP1's pointer as the process-global target
  embedding; this is essential for layer-zero target execution on HIP0.
  Legacy non-service diagnostic behavior is unchanged. The drafter and
  its graphs are destroyed before its embedding is released.
- Variable prompts and completion budgets 1..256 within the exact admitted capacity.
  Fixed spec=4, confidence threshold 0.5, serial prompt chunks 1..8.
  Buffered or incremental streaming is explicitly selected.
- Serial prompt prefill remains the default. Explicit native bulk is an experimental bounded candidate; see below.
- Above 8K requires explicit `long_context` and `workload_mode` (the existing
  native long-context allocation and deadline contract); completion stays
  capped at 256. No lookup, compact/fast pipeline, guarded direct,
  automatic restart, stochastic sampling or MTP
  diagnostic captures in this candidate.
- All-domain target prefix commit remains authoritative. Distinct final
  HELLO and request metadata ACK bind the actual prompt and budget.
  Grouped HIP-only has one protocol domain but two physical verifiers;
  its ACK is only sent after both verifier commits and the MTP observer.
  Near the physical KV limit the ordinary anchor path avoids speculative
  writes. Reset clears target and draft state; no physical zeroing of
  rejected suffixes is claimed.

## Incremental MTP streaming

New source capability `mtp_stream_cap=1`; HIP-only requires
`mtp_hip_pair_cap=1` as well. Streaming MTP readiness is
`HETERO_READY 1 MTP_STREAM`, not the legacy single-token STREAM marker.
After a successful all-owner commit:

```
HETERO_PROGRESS request window committed_position BATCH terminal count [token IDs...]
```

A checkpoint before the full prompt reports an empty nonterminal batch.
At the last prompt checkpoint, publish the first target prediction unless
it is a stop. For a committed decode window, omit the previously announced
anchor and publish newly accepted draft tokens and the verified successor
(non-stop), at most 5. Stop is metadata, not a visible token in the batch.
The C++ stream cursor checks ordered positions, immutable published prefix,
stop exclusions and output budget. Python verifies the exact batch size
against committed positions, request/window IDs, token vocabulary, terminal
status, and the final completion record. The capped final token is
withheld until completion validation. The existing generator yields tokens
individually; cancel/close during any batch yield drains a cancellation
ACK at the same checkpoint before subsequent request admission.
Private loopback SSE passed at 2K and 256K configured capacity. No guarantee of network-level latency or production cancellation yet.

## Native bulk prompt seed (2K qualified; larger capacity candidate)

Explicit `--native-prefill-chunk 1..64` with MTP now admits prompts under the same exact admitted combined prompt/completion limit. Above 2K, native bulk must be explicitly enabled. One token bypasses bulk. The target may emit up to 64 residual rows; the cursor subdivides them into synchronous native graph batches of at most max_t (8), with actual successors across every boundary. The terminal HIP prefill's fenced `on_chunk` consumes completed R[p] with the actual prompt token at p+1, synchronous draft prime before R reuse. No host residual path to the drafter, no global embedding publication. The final prompt token remains in the ordinary verifier; a successful all-domain bulk ACK is an empty streaming checkpoint, allowing cancellation before that final token. The worker validates the exact request-local prefix and position; adoption requires a fully drained target and draft cursor at N-1. Capabilities `mtp_context_cap=262144`, `mtp_bulk_prompt_cap=262143`, `mtp_bulk_chunk_cap=64` and `long_context_cap=262144` gate long-context MTP activation. Earlier 0051/0052 coordinators remain compatible within their smaller tiers. Prompt metadata is bound and validated once on the terminal cursor, rather than comparing the entire prompt on every residual chunk. The terminal callback checks exact chunk shape/position and device before priming. The native Q8 admission estimate now separately counts resident context-dependent allocations (shared-RoPE QSA state, identity rows and dense attention scratch) in addition to the original conservative cushion. The terminal prefill allocation preserves the existing reserve plus 512 MiB prime headroom while its scratch is live. These checks are allocation-boundary checks, not a measured peak. All owners additionally use allocation-boundary reserve and full-residency checks in service mode. Only full-prompt MTP request/ACK metadata uses the existing native prefill token ceiling (262144); ordinary verifier tokens, wire layout, per-piece R cap and proposal width are unchanged. The existing extended prefill deadline is required above 8K. This does not make dense draft attention linear-time or establish long-position correctness.

## Test configuration

Create a new private server JSON using the established 2K stage configs
and matching rebuilt binaries. Never edit the existing 256K configurations.
Example fragment inside heterogeneous:

```json
{
  "mtp": "/home/chris/workspace/Strata-data/mtp/rt",
  "spec": 4,
  "spec_min_p": 0.5,
  "max_context": 2048,
  "max_new_tokens": 64,
  "kv": "int8",
  "shared": true,
  "group_native_hip": true,
  "prompt_chunk": 8,
  "stream_tokens": true,
  "workload_mode": false,
  "long_context": false,
  "native_prefill_chunk": 0
}
```

Include actual coordinator/worker paths, plan, fingerprint, and ordered
2K configs. HIP-only needs only hip_worker. Omit pipeline_mode. CLI
accepts --mtp PATH --spec 4 --spec-min-p 0.5 --resident, plus
--stream-tokens for batch mode (or --write-server-config for CPU-only
config). Native opt-in is --mtp-resident-service --mtp-probe-rt ABS_PATH
--hip-split-overlap. Do not start alongside another active GPU engine.

## Validation and next work

0048: mixed buffered hardware passed. 0049: integrated host/CUDA/HIP builds and focused CTest 8/8 and Python 44/44 passed. HIP-only streaming A/B/cancel/A and mixed streaming regression passed with 46.88 s GPU process wall time, normal shutdown. Network HTTP and full state parity remain unverified.

0050: all three integrated builds, CTest 9/9 and Python 45/45 passed; short HIP-only bulk A/cancel/A and serial comparison, and mixed bulk smoke, passed. The report records 62.09 s GPU process wall time and normal closure. No HTTP or long-context MTP qualification.

0051: all three builds and focused CTest 9/9, Python 45/45 passed; 77-token HIP-only A/cancel/A, mixed bulk and loopback HTTP SSE passed in 50.08 s total GPU process wall time. Context sizing and terminal headroom checks observed. No production admission, measured peak or independent state parity.

0052: integrated host/CUDA/HIP builds, CTest 9/9, Python 45/45 and short 8192-cell-configured HIP-only A/cancel/A and mixed smokes passed. Both reached position 92, 30.62 s total GPU process wall time; no near-8K execution or peak-memory proof.

0053: 256K-configured HIP-only short A/cancel/A and mixed private HTTP SSE passed after repairing a stale long-context worker-mode gate and a stale test expectation. All three builds, CTest 13/13 and Python 54/54 passed. Successful native process wall time 34.39 s (initial safely refused attempt not independently timed). Actual positions reached 92; no high-position correctness, transient peak, long-prompt performance or production validation.

For an isolated 256K configured-capacity MTP smoke, use private copies of the exact 262144-cell full-resident Q2_0 stage configs, `max_context=262144`, `native_prefill_chunk=32` (or 64), `workload_mode=true`, `long_context=true`, and `max_new_tokens<=256`. Keep the same runtime and per-owner reserve. This is an opt-in candidate, not a change to existing 256K ordinary services. Use short prompts until an actual long-position check is authorized under the test budget.

## Failure retirement (0054 candidate)

MTP failure frames now strip the request-only completion-budget extension so metadata refusals retain their error reason. A failed final MTP domain exits nonzero after executor retirement; EOF during an uncommitted MTP window is a failure. A zero accepted prefix is rejected by the coordinator before any commit is sent. No attempt is made to roll back partial target commits or reuse a failed session. Python validates integer request identity, refuses speculative advances at the physical tail, clears stale request timing on failure, and shares one cooperative retirement check for failed launch, restart and close (coordinator exit, output reader joined and original process group absent). Group existence detection uses signal zero only. MTP automatic restart remains forbidden. Normal MTP quit uses its configured cooperative drain deadline rather than the generic 1.5s grace. An unresolved process group is an operator-inspection error, not proof of GPU failure recovery. New forked CPU mock failures cover metadata, prefix, proposal, reset, post-commit observer and zero-keep/EOF; these integration checks are pending on the local checkout. Chat AI isolated exact-blob Python retirement/stream suite passed 10/10. No GPU fault injection or high-position run performed.

0054: integrated builds passed, CTest 15/15 and Python 62/62. Short 256K-configured HIP resident A/cancel/A and private SSE completion passed in 37.82 s total GPU-process wall time. The adapter and driver verified the retired coordinator group absent. CPU mocks preserve the exact failure explanation and forbid reusing a poisoned MTP domain. No actual HTTP disconnect, high-position correctness, native GPU fault recovery or peak VRAM qualification.

0055 candidate: explicit Service iterator ownership through OpenAI, Responses and Anthropic encoders, deterministic closure in SSE handlers, and explicit queued-disconnect terminal accounting. Cancellation remains cooperative: native bulk execution must reach an existing committed checkpoint before a cancel command can be sent. No chunk-interrupt or forced GPU termination is introduced. CPU loopback test uses a simulated native transport with event gates and the real resident adapter and HTTP Service to check that a second HTTP call cannot get native admission before the first cancellation ACK. Integrated tests and native HTTP disconnect verification pending.

0055: real HTTP disconnect at the genuine native bulk checkpoint, cancellation ACK before queued recovery on the same coordinator, and verified cooperative retirement passed in 17.54 s total native wall. CPU suites 37/37 and 119 tests (3 skipped). Actual committed positions reached 62, not high-position qualification. The checked-in mixed launcher profile explicitly enables MTP with chunk 32 and budget 256; its launcher binds the existing ordinary 0044 target reference (chunk 128) while validating MTP profile settings separately. The ordinary HIP-only profile and original reference requirements remain unchanged. Run `./run-q2_0-heterogeneous.sh --prepare-only` to check real local paths and native capabilities without starting GPU workers.
