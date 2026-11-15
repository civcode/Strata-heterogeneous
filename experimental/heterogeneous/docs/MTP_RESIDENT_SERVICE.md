# Request-local resident MTP

Report 0048 qualified the bounded mixed Q2_0/Q8 resident buffered candidate
on Oberon: identical 32-token outputs to its short non-MTP reference,
33 proposed/18 accepted drafts per request, and variable-length A/B/A
reset. This is not a robust speed comparison or production admission.
The reference and MTP had different native overlap settings. No long
contexts or HTTP/cancellation qualification in 0048.

## Current candidate (2K expanded bulk and private SSE qualified in 0051)

- Exact Q2_0 pack/runtime identity admitted by the pinned MTP validator;
  target conversion checkpoint provenance remains unresolved.
- Mixed CUDA0/HIP0/HIP1 15/18/15 or HIP-only 24/24, grouped HIP overlap,
  shared transport, full target and draft experts, max_t=8, 1 GiB reserve
  per owner, exact aligned 2048..8192-cell worker capacity (above 2K is an unqualified 0052 candidate), Q8 target and native Q8
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
- No long-context, lookup, compact/fast pipeline,
  guarded direct, automatic restart, stochastic sampling or MTP
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
No guarantee of network-level latency or production cancellation yet.

## Native bulk prompt seed (2K qualified; larger capacity candidate)

Explicit `--native-prefill-chunk 1..64` with MTP now admits prompts under the same exact admitted combined prompt/completion limit. Above 2K, native bulk must be explicitly enabled. One token bypasses bulk. The target may emit up to 64 residual rows; the cursor subdivides them into synchronous native graph batches of at most max_t (8), with actual successors across every boundary. The terminal HIP prefill's fenced `on_chunk` consumes completed R[p] with the actual prompt token at p+1, synchronous draft prime before R reuse. No host residual path to the drafter, no global embedding publication. The final prompt token remains in the ordinary verifier; a successful all-domain bulk ACK is an empty streaming checkpoint, allowing cancellation before that final token. The worker validates the exact request-local prefix and position; adoption requires a fully drained target and draft cursor at N-1. Capabilities `mtp_context_cap=8192`, `mtp_bulk_prompt_cap=8191` and `mtp_bulk_chunk_cap=64` gate Python activation above 2K. The 0051 coordinator remains compatible for 2K only. The terminal callback checks exact chunk shape/position and device before priming. The native Q8 admission estimate now separately counts resident context-dependent allocations (shared-RoPE QSA state, identity rows and dense attention scratch) in addition to the original conservative cushion. The terminal prefill allocation preserves the existing reserve plus 512 MiB prime headroom while its scratch is live. These checks are allocation-boundary checks, not a measured peak. All owners additionally use allocation-boundary reserve and full-residency checks in service mode. The existing 8192-token metadata wire ceiling is preserved; no `--long-context`, workload, or 256K MTP is admitted.

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

0052: opt-in aligned 2K..8K capacity candidate, exact worker capacities and full residency, broader metadata wire bounds without a format change. Above 2K requires explicit native bulk (1..64). Integrated validation pending. A short 8K-configured smoke checks allocation/admission, not correctness at long positions. Do not enable 256K MTP or claim long-position qualification from a short prompt.
