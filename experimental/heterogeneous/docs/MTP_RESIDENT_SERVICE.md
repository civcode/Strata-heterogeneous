# Request-local resident MTP

Report 0048 qualified the bounded mixed Q2_0/Q8 resident buffered candidate
on Oberon: identical 32-token outputs to its short non-MTP reference,
33 proposed/18 accepted drafts per request, and variable-length A/B/A
reset. This is not a robust speed comparison or production admission.
The reference and MTP had different native overlap settings. No long
contexts or HTTP/cancellation qualification in 0048.

## Current candidate (new HIP-only and streaming paths awaiting validation)

- Exact Q2_0 pack/runtime identity admitted by the pinned MTP validator;
  target conversion checkpoint provenance remains unresolved.
- Mixed CUDA0/HIP0/HIP1 15/18/15 or HIP-only 24/24, grouped HIP overlap,
  shared transport, full target and draft experts, max_t=8, 1 GiB reserve
  per owner, exact 2048-cell worker capacity, Q8 target and native Q8
  draft cache. Native drafter runs on terminal HIP1.
- The drafter now borrows its own device-local/mapped native embedding.
  It does not install HIP1's pointer as the process-global target
  embedding; this is essential for layer-zero target execution on HIP0.
  Legacy non-service diagnostic behavior is unchanged. The drafter and
  its graphs are destroyed before its embedding is released.
- Variable prompts and completion budgets 1..256 within 2048 total.
  Fixed spec=4, confidence threshold 0.5, serial prompt chunks 1..8.
  Buffered or incremental streaming is explicitly selected.
- No native bulk prefill, long-context, lookup, compact/fast pipeline,
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

0048: mixed buffered hardware scope passed; original focused CTest 7/7
and Python 55/55 passed. New HIP-only and streaming work has not been
compiled in the integrated checkout or exercised on GPUs/HTTP. Chat AI
compiled exact new C++ stream cursor/header and test with g++ C++17
-Wall -Wextra -Wpedantic -Werror (18,248 assertions passed), ran the
three independent Python stream contract tests against exact checked-in
contract/long-context module blobs (PASS), and checked Python syntax for
those two modules. Full adapter/CLI, native integration build and real
GPU validation are outstanding (handoff 0049). The local full checkout
remains blocked by GitHub DNS/access. No changes to default profiles.

Next: terminal native bulk residual seed with actual prompt successors,
then scalable draft KV/memory admission, with short tests only. Do not
enable 256K MTP until those features are implemented and admitted.
