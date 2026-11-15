# Request-local resident MTP — initial candidate

Report 0047 qualified the bounded guarded CUDA→HIP diagnostic. This MTP
candidate is independent: the normal HTTP launch profiles, 256K capacity,
fast pipeline and guarded direct path remain unchanged.

## Supported candidate (not hardware-qualified)

- Exact Q2_0 pack/runtime identity already admitted by the historical MTP
  validator; target conversion checkpoint provenance is still unresolved.
- Mixed CUDA0/HIP0/HIP1 placement 15/18/15, grouped HIP overlap, shared
  transport, full target and draft expert residency, max_t=8, 1 GiB
  reserve per owner, exact 2048-cell worker capacity, Q8 target and native
  Q8 draft cache. The native drafter loads on HIP1; its actual state and
  memory are checked, not assumed.
- Resident variable-length prompts and request-specific completion budgets
  1..256 within 2048 combined tokens. Fixed spec=4 and native chain
  confidence threshold 0.5, matching the user's vanilla reference.
- Serial verifier prompt chunks (1..8), buffered output. No native bulk,
  incremental streaming, HIP-only, long-context, lookup, compact/fast
  pipeline, guarded direct, automatic restart, stochastic sampling, or MTP
  diagnostic captures in this candidate. Unlike the old fixed diagnostic,
  this path needs no logit capture or terminal diagnostic proposal.
- No changes to native math. The target decides each accepted token; the
  same all-domain prefix commit barrier applies. Request metadata (actual
  prompt IDs and completion budget) is sent to the final worker after
  reset and ACKed before any window. A distinct final HELLO rejects old
  workers. Both coordinator and observer check actual prompt slices and
  live offers; near the physical KV limit, decline native speculative
  writes and feed an ordinary verified anchor. Reset clears target and
  native draft state before the next request. Failed/uncertain protocol
  remains terminal. No physical erasure of rejected suffixes claimed.

## Test configuration

Generate a fresh server JSON using the established 2K stage configs and
matching rebuilt binaries. Existing 256K launcher and its referenced
configs must not be edited. In the new JSON's heterogeneous object the
explicit additional settings are:

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
  "stream_tokens": false,
  "workload_mode": false,
  "long_context": false,
  "native_prefill_chunk": 0
}
```

This is a fragment, not a complete server JSON. Include the actual
coordinator/workers, plan, fingerprint and ordered configs for the 2K
mixed setup. Omit pipeline_mode. The CLI accepts --mtp PATH --spec 4
--spec-min-p 0.5 with --resident (or --write-server-config for CPU-only
config generation). At native level the opt-in is --mtp-resident-service
--mtp-probe-rt ABS_PATH --hip-split-overlap. Do not start while another
server/benchmark owns GPUs. Buffered HTTP may use an SSE client but only
releases output after the native request's validated completion.

## Validation and follow-up

Standalone request policy was compiled with g++ C++17 -Wall -Wextra
-Wpedantic -Werror; 85,372 assertions passed, exact Git blob hashes match.
Integrated protocol, worker, CLI and HTTP changes/tests are authored but
not compiled/executed in Chat AI's environment (GitHub DNS restriction).
Do not enable this candidate in the production profile. Next integration:
actual short GPU qualification, batched committed-token streaming,
HIP-only ownership (review process-global NativeEmbed), native terminal
bulk residual seed with real successors, and draft KV memory planning for
large contexts. Preserve prior diagnostics; no repeated 256K stress runs.
