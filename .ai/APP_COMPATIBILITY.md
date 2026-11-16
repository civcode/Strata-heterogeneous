# Vanilla Strata application compatibility — mandatory gate

Owner requirement: native public launch config, exact native launch Python,
native default terminal narration, and native HTTP/web integration. Source
identity of individual files is not enough; compare effective behavior.

## Reference and scope

Pinned Oberon reference: `8db7ddc46cd2575514408c4ab7dd878f69df2f5f`.
Use the commit object in /home/chris/workspace/Strata, not its dirty working
tree; never inspect Strata-dev. Repository main
`146df3c468739147f520ac3816de806f71232571` was examined as an ancestor only.
Report 0060 identifies 680 net changed paths relative to the pin and remains
incomplete. Compare every shared change and classify as native upstream,
necessary boundary, shared safety hardening, experiment to retire, or
avoidable divergence. Do not revert qualified correctness gates blindly.

## First source changes in this increment

- The setup-written shell command and setup start now share the same native
  command builder with the Oberon binder. Public invocation is
  `python serve/server.py --engine strata --config FILE --port 8081 --open`.
  The binder retains a lock, recorded-source/hash checks, local paths and the
  explicit offline safety argument. Legacy --engine heterogeneous remains a
  test/qualification entrypoint. The nested heterogeneous object still
  selects the resident adapter; this is not complete native integration.
- Standard port/parallel fields are present in both checked-in profiles.
  log, cwd, env, lib_dirs, open_browser and parallel=1 are understood by the
  adapter path. The normal shared child_env implementation applies explicit
  environment/library-path/offline rules; it skips homogeneous vendor
  selection and kernel defaults for the mixed coordinator to preserve
  qualified per-backend ownership. Actual native defaults must be extracted
  per worker. Preflight and live coordinator receive the same child env/cwd.
- By default the wrapper appends diagnostics to the standard log field
  (~/tmp/strata-heterogeneous/<profile-stem>.log); raw preflight, worker,
  timing, readiness and retirement diagnostics remain available. Existing
  explicit STRATA_HETERO_PROFILE=1 terminal capture remains when no log is
  specified. A direct ordinary --engine strata launch with mixed metadata
  and no log key uses the config's sibling .log (unless explicit profiling).
  --no-open is available on the binder for headless runs. Do not use logs
  with tracing enabled as a speed measurement.
  Protocol stdout stays separate and validated. The shared native model
  loading and startup narrator are reused; no fabricated expert loading
  or placement facts. The shared Service continues progress and completion
  output, and the original browser startup runs when --open is passed.
- Second interrupt must not forcibly kill the mixed coordinator and strand
  its native worker group. Keep existing cooperative retirement gate.
- No MTP algorithm, model quantization, kernel selection, resource admission,
  cache capacity, sampling policy, or topology change in this increment.
  Full pinned checkout/runtime parity is NOT established.

## Direct implementation increment (0062)

The ordinary args array now owns context, KV format, fully resident KV,
explicit prefill chunk, MTP path, speculative window and threshold. The
binder fills exe, tokenizer and attested pack/GGUF/PLE/profile arguments,
preserving args_written and native user_args merge precedence. Private
placement and qualified resource metadata are still required. The public
mixed profile now uses native --spec 4 (three drafts plus anchor), with an
explicit coordinator and HIP-worker capability gate. Only explicit legacy
qualification retains four candidates plus anchor. The native confidence
window selector is used for the public mode; GPU evidence is pending.

The vanilla generate.cpp request-summary formatting was extracted into
include/strata/program/request_log.hpp and used by both binaries. Mixed
summary numbers derive from committed work, with zero reuse/checkpoints;
elapsed coordinator timing includes transport/control overhead. The native
request log follower, Service progress fields, pack tokenizer loader and
end-token policy are shared. Canonical mixed startup verifies the pack
tokenizer against GGUF. More ordinary HTTP configuration keys are passed
to the existing Service, not reimplemented.

Local CPU checks: exact Git-blob-matched C++ headers and request-policy
suite passed 86,274 assertions; shared formatter matched normal and
cancelled fixtures byte-for-byte. Four Python config tests passed against
the exact mixed_runconfig module with extracted native user_args helpers.
No complete checkout/build or GPU/HTTP qualification was possible in this
environment. See reports/report-0062-chat.md. These changes narrow the gap,
not a completed application compatibility gate.

## Native public config contract still blocking

The known-good reference JSON contains these native top-level fields:
exe, args, cwd, tokenizer, model_name, log, lib_dirs, port, backend, env,
gpu, gpus_asked, layer_split, vision, args_written, parallel. Retain
native spelling and meaning, including args_written as provenance rather
than a second effective command. Vision config uses exe, mmproj, model, gpu,
max_tokens, cpu_exe. Reuse native setup.py, strata_control/user_args.py,
strata_cli/launch and serve/server.py to resolve precedence and values.
Backend-local identity and mixed partition metadata may need a single
minimal extension; do not make the user configure ordinary parameters twice.
Worker key=value configs must be generated implementation details rather
than a replacement public schema. Until shared native parser/dispatch is
wired, the existing profile rejects unsupported standard args explicitly.

All 24 supplied native engine options (case/spelling/arity verified against
ancestor generate.cpp parser; confirm pinned parser and execution):
  
| Option | Supplied example | Remaining native integration |
| --- | --- | --- |
| --pack | directory | Bind from native parsed option |
| --native | GGUF shard | Native model/shard resolution |
| --ple-gguf | GGUF shard | Native PLE owner |
| --expert-profile | file | Native profile lifecycle |
| --expert-cache | auto | Native actual allocation and admission |
| --mmap-experts | flag | Native provider selection |
| --prefill | auto | Native auto selection and pipeline |
| --spec | 4 | Public total window implemented; confirm GPU operation, complete native parser/caller |
| --spec-min-p | 0.5 | Native threshold, no fixed-value replacement |
| --mtp | directory | Native request/catch-up/rollback caller |
| --max-context | 262144 | Native context capacity/limits |
| --kv | int8 | Native format parser and dispatch |
| --kv-resident | 0 | Native full/streamed semantics |
| --vision | flag | Native image encoder and embedding path |
| --vram-reserve-mib | 700 | Native units and per-device memory admission |
| --control-vector-scaled | FILE:1.0 | Native control vector loader |
| --control-vector-layer-range | 4 44 | Native inclusive layer range |
| --cvec-mode | project | Native projection semantics |
| --cvec-dir | per-layer | Native direction choice |
| --pipeline-windows | 0 | Native pipeline semantics |
| --prompt-cache | 6 | Native checkpoint lifecycle |
| --prompt-cache-spread | flag | Native cache policy |
| --conversation-cache-slots | 4 | Native parked-conversation lifecycle |
| --conversation-cache-min-tokens | 8192 | Native admission policy |

Environment: HSA_ENABLE_SDMA=0, STRATA_KEEP_LEFTOVERS=1,
STRATA_NO_PRESET=1, STRATA_SPLIT_OVERLAP=1, STRATA_PIPELINE_AUTO=0,
STRATA_TRANSFER_TRACE=1. Confirm actual pinned owners, per-backend
dispatch, effective values and precedence. Trace flags are not production
defaults. Public native `--spec 4` and old service `spec=4` are NOT comparable; the latter remains only for legacy qualification.

## Remaining terminal and HTTP gates

Use the actual native startup events, engine log and shared narrator for
device/layer split, draft mode, engine priority, mapped experts/cache fill,
ready and progress. Extend the existing protocol to report real committed
prompt progress per native chunk, not fabricated progress and not repeated
per-worker user messages. Dynamic hardware counts, context, images-on,
rates and elapsed time are truthful, not literally equal across machines.
The user's log files are captured server terminal output, not raw binary
stdout; test native protocol stdout and default terminal output separately.
No HETERO_ trace flood in ordinary terminal; all internal records remain in
the log or an explicit trace mode. Failures express their real cause.

Compare native HTTP lifecycle and all native features, including OpenAI
chat/completions, Anthropic messages, responses, tools and structured
output, sampling and logprobs/logit bias, tokenization/templates/stop IDs,
vision, UI settings, monitor/telemetry, streaming cancellation/disconnect,
capacity, request queue, unload/reload/restart, ready files, parallel slots,
slot save/restore and cache. Shared web sources must remain identical to
the pinned reference unless a documented shared upstream fix is necessary.
Existing 256-token resident completion limit and greedy-only restrictions
are partial qualification, not native feature parity. Reuse the existing
StrataEngine and native generate/serve lifecycle where possible; extract
minimal engine-side backend interfaces, don't clone it or the scheduler.
Do not bypass the all-owner commit, finite, resource, or retire checks.

## Qualification

Run existing native CLI/setup/serve tests in the local venv plus the
new compatibility tests. Test the supplied standard args/config against
the real native parser, compare effective options and native defaults
(including malformed/unknown args and conflicts), derive worker plan
without changing native options. Compare API status, schema and streaming
events for identical requests with same model, quantization, seed and
settings; normalize only legitimate volatile values. Test true cache reuse
and native vision, sampling, control vector and cancellation. Compare
normal terminal output (format/order/meaning), raw protocol and logs
independently. No GPU batch until source/CPU gates are met. Follow test
budget; no Actions and no forced termination. Publish actual pinned source
diff, builds/tests, unsupported features, effective settings and any blockers.
