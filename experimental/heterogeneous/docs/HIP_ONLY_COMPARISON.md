# M0035 — HIP-only versus mixed runtimes and vanilla

Keep the physical model fully partitioned; do not add a second inference or sampling implementation. The experimental ordinary HIP-only plan is

    hip:0:0-24,hip:1:24-48

Two independent HIP workers use the existing per-stage coordinator, entry embeddings, handoff and terminal head. Grouped HIP now admits the full two-stage domain; the native Verifier chain already handles input=nullptr at layer zero and owns a native local mapped handoff. Only worker executables named in the plan are needed by CLI/server. No CUDA process is launched for either HIP-only plan. The grouped full domain consumes a native prefill token spec once, processes the complete prefix on its two local Prefill objects using the same forced chunk-major implementation and existing lifetime/drain path, adopts the native prefix and ACKs the coordinator. Unlike the mixed path, there are no intermediate R frames on the coordinator socket; the ordinary final-token verifier and ACK gate are unchanged. Existing mixed grouping, cross-runtime R relay and numerical diagnostics remain separate.

HIP-only requires full 512 experts/layer, two stage configs, ordinary inference. MTP, diagnostic captures, explicit split overlap, receiver-arm and mapped CUDA egress are not admitted for this experiment. Native bulk requires grouping; separate HIP workers have serial prompt processing only. At the 0035 baseline the default was FP16. After 0038 the ordinary inference default is native INT8 (Q8); use `--kv f16` for historical reproduction. Qualification is pending. The 32K opt-in context cap and 4096 completion cap remain unchanged; no 256K runtime admission yet. Same device allocation/free-memory guards apply. Full local native link follows the existing same-runtime native pipeline; this is not a new physical boundary-byte qualification.

## Candidate matrix

| Arm | Plan | Grouped | Prefill |
| --- | --- | --- | --- |
| A | original 12/18/18 mixed | yes | serial |
| B | original 12/18/18 mixed | yes | native bulk chunk 128 |
| C | HIP 24/24 | no | serial |
| D | HIP 24/24 | yes | serial |
| E | HIP 24/24 | yes | native bulk chunk 128 |
| F | authorized vanilla dual HIP | vanilla | serial if supported |
| G | authorized vanilla dual HIP | vanilla | native bulk, matched chunk if supported |

Optional 15/18/15 mixed (separate handoff, no concurrent modifications) after/if compatible; do not conflate a changed layer split with runtime overhead. MTP and prefix reuse disabled initially on every arm; match model assets/quant/tokenizer/chat template, context, prompt IDs, sampling, max completion, resident warmup, kernel build/config flags, and effective settings. Actual vanilla operator 4000 prefill /120 decode figures are not matched reference; see VANILLA_STRATA_BASELINE.md. No speculative comparison until reliable no-draft baseline; later reproduce vanilla MTP and existing bounded mixed MTP separately if compatible, not a claim of general HIP-only MTP.

For HIP only create two new configs from the prior approved HIP configs with backend unchanged, stage indices 0,1, ranges [0,24),[24,48), max_cache_slots 12288 each. Copy the actual authorized PLE shard field used by the original entry CUDA stage to the HIP entry config; terminal no PLE shard field. Keep same model/index/profile/GGUF, max_t 8, reserves, full slots and unchanged pack, new paths. Use existing source-bound readiness/model gate (actual PCI mapping, not ordinal assumption) and no-run CLI. Device memory must fit actual allocations with safe reserves; 24/24 is only a proposed split. Begin 2048 then 8192 or 32768 only after fit.

Generate server config using existing strata.py heterogeneous-generate flags and actual built coordinator/HIP worker paths, --plan hip:0:0-24,hip:1:24-48, --config NEW_HIP0 --config NEW_HIP1 --fingerprint actual --write-server-config new.json --shared --workload-mode --max-context 2048 --max-new-tokens 512 --prompt-chunk 1 --stream-tokens. Omit --cuda-worker. For D add --group-native-hip; for E also add --native-prefill-chunk 128. Before running inspect actual --help and use the report-0032 validated loopback server startup command. Do not use placeholder paths verbatim. Separate workers must never silently use bulk. Existing mixed configs unchanged.

A CPU-only helper now derives both config files from the three ordered, previously approved mixed v2 configs, retaining each HIP owner's reserve, common source identity and full expert residency. Use a fresh private output directory and set the same `max_cells` capacity on comparison arms; the helper refuses capacity greater than the source configs, so use the previously qualified config set at the desired context. It records source and output SHA-256 digests. It does not establish device identity, source integrity, runtime admission, or memory fit. For example (substitute actual approved absolute paths):

    .venv/bin/python tools/heterogeneous_hip_pair_config.py --source "$CUDA_CFG" "$HIP0_CFG" "$HIP1_CFG" --output-dir "$NEW_DIR" --split 24 --context 2048

Use the generated `hip0.cfg`, `hip1.cfg`, and plan/fingerprint in `manifest.json`; complete native source/readiness preflight before any GPU launch. Keep the manifest private with the run evidence. For a new context/split, use another fresh directory. If a source has a nonzero stage arena cap, stop and review rather than silently discard it.

## Benchmark procedure

Use same frozen workload manifest for every arm. The short smoke suite demonstrates API function but is not a speed benchmark; run cancellation in a separate lifecycle pass with drain readiness, not between timing requests. Use representative repository review/document prompts and nonrepetitive long answers with at least 128 completion tokens where natural; task outputs must be evaluated or clearly UNSCORED. Same manifest bytes, case IDs, exact rendered prompt token IDs, mode, output cap and streaming. Use one server at a time, repeat balanced reversed arm order in independent launches. Keep startup separate. Record cold/warm distinction, at least one disclosed warmup and five measured repeats per case, quality and failures, actual complete token counts. Use --repeats 6 with the existing workload runner and no secret output in repository. Native and vanilla can expose different timing denominators; keep client TTFT and end-to-end latency as the primary comparable metrics. Preserve raw native progress/owner/kernel timings and actual server timing source. No rate from SSE fragments, no short decode claimed as sustained.

Example offline comparison, after real same-manifest JSONL files have been collected:

    .venv/bin/python tools/heterogeneous_architecture_compare.py --manifest "$MANIFEST" --reference grouped_hip --arm grouped_hip="$GROUPED" --arm separate_hip="$SEPARATE" --arm mixed="$MIXED" --arm vanilla="$VANILLA" --warmup 1 --output "$OUT"

The analyzer checks complete attempt inventory, same manifest hash, model label and repeat count; matches first HTTP stage and observed prompt token count. It reports median/min/max and count, paired ratios with >1 better; elapsed and decode ratios require equal actual completion counts and finish reason. Server-rate ratios require identical explicit timing source (and uncached prefill), minimum 128 completion tokens by default for decode. It excludes tool-dependent follow-up timing and retains overall quality/failures. Raw inputs, model/build/runtime config equality and token IDs must be independently verified. Same label/count does not prove them. Missing vanilla/unsupported arms must be explicitly NOT_RUN; do not create dummy run files. Compare multiple independent launch blocks separately.

## Attribution

C vs D (same HIP hardware/split/kernel and serial) is strongest evidence for process boundary/control overhead, but also differs in local handoff and chain scheduling. D vs E measures local bulk scheduling advantage; B vs E includes CUDA hardware and placement changes, not pure interoperability cost. E vs G is closest to vanilla, after matching chunk, precision, kernel options, KV format, prompt IDs, cache, MTP and timing denominator. Difference may include coordinator, per-token checkpoint/ACK, graph, native kernel and scheduling behavior. Inspect per-window coordinator timing spans in separate trace runs; do not sum overlapping spans or infer GPU kernel cost from host timings. Use matched off/1 Hz/4 Hz monitoring controls, phase-aligned passive GPU utilization/memory/power and separate vendor profiles if available. Trace and profiler runs cannot serve as clean throughput runs. Record actual MTP acceptance in separate arms only when admitted, and no claimed improvement until measured. Transfer any narrowly identified reusable optimization to mixed on a separate reviewed commit with repeatability, correctness, resource and task gates unchanged.

### Native-prefix trace coverage

With `STRATA_HETERO_PERF_TRACE=1`, the resident coordinator now emits `HETERO_NATIVE_PREFILL_TIMING` after successful native-prefix ACK/drain, alongside the existing commit marker. The analyzer joins them by request and row count, reports the inclusive host round-trip separately, and verifies that its timestamps fit the matching benchmark prefill phase before ordinary window 2. Old traces remain readable but untimed prefixes cannot pass complete matching coverage. This is not a breakdown of GPU compute, transfer or ACK time and does not change inference scheduling. Rebuild and validate on Oberon before using the new timing for attribution.
