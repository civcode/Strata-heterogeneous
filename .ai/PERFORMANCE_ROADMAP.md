# Strata heterogeneous — parallelism-first performance roadmap

Status: proposed development priorities following report 0059 and user direction. This document is a roadmap, not a hardware qualification, implementation result, or a request to run a long benchmark. Keep the existing test budget in [.ai/TEST_BUDGET.md](TEST_BUDGET.md).

## Objective and architectural rule

Make heterogeneous Strata outperform the best relevant vanilla Strata configuration on the same Oberon machine, for equivalent workloads and resource constraints. An extra GPU used only as another serial layer stage does not guarantee a speedup: single-stream autoregressive dependencies and the sum of stage times still dominate. Exploit native Strata parallelism within stages, between stages and chunks, and across independent work; minimize the exposed critical path. Use serial modes only as a correctness/fallback reference. Do not conflate faster prompt processing, faster single-stream decode, and greater concurrent-request throughput.

Vanilla-first synchronization is mandatory. Audit the pinned vanilla checkout and its current native APIs before changing them. Reuse native verifier graph execution, stream/event dependencies, mapped flags and bounded waits, native prelaunch, commit, KV/prompt reuse, prefill pipeline and backend-dispatched kernels. Extend a native mechanism only when the cross-runtime/process boundary requires it; do not build an independent general scheduler or parallel synchronization stack. Never assume a CUDA event/stream handle is usable by ROCm or a device pointer can be shared across incompatible runtimes. Retain necessary publication/acquire semantics, validation, deadlines, state ownership, complete commit acknowledgements, cooperative cancellation and retirement. Remove an extra fence only if its completion guarantee is already provided and demonstrated. Keep full resident experts, INT8 KV and qualified memory reserves. No new GPU kernel math unless a documented native/backend gap warrants it. Historical numerical drift stays low priority absent a concrete new defect.

## Existing owner-defined performance targets (unchanged)

The authoritative acceptance criteria are [M5 performance](../experimental/heterogeneous/docs/M5_PERFORMANCE.md), [M6 pipelined performance](../experimental/heterogeneous/docs/M6_PIPELINED_PERFORMANCE.md), and the [M6 agentic coding qualification protocol](../experimental/heterogeneous/docs/M6_AGENTIC_CODING_BENCHMARK.md). This roadmap implements those targets, not a replacement target.

- Primary gate: median >=160 accepted output tokens/s on representative real-world single-agent coding workloads, same authorized Q2_0 model, precision, quality, comparable context and existing safety limits. No credit for rejected proposals, replayed tokens, prefill, pooled multi-request throughput, peak measurements or favorable repetitive-copy lookup. Preserve the existing equal-task median and coverage/quality protocol; bounded diagnostics cannot establish primary qualification.
- Intermediate comparison: ~120 decode tok/s, operator-reported vanilla dual-R9700. Not independently reproduced or matched yet.
- Productive per-device GPU utilization approaching 95–100% where achievable, sustained high useful power draw within existing thermal/power limits. Never burn power or do redundant computation to inflate a metric.
- Prefill comparison: operator-reported vanilla reaching ~4,000 tok/s; this is an attained observation, not an established sustained benchmark or a separate agreed numerical target. See [vanilla baseline](../experimental/heterogeneous/docs/VANILLA_STRATA_BASELINE.md).
- Primary target remains unverified; 160 tok/s implies 6.25 ms per accepted output token on average. Aim at reducing cost per accepted token through native parallelism, not simply shortening a control span.

## Evidence and baseline

Reports [0045](reports/report-0045.md), [0046](reports/report-0046.md), [0047](reports/report-0047.md), [0058](reports/report-0058.md), [0059](reports/report-0059.md) are the starting evidence, not reasons to repeat those tests.

* Ordinary Q8, 32K capacity, 4K prompt, 64 outputs: vanilla client TTFT/elapsed 3327.1/4263.7 ms; grouped HIP 2341.1/3739.8 ms; mixed 1900.7/3323.7 ms. Different native adapter settings; no exact engine-speedup claim.
* Seven-token equivalent client workload: vanilla elapsed 996.1 ms; grouped HIP 1507.4 ms; mixed 1549.2 ms. Decode must improve. Native ordinary feedback rate about 44–46 tok/s at 32K.
* Ordinary 256K workload: HIP 1780.12 vs mixed 2381.81 prefill tok/s. Existing capacity tests are complete; 256K MTP configured short smoke does not prove high-position MTP.
* 0046 ordinary fast mode (compact results, fanout, early metadata, fence polling) warmed elapsed 827.27 -> 607.92 ms, one paired short observation, factors not isolated.
* 0047 guarded direct fixed-token prelaunch verified bounded packets and ordering, not the production MTP service, bulk prefill or long-context admission. No general speedup established.
* 0058 513-token mixed MTP: mean narrow callback 19.63 ms vs batch 5.11 ms; submit-to-bulk-checkpoint about 867 ms for both. Leave batch mode opt-in pending sufficient benefit; further draft-prefill micro-optimization is low priority.
* Mixed 15/18/15 CUDA/HIP/HIP at ordinary 256K has about 1437/14224/16774 MiB allocator-boundary free in report 0045; CUDA reserve is tight. These are not transient peaks.

## P0 — Native mechanism inventory and profiling foundation (next implementation)

Before optimizing: inspect the exact vanilla reference on Oberon (the pinned ref is not accessible from the Chat AI connector), effective options, backend dispatch decisions, existing traces and calibration tools. Inventory applicable native parallel paths, default/opt-in status, vendor/architecture requirements, evidence of actual activation, owners and memory costs. Especially distinguish independent-slot batch mechanisms from multiple speculative rows of one sequence. Do not enable all flags blindly.

Measure matching warmed short/medium workloads, not long tests. Prefer existing built-in profiling first:
* STRATA_VERIFY_PROFILE, STRATA_VERIFY_TRACE, STRATA_COMMIT_TIMING;
* STRATA_PREFILL_TIMING, STRATA_MTP_PROFILE, STRATA_DRAFT_TIMING;
* STRATA_HETERO_PERF_TRACE, existing reset/commit/stage spans and MTP bulk summary.
Add a deliberate profiling launch path preserving only approved diagnostic environment variables; regular launch remains unchanged. Some variables are presence-based, so unset them for control. Record exact source, model, executable hashes, plan, capacity, active paths, prompt IDs/length, output count, settings, reserve, warmup and raw samples. Compare traces with an untraced run, never claim traced time as uninstrumented speed.

Reuse the existing tools/heterogeneous_critical_path.py, tools/heterogeneous_utilization.py, tools/heterogeneous_telemetry.py and M6 implementation roadmap/inventory ([M6 implementation roadmap](../experimental/heterogeneous/docs/M6_IMPLEMENTATION_ROADMAP.md), [GPU bottleneck investigation](../experimental/heterogeneous/docs/M6_GPU_BOTTLENECKS.md)); extend only missing coverage rather than recreate the harness. Build on their common monotonic timeline keyed by request/window/chunk/stage; collect native GPU phase timings plus host boundaries for reset, allocation, prefill, draft/proposal, verification, transfer, commit and stream publication. Report dependencies, critical path, overlapping work, per-device active vs idle time and IPC gaps; do not sum overlapping spans as elapsed. Prefer existing timestamps and tools over inserting additional synchronization. Use Nsight Systems (CUDA) and ROCm profiler (rocprofv3 if available/compatible) for short scoped timeline only when native data cannot explain gaps. Check local installation and clock alignment first; collect separately if combined tracing is not supported. Use perf/strace only for unresolved host bottlenecks; instrumentation is not a performance result.

Deliverable: reproducible profiling harness, native feature inventory, critical-path breakdown and prioritized defects with evidence. No inference behavior or default changes in P0.

## P1 — Native asynchronous verifier and CUDA–HIP handoff

Highest priority for single-stream decode. Map current worker transport and extra adapter event fences against vanilla Verifier::run, wait_commit, native chain_overlap/prelaunch and bounded handoff waits. In current shared core, Verifier::run synchronizes its streams and native HIP chain prelaunch already overlaps adjacent stages; heterogeneous CUDA/HIP adapters add event record/query loops. Identify real duplicate waits and unnecessary host/device/host copies. Use native stream/commit semantics to eliminate them where safe. Keep required device-boundary synchronization.

Reuse the native overlap graph and the existing narrowly admitted guarded direct path (0047) instead of inventing another readiness protocol. Audit request IDs, producer publication, consumer prelaunch, bounded wait/release, slot lifetime, full finite/checksum gates and failure cleanup. Stage compatible downstream metadata/admission early when useful, but measure and treat this as preparatory overlap, not full GPU handoff. Move to bounded direct/asynchronous handoff only after ordinary and MTP service state/transport compatibilities are established. Reuse existing native fanout/fast control features where applicable; compare factors independently. Goal: reduce exposed GPU idle/host round-trip time and improve actual decode tokens/s; no premature default enablement.

## P2 — Parallel attention and target verification

Equally important; do not focus exclusively on transport. Profile per-layer QSA (selection/indexer, scores, top-k, attention, KV), GDN recurrence, projections, shared and routed experts, output head and commit. Confirm the exact kernel dispatch on CUDA sm_89 and HIP gfx1201. Identify serial fallbacks and reuse optimized vanilla paths for the supported shapes and INT8 cache.

Native candidates to inspect:
* STRATA_QSA_SPLIT_ATTN (split tensor-core path where eligible; code may decline, HIP availability is not established);
* native multi-row qsa_decode_attn_batch, STRATA_DEC_BATCH, batched projections/experts and native independent stream branches;
* STRATA_GDN_CHUNKED on gfx12 where applicable in prefill, native batched GDN commit (STRATA_COMMIT_BATCH), QSA projection/norm/gate fusions;
* STRATA_BATCH_ATTN_ROWS, STRATA_QSA_SLOTS and STRATA_BATCH_GDN_GROUPS are independent-slot batch paths, not automatically applicable to single-sequence MTP; evaluate later with native batch admission.
* Inspect existing GPU nan guard and ordinary compact final-result path: MTP greedy verification uses target IDs and finite status; current full-logit transport adds substantial copy/serialization work. Enable compact MTP only if native guard, request/result identity, argmax and all-stage commit semantics are retained and stale/incompatible workers are refused. Keep full logits for diagnostic modes.
* Investigate speculative multi-row verification efficiency (1–5 rows now, graph max_t=8), not increasing speculative width blindly. Profile at different bounded positions when needed, using existing short tests and CPU contracts. Do not introduce cross-GPU tensor-parallel attention as a default. If native supported intra-layer/peer paths are insufficient and attention is actually dominating, separately design native-dispatch-compatible context/head/sequence parallelism with measured transfer/reduction cost.

Goal: lower GPU time per verified/committed output token, actual kernel utilization and reduced output-transfer cost; valid target-authoritative acceptance. Every backend path must confirm actual selection, not just an environment flag.

## P3 — Parallel prefill and inter-stage pipeline

Reuse vanilla STRATA_PREFILL_PIPE / Prefill::pipeline_chunk, STRATA_HANDOFF_PIPE (piecewise handoff using events and pinned buffers), persistent HandWorker, native expert/projection concurrency, batch kernels and QSA/GDN prompt paths where applicable. Prefill already bridges native chunks across backend processes; verify actual overlaps across CUDA, HIP0 and HIP1 rather than assuming the path is serial or fully parallel. Examine whether native planning and handoff mechanism are used by run_external and whether fixed chunk32 MTP, synchronous copies or terminal callback prevent overlap.

Aim for chunk c on CUDA, c-1 on HIP0 and c-2 on HIP1 simultaneously, with bounded double buffering and per-chunk/piece identities; individual upload/download work overlaps when safe. Use native pipe calibration/planning principles; adapt for differing stage speeds if evidence requires. Respect actual admitted chunk caps and memory. Remove cap64 MTP native chunk restriction only through a deliberate contract/scratch change qualified separately, not by forcing unsupported flags. Target concurrent scheduling across GPUs, not infinitely smaller chunks; avoid pipeline fill/drain overhead for short prompts. Keep native draft seeding integrated with terminal producer stream and its residual lifetime. Existing full-resident prefill/batch data are evidence; use short/4K only where necessary under test budget.

Goal: faster TTFT and prompt tok/s, reduced pipeline bubbles and no staging overwrite or state corruption.

## P4 — Prompt-cache reuse and optimized resets

Audit vanilla prefix-cache/session reuse, common-prefix matching, GDN/KV/indexer state restore, native prefill reuse and lifecycle contracts. Current resident adapter treats requests as independent, and new Prefill/stream/scratch may be initialized per bulk request. Measure target/draft reset, graph recreation (if any), scratch initialization, and upload separately. Prefer reuse of native prefiller/graphs/buffers within measured reserve, and reset only needed state when native semantics permit. No blind retention of large scratch across decode.

Introduce opt-in exact-token committed prefix reuse, respecting model/config/identity, full state owner participation, MTP's residual+successor binding, branch/truncate, no reuse of provisional or cancelled work, invalidation on errors and reset. Do not cache partial domain success. Cache-hit/miss and reused vs new rows must be explicit; distinguish full fresh request and reuse latency. Cover exact extension, differing suffix, shorter prefix, cancelled request, alternating histories and model/config change; later adopt native multi-slot/cache policies where supported.

Goal: substantially lower repeated-chat TTFT and amortized prefill/reset work without increasing memory beyond reserve or changing committed semantics.

## P5 — Adaptive MTP and native speculative overlap

Inspect vanilla MTP controller, draft confidence/acceptance/budget and native piped/deferred-commit mechanisms before selecting a policy. Native STRATA_PIPED_MTP refers to batch-slot deferred commits and may not be usable directly for a single resident request. Profile offered/accepted/verified/committed rows, graph size, per-round draft/target/commit cost, acceptance distribution over prompts. Optimize effective emitted tokens per second, not acceptance rate alone.

Start with request-local bounded adaptive proposal length 0..4 (anchor-only path when low predicted payoff), only after worker and coordinator contracts can agree on when an offer is optional and draft state pairing remains valid. Current service requires spec=4 and checks offer admission; a width change needs coordinated policy/metadata/test updates. No forced speculative skip violating cursor state. Evaluate confident vs difficult/rejecting prompts, length/stop/tail and reset; compare against fixed MTP and ordinary no-MTP. Overlap independent draft/target/commit work only where native ownership and dependencies allow; autoregressive next-token dependency remains. Consider native independent-slot piped MTP later. Keep target authoritative and default unchanged until evidence.

Goal: improve actual useful tokens per verification cost, avoid expensive low-acceptance rounds and preserve correctness/cancellation.

## P6 — Balance placement and native parallel throughput

Once measured stage profiles exist, calibrate layer distribution against target phase costs and memory (CUDA is the limiting owner; AMD1 also hosts head/draft). Do not increase CUDA burden just because prefill arithmetic is fast. Evaluate small admissible placements only when likely to improve critical path. Keep existing configurations as references. Compare single-request throughput separately from batched/concurrent aggregate throughput. For the latter, investigate vanilla slot batching, native independent-slot QSA/GDN, STRATA_PIPED_CONCURRENT and piped MTP, not custom host-level concurrency layered on a single state owner. This is a later direction, not an instruction to lift current single-request contract.

## P7 — Integrated optimization and acceptance

Follow the established M5/M6 targets and the M6 agentic coding qualification protocol above. Interim engineering milestones: eliminate the regression on short fresh requests relative to optimized comparable vanilla, exceed the matched intermediate decode comparison, retain/improve mixed prefill, and reduce repeated-prefix TTFT. The primary gate remains median >=160 accepted output tokens/s for representative real-world single-agent coding, with the existing quality, coverage and useful utilization objectives. No new numerical target is substituted. Extended primary qualification remains subject to explicit authorization under the current test budget. Evaluate native optimized HIP-only vs mixed on identical work, with MTP enabled on both when comparable; include best vanilla reference on same available hardware and note vendor/hardware/placement differences. Normal untraced repeated bounded samples, warmups, medians/ranges and interleaved A/B when justified. No general speedup claim from single pair or synthetic repetitive prompt. Report TTFT, std/median or range, accepted useful output rate, prefill, aggregate throughput separately, stage idle, exposed transfers, cache-hit latency, per-owner memory and errors. Verify target output/commit semantics, finite checks, cancellation and cooperative teardown. Keep prior capacity proof and test budget; any newly changed capacity/high-position behavior that needs long tests remains unqualified until explicitly authorized.

## Execution order and working discipline

P0 first. Then P1–P3 driven by the measured critical path, potentially interleaved conceptually but implemented in isolated cohesive changes with serial Chat AI/Coding Agent ownership. P4 starts once native reset/cache inventory is clear; P5 after verifier/draft cost breakdown and stable native state. P6/P7 only when earlier results justify them. This roadmap supersedes further blind batch-seeding tuning and serial-only optimization as the performance direction; it does not override current safety/capacity gates. Specific handoffs are created only when real local work is needed. CPU tests first, existing matching toolchain/venv and 28 jobs; short bounded GPU qualification only while idle, ordinarily within five minutes per batch, cooperative drain, no forced GPU termination and no GitHub Actions. Each change needs actual measured benefit, attributable mechanism, tested invariants, clear scope/limits and an opt-out until justified. Avoid repeated benchmark matrices or unproductive parity investigations.

## 0060 — Combined implementation candidate (source-side)

- P0: profile launcher opt-in; reset/native-prefill/proposal spans and critical-path attribution of early metadata/host overlap; added analyzer fixture.
- P1: opt-in native verifier completion and early mixed-domain metadata, plus existing independent commit fanout. Metadata overlap is not cross-runtime GPU graph prelaunch; actual direct GPU handoff remains gated.
- P2: existing compact-result path admitted for greedy resident MTP with the native terminal finite guard. No new attention math; measure native QSA/GDN and split/pipeline dispatch before selecting additional kernels.
- P3: ordered native HIP stream uploads of validated bridge pieces; no new scheduler. Existing native local successor pipeline, chunk limits and buffer ownership retained.
- P4: parallel independent HIP session_zero enqueue with all-owner drain; prefix cache reuse is not implemented.
- P5: request-local adaptive native draft width (0–4), feedback only after terminal commit, subsequent global ACK before the next offer. Zero width uses existing native tail prime, preserving the anchor and exact-offer contract.
- P6/P7: defer placement/throughput selection and representative acceptance to actual profiles; the 160 accepted tok/s objective is unchanged.

All new execution switches are experimental opt-ins, with existing defaults unchanged. Standalone adaptive source/test exact-blob compilation passed 5,396 assertions; existing policy, bulk pairing and accounting suites passed 86,271 / 1,040,444 / 42 assertions. Exact launcher shell passed syntax and four stub-executable cases. Full integrated host/native build and new analyzer/protocol fixtures pending on Oberon (direct checkout DNS unavailable). No GPU speedup, high-position MTP qualification, or task performance claim.
