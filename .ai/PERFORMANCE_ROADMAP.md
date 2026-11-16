# Strata heterogeneous — vanilla-first parallel performance roadmap

Active plan after source-side correction to handoff 0060. This is an execution
roadmap, not a hardware qualification. [VANILLA_FIRST.md](VANILLA_FIRST.md) is
a mandatory project-wide rule; [VANILLA_DIVERGENCE_AUDIT.md](VANILLA_DIVERGENCE_AUDIT.md)
records the evidence and blockers. Supersedes the earlier 0060 adaptive-width,
reset-overlap and prefill-piece-upload proposal. Historical measurements are
unchanged. Respect [TEST_BUDGET.md](TEST_BUDGET.md).

## Objective and architectural rule

Exceed optimized vanilla Strata using **proven native parallel mechanisms**:
native layer-split overlap, parallel attention/selection/GDN/experts within
each device, pipelined native prefill, native speculative windows and native
conversation-cache reuse. Serial operation is a correctness/fallback reference,
not the optimization target. Use one CUDA domain and one grouped HIP domain;
AMD-to-AMD uses native Strata chains. Only the CUDA–HIP edge and necessary
distributed ownership/admission/commit/failure handling are new. Preserve
current qualified placement and all reserves. Do not migrate to tensor
parallelism or invent a new inference/synchronization/prefill/cache/speculation
algorithm. When the actual native source is unavailable, defer the algorithm
change, not the architectural requirement. Extract and share the exact native
mechanism, do not create a second approximation. Keep current homogeneous
defaults; prove the applicable native path and its effective dispatch first.

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

## P0 — Native-source audit and profiling foundation

Source-side audit and profiler plumbing exist. Start from the pinned vanilla
Oberon reference and actual operational flags; compare optimized vanilla and
grouped HIP (and mixed only where it informs the necessary edge). Resolve
the precise source and effective dispatch for vanilla generation,
STRATA_SPEC_COST, native window sizing/catch-up, layer-split device planning,
native startup/attention options, prefill, reset/cache and rollback. Complete
the changed-file audit against the exact vanilla checkout; repo main is an
ancestor reference, not that exact binary. Follow the inventory in the audit.
No new algorithm or blind flag sweep.

Reuse STRATA_VERIFY_PROFILE, STRATA_VERIFY_TRACE, STRATA_COMMIT_TIMING,
STRATA_PREFILL_TIMING, STRATA_MTP_PROFILE, STRATA_DRAFT_TIMING, the existing
critical_path/utilization/telemetry scripts and native counters. The 0060
launcher profile mode and extra host spans are retained. Record actual GPU
kernel dispatch, critical path, per-owner timing and useful utilization,
proposals/accepted rows, cache state and memory reserve. Use Nsight/ROCm
profilers only for unresolved timeline evidence. Distinguish host spans from
GPU work, and traced samples from clean throughput. Scope: short bounded
workloads only, no new extended qualification matrix.

## P1 — Native overlap and CUDA–HIP boundary only

Native Verifier::run, chain_overlap, prelaunch, wait_commit and mapped ready
flags own same-runtime execution. Profile redundant adapter fences; do not
replace native waits with custom polling. 0060 native-completion switch calls
the existing native completion contract and omits only the duplicate adapter
events after completed run/commit. Retain DMA/publication/borrowed-buffer
lifetime/diagnostic drains. Compact-result/fanout/early-metadata extensions
reuse existing boundary mechanisms; they are not GPU prelaunch. Keep opt-in
until validated with native source and actual workers.

For cross-runtime GPU-ready publication, reuse existing vanilla prelaunch/
handoff graph and bounded ready-word semantics, using independently registered
host buffers only where safe. Guarded direct (0047) is a short diagnostic, not
service qualification. No new generalized scheduler, cross-runtime events,
custom spin loops, independent HIP stage processes, unvalidated direct mapping
or bypass of all-owner ACKs. If vanilla assumptions need extraction, minimize
the seam and prove it before service enablement. Measure exposed critical path,
not just CPU send time.

## P2 — Native parallel attention and target verification

First compare actual kernel-selection initialization (native_worker_defaults)
against pinned generate.cpp, including short attention, GDN/QSA fusions, native
matrix/projection and expert options. Consolidate exact shared native setup
rather than maintaining duplicate hardcoded dispatch policy. Profile native QSA
indexer/selection/attention, GDN recurrence, native projections/experts,
head and commit. Inspect native STRATA_QSA_SPLIT_ATTN, qsa_decode_attn_batch,
STRATA_DEC_BATCH, STRATA_GDN_CHUNKED, STRATA_COMMIT_BATCH and existing supported
fusions. Enable only demonstrated eligible native paths on each backend; use
native dispatch/shape checks. Independent-slot batch paths
STRATA_BATCH_ATTN_ROWS/STRATA_QSA_SLOTS/STRATA_BATCH_GDN_GROUPS are not
automatically single-stream MTP paths. No new attention math, cross-GPU
tensor-parallel attention or independent verifier. Extend native compact
head IDs/finite-guard result across the wire instead of downloading full
logits (0060 candidate). Preserve result identity/all-stage commits.

## P3 — Native prefill pipeline

Use actual vanilla Prefill::run_impl, pipeline_chunk/STRATA_PREFILL_PIPE,
STRATA_HANDOFF_PIPE and persistent HandWorker/native successor chain. Retain
R-only bounded transport and source lifetime across CUDA–HIP; local HIP
successor chain already pipelines. Do not create another chunk scheduler.
Measure actual overlapped CUDA/HIP0/HIP1 prefill and native piece events,
planning, full-resident scratch, allocations and terminal draft seed.
The standalone 0060 piece-upload experiment is withdrawn. Current small MTP
chunks fit a single wire piece, so this was not a proven multi-piece overlap.
Keep the native bridge staging fence pending a native-backed seam. Native
E-9 draft_kv batch minimum restored to 64; local embedding ownership remains.
The old 32-row opt-in measurements are historical, not evidence for altering
the native threshold. Keep 64-row local use opt-in with existing capacity
gates; use the native narrow fallback. Do not force STRATA_MTP_PREFILL_SYNC=0
on HIP given known stalls. Derive chunk sizes from native planning once
admission/scratch support permits. Do not increase resource budgets blindly.

## P4 — Native prompt-cache reuse, session lifecycle and reset

Use vanilla conversation prefix reuse/checkpoint/KV state restore, native
common-prefix matching and the native request lifecycle. Inspect the real
generate.cpp/session restore orchestration, conversation_snapshot/state,
prefill reuse, GDN/QSA/PLE and draft state. Heterogeneous currently resets
native state per request, creates per-request Prefill scratch, and has no
prompt-cache reuse. The unqualified independent reset-overlap experiment is
withdrawn. Keep vanilla session_zero and its current drain semantics; profile
reset stage/initialization. Avoid duplicating cache algorithms/snapshots, bypassing
zeros or assuming unchanged tokens alone guarantee a safe restore. Extract the
native cached-session operation and add only cross-domain identity/transaction,
drain and ownership validation. Cache all-owner committed state only. Use
native restore/truncate/rollback rules for extension, changed suffix, shorter
prefix, cancellation and invalidation. Do not silently reuse provisional work.
Consolidate duplicate stage load/cleanup with native owner APIs when verified;
the current global-QSA-ordinal release workaround must not simply disappear.

## P5 — Native adaptive MTP and speculative overlap

Use MtpDrafter::draft, its spec_min_p and the existing shared
pipeline_window_size/pipeline_accepted_drafts extracted from native scheduling.
Delete the independent acceptance-counter width controller; no replacement
heuristic. The 0060 native-window candidate uses the shared native selector
on generated probabilities before existing request stop/length clipping;
native draft itself still decides how many rows to generate. Existing spec4,
min_p0.5, exact offered IDs, positive accepted prefix and current limits stay.
Keep this selector opt-in until compare with actual pinned vanilla native loop.
STRATA_SPEC_COST is NOT wired merely by setting an environment flag: native
set_chain_rule requires the exact vanilla caller policy. Refuse nonzero cost
mode until that policy is extracted/reused, validated and qualified. No invented
acceptance counters, bounds or cost formula.

Investigate replacing redundant per-window native prefill/one-row draft
orchestration with vanilla multi-row draft catch-up using the actual verified
residuals/target next IDs. MtpPrefixCursor currently enforces necessary actual
successor binding/offer state; do not remove it without a native equivalent and
all-owner safety proof. Reuse native chain_launch, chain_outputs_ready,
pipeline-window launch/finish/commit, lookahead and snapshots/rollback, not a
parallel scheduler; native STRATA_PIPED_MTP may be for independent slots.
Preserve target authority, accepted-prefix commits, cancellation and resident
state. Measure useful output per cost and actual acceptance. Do not increase
width, change confidence or implement new speculation policy before source
equivalence and measurements.

## P6 — Native placement and productive parallel throughput

Inspect vanilla layer-split auto/calibration and native parallel options, keep
the original layer-range topology. Calibrate observed per-owner execution and
reserves (CUDA tight, terminal HIP head/draft); extend existing native planning
rather than invent a load balancer. Do not treat aggregate multi-request rate
as single-agent rate. Native slot batching/STRATA_PIPED_CONCURRENT and piped
MTP are later only if measuring and extending the proven native implementation
with safe slot ownership. No custom batching scheduler. Keep single-active
resident request contract until native replacement is proven.

## P7 — Integrated performance acceptance

Existing M5/M6 gates and agentic coding protocol above remain authoritative:
median >=160 accepted single-agent output tok/s, useful utilization approaching
95–100%, bounded power/thermal and memory reserves. Intermediate reported
vanilla comparison ~120 decode tok/s, ~4000 attained prefill reference, not
independently matched. Compare best comparable optimized vanilla to native
grouped HIP and mixed; same source/model/precision/request/settings/quality
and actual kernel dispatch where feasible. Interim milestones eliminate
short-request regression, improve sustained decode and prefill, demonstrate
native repeated-prefix TTFT. Report TTFT, emitted/committed feedback/accepted
tokens, throughput denominators, elapsed, cache reuse, memory and coverage
separately. No artificial idle-burning or credit for rejected/replayed rows.
No primary qualification claim from short synthetic tests. Extended acceptance
requires explicit owner authorization per test budget.

## Implementation sequence and evidence gates

P0 source equivalence and profiling first. P1–P3 driven by native mechanisms
and measured critical path, in cohesive changes. P4 native cache/lifecycle
extraction and P5 exact native MTP policy/scheduler are priority parallel
directions, not new algorithms. P6/P7 follow measured evidence. Each change
must identify upstream revision, function/option, exact behavior retained,
necessary boundary seam, ownership, test results and unverified scope.
Keep exclusive Chat AI / Coding Agent handoffs. CPU tests first; matching
toolchain and venv, 28 jobs, no Actions; bounded GPU testing only on idle
Oberon, ordinarily <=5 minutes, cooperative drain. No unproven algorithms
even behind opt-ins. Hardware qualification is separate from source reuse.

## 0060 revised implementation candidate (source-side)

Retained profiling and existing-native-derived compact result, fanout, early
metadata (mixed only) and native completion opt-ins. Replaced separate adaptive
width controller with the shared native pipeline-window confidence selector
(opt-in, existing greedy acceptance helper already shared). Restored vanilla
64-row E-9 batch minimum and existing reset/bridge lifetime behavior; removed
unqualified adaptive/reset/upload experiments. Stale enabled switches fail
closed. No production/default profile change (32-row batch opt-in deliberately
returns to native narrow path). The native cost-aware caller, full speculative
pipeline, native attention dispatch audit, cache/restore, stage owner refactor
and final performance qualification remain open. Local exact shared policy test
1816 assertions and unchanged MTP request/bulk/accounting suites
86271/1040444/42 pass. Full integrated host/native build, new protocol and
profiling integration, native pinned source comparison, and GPU testing pending.
See revised handoff 0060 and divergence audit.
