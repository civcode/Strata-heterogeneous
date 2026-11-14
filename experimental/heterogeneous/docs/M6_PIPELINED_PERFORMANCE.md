# M6 — Productive GPU utilization and pipelined performance

Goal on Oberon: **Median >=160 accepted single-agent decode tok/s on representative real-world agentic coding workloads**, same authorized Q2_0 target weights, quality and comparable workload/context. See [agentic coding qualification protocol](M6_AGENTIC_CODING_BENCHMARK.md) for the primary gate and separate bounded replay/controls; repetitive lookup is not a success criterion. 120 tok/s operator-reported vanilla dual-R9700 is an intermediate comparison, not an independently reproduced baseline. Also target sustained per-device GPU utilization approaching **95–100% where productive and achievable**, maximize useful simultaneous activity and eliminate idle gaps. GPU busy counters alone cannot prove useful compute, full SM/CU occupancy, or output correctness; never increase utilization by redundant work.

Evidence: report 0005, tested 2d8babd4, M5 partial. Ordinary combined shared warm median 60.373 feedback tok/s; 157.851 only on highly repetitive matched copy task (111/111 drafts accepted, 16 multirow windows). Not general 160. M5 peak samples 87/79/66% are not sustained. Final stage and middle HIP stage dominate remaining serial latency. Trace baseline and combined are separate from throughput runs. See M5_PERFORMANCE.md and report 0005 for exact metrics.

## Active implementation roadmap

See [M6 implementation roadmap](M6_IMPLEMENTATION_ROADMAP.md) for the implemented B1/B2 changes and the remaining B3–B5 hardware-dependent sequence. Handoff 0010 qualifies the actual implementation; do not treat unverified changes as a measured gain.

## Immediate priority after report 0008

See [GPU starvation and vanilla dual-R9700 bottleneck investigation](M6_GPU_BOTTLENECKS.md). Establish matched prefill and decode baselines, trace critical paths and native GPU timelines, and prioritize evidence-backed bottleneck fixes before further speculative or placement tuning. The operator's approximate vanilla 50/100% decode and 100/100% prefill utilization are comparison points, not matched measured results. Report 0008 replay 58.987 tok/s lookup-off and 58.172 lookup-on; primary agentic qualification remains unverified.

## M6 sequence

| Step | Objective | Current status |
| --- | --- | --- |
| M6.0 | Close M5 startup/readiness and lookup gaps; repeat ordinary and lookup matched benchmarks; add mandatory phase-resolved utilization. | Readiness and metrics source implemented; native qualification pending. |
| M6.1 | Measure stage latency, PCIe handoff and balance actual layer placement. | Controlled plan sweep in handoff 0006, not a validated default plan. |
| M6.2 | Native Strata MTP in heterogeneous pipeline. | Verify approved runtime assets and host-native lifecycle/bind dependencies first. |
| M6.3 | Async activation handoffs with reusable host slots and backend-owned events. | Design after profiling; no cross-runtime device pointers/events. |
| M6.4 | Two-window speculative decode overlap. | Future implementation; reuse upstream scheduling and native verifier mechanisms where feasible. |
| M6.5 | Speculative state rollback/restoration and commit/cancel validation. | Must be in place before publishing M6.4 output. |
| M6.6 | Pipeline prompt chunks across stages. | Separate multi-window state/transport qualification, not M4 chunk-size tuning. |
| M6.7 | Representative real-world agentic coding median >=160 tok/s with matched quality, utilization, lifecycle and power qualification. | Target not reached; bounded replay baseline needed now. |

Native MTP (multi-token proposals) and cross-stage two-window lookahead (overlap between verification windows) are distinct. CPU suffix lookup is a useful existing control but does not generalize to nonrepetitive text. Reuse existing upstream MtpDrafter, NativeHead/final_R, verifier snapshot/restore and pipeline-windows code where feasible. Final-stage drafter first may simplify head/residual ownership; first-stage placement for lookahead is an architectural option only. Distinguish ordinary accepted-draft ratios from successful lookahead windows/attempts. Current one pending window/shared activation slot is not multi-window safe; do not bypass the commit gate or start adjacent provisional work without implementing independently owned slots/state and verified restoration. No cross-vendor tensor parallelism in this batch.

## Bounded startup fix

Report 0005 had a first-run failure in which earlier CUDA timed out waiting for slower HIP preparation, with identical retry successful. This is a plausible readiness-barrier mismatch, not a proven diagnosis of all cold-start problems. Native coordinator exports its existing bounded --timeout-ms value (default 120000) as STRATA_HETERO_STARTUP_WAIT_MS. Waiting for the first READY byte after HELLO uses this separate bound; after a byte or EOF is readable, the configured worker frame timeout still applies. A shared coordinator HELLO deadline bounds all peers rather than per-peer cumulative loading waits. Native public worker config's 30000-ms I/O/fence deadlines, descriptor validation, all-worker identity barrier, fail-stop semantics and cleanup stay unchanged. CPU mock/default worker API without explicit startup wait retains old behavior. No automatic restart or increased worker I/O timeout. CPU tests simulate late peer and bounded incomplete READY/EOF; native cold-start robustness requires benign real launches.

## Phase-aligned metrics

Existing passive sampler records monotonic timestamp, read timing, cycle index, requested interval, power and available PCIe sensors. Resident JSON records monotonic startup, request start, last prompt commit, last decode checkpoint, completion and shutdown boundaries. On the same Linux host and boot, run one sampler per benchmark, retaining raw JSONL. Example (replace monitor indices with independently verified mapping; they are not necessarily native ordinals):

    .venv/bin/python -m tools.heterogeneous_telemetry --device cuda:0 --device hip:0 --device hip:1 --interval 0.25 --duration 180 > "$STAGING/telemetry.jsonl"

Run the matched resident benchmark concurrently with the passive sampler (separate from trace-enabled diagnostics); capture benchmark JSON stdout. Choose duration covering startup, requests and cleanup; allow sampler to end normally or interrupt only the sampler. Then:

    .venv/bin/python tools/heterogeneous_utilization.py --benchmark "$STAGING/benchmark.json" --telemetry "$STAGING/telemetry.jsonl" --device cuda:0 --device hip:0 --device hip:1 > "$STAGING/utilization.json"

Report per GPU and per mode: decode-only mean, median, sampled peak, valid/observed sample counts, percentages of samples >=90%, >=95% and <10%, VRAM used/total, power if present. For same-cycle complete samples with acquisition span <= half requested interval, report percentage where all GPUs >=10%, >=90%, >=95%. These are sampled overlap proxies, not exact simultaneous execution or fractions of wall time. NVML/amdgpu busy averaging windows differ; report methodology and missing readings as unavailable, not zero. A low-sample warning (fewer than 10 complete cycles or <80% of expected approximate cycles) must not be interpreted as valid sustained evidence. Increase warm repetitions within original max_new/context bounds rather than disable stop logic. Segment startup, prefill including reset, decode feedback, final completion tail, resident idle and shutdown separately. First request excluded from warm pool; pooled decode feedback tok/s uses summed accepted rows / summed decode seconds; retain per-run median/range. Record raw device-mapped baseline and post-close VRAM. If sufficient time/power coverage, estimate GPU-only J/accepted feedback token with limitations; do not substitute for wall-power measurement. Measure sampler overhead with matched sampler-on/off throughput.

HETERO_PERF adds start_ns/end_ns (C++ steady_clock, expected to use same CLOCK_MONOTONIC epoch as Python monotonic on target Linux; confirm on host). Correlate separate traced per-stage phase timing with inactivity, native diagnostics and transfers; do not infer kernel duration from overlapping inclusive spans or treat a 250-ms sensor read as a per-token trace.

## Placement and matched controls

Start with order CUDA0 (PCI 04:00.0) -> HIP0 (03:00.0) -> HIP1 (07:00.0), original full slots 12/18/18. Given M5 measured stage costs (CUDA fastest, final slowest), compare bounded candidates 12/20/16 and 14/19/15 only if exact stage/pack/profile/PLE gates, passive memory arithmetic, actual free memory and original arena/reserve checks permit. Experimental layer ranges on same GPUs, not changes to GPU ordinals or budgets; do not silently select a new default. For each candidate create fresh config copies changing only stage layer bounds and matching full slot counts, with correct layer-1 PLE ownership. Review native estimator omissions and retain conservative headroom. Skip/report unfitting configurations. Compare same weights/artifacts, context, token caps, sampler, flags and PCI mapping. Check full residency, zero fallback and per-stage math. Record quality differences (teacher-forced logits/margins/KL with existing diagnostics when feasible; free text divergence alone is not a proven bug). Profile CUDA OCuLink and HIP negotiated PCIe link read-only. Broader stage orders and contexts deferred.

## Local verification and boundaries

New utilization analyzer and tests: exact Git blobs 1c0aec1b1adda60957251230c38a22e20f1f78fe and 7880b2f087152ad098723b540f624de81094ddb9; five Python fixture tests PASS locally, no GPU. Exact performance.hpp fc517e7e896c9a161e23a01b2eb89d0d3d13c697 compiled strict GCC C++17; valid/invalid startup values and monotonic trace checks PASS. Direct checkout still fails GitHub DNS; modified worker/coordinator integrated build and new CPU socket tests not run here. M6.0/1 native qualification pending; M6.2–7 not implemented by this commit. No GitHub Actions or upstream writes.

## Report 0006 and hardening gate

See [M6.1H — shared-core hardening](M6_HARDENING.md) for the measured 0006 results and partial private patch integration. Finish hardening and verify native MTP assets before unsafe new pipeline state paths. Mandatory sampled utilization and power (per-device and complete aligned total) accompany throughput; maximize useful work within existing thermal/power limits, not tokens/watt. 12/18/18 remains reference. No M6.2–7 implementation or general 160 claim.
