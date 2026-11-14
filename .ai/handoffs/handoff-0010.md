# Handoff 0010

## Objective
Qualify the implemented critical-path instrumentation and transport optimizations on Oberon, and identify the dominant remaining prefill/decode GPU starvation causes. Establish measured before/after data for the next implementation batch.

## Context
Read `experimental/heterogeneous/docs/M6_IMPLEMENTATION_ROADMAP.md` (B1–B5). B1/B2 code already implemented in commits f0379de76e0a386fe6506763781a1a17fd1462f0 and 3400b7c3728af9083b7698c3ae0f8b704bea071b; 14 local Python CPU tests and isolated wire encode check passed. Integrated build and all new GPU results are pending. Previous report 0009 only reanalyzed historical data. Reference revision for before/after is f36a0f9a5cb42b5c1bb1e0afcb46e345a47eea3e. Existing replay corpus and artifacts are on Oberon as recorded in report 0008; reuse the frozen inputs. Historical heterogeneous decode ~59 tok/s, busy ~13/34/46%; prefill ~159 prompt tok/s including reset, busy ~13/27/50%. Operator vanilla dual R9700 ~50/100% decode, ~100/100% prefill is unmatched.

## Tasks
1. **B1/B2 integration:** sync exact handoff commit, build host and independent CUDA/HIP targets with existing options, run affected host/device/shared/final pipeline and Python tests (including new critical path and golden frame regression). Autonomously fix recoverable build/test defects. Confirm original full residency, byte-boundary/commit correctness, token IDs, cancellation and clean shutdown with ordinary supported tests.
2. **Hardware A/B:** compare exact before/after revision (separate safe build/staging artifacts, no destructive reset) with matched original 12/18/18 full Q2_0, optimized flags and lookup off. Use ordinary prose and frozen nonrepetitive replay, socket and shared memory, chunk 1 and safe auto where relevant. Five warm repeat samples where practical; trace off for rate measurements. Report phase throughput, TTFT proxy, equal-task median feedback, output equality, per-card prefill/decode busy, watts, temps, clocks when available and aligned sample coverage. Use existing passive sampler and updated utilization analyzer; historical records can be reanalyzed, but cannot substitute for fresh test.
3. **B3 critical path:** collect separate native traces with STRATA_HETERO_PERF_TRACE=1 and successful resident benchmark JSON; invoke `.venv/bin/python -m tools.heterogeneous_critical_path --trace TRACE.log --stages 3 --benchmark BENCH.json > CRITICAL.json`. Verify matched coverage, worker span counts, stage/transfer/copy/receive/barrier/checkpoint gaps and phase differences. Use installed supported nsys/rocprofv3 for bounded actual GPU/API timeline captures, separately from performance runs. Determine layer/head/memcpy/launch/wait contribution where available. No synthetic timings or additive nested spans.
4. **Comparison:** reproduce existing vanilla dual-R9700 reference only if an authorized runnable executable/config/assets are accessible in scope; match workload/weights/residency/sampling/timing and explicitly establish draft/MMQ/graph mode (draft off first). If inaccessible, proceed with heterogeneous tests and state precise blocker. Identify native bulk-prefill/pipeline reuse opportunities and rank B4/B5 by evidence. Correct clearly demonstrated small defects, rerun affected checks and preserve existing safeguards.

## Constraints
Only agreed fork/branch and /home/chris/workspace/Strata-heterogeneous, /home/chris/tmp/strata-heterogeneous. Ordinary scoped GPU tests implicitly authorized. No Strata-dev access, external assets, other services, upstream writes, Actions, force push, GPU reset/driver changes, forced kills, native fault injection, precision/clock/power/context/arena/reserve changes. Use vi/vim, existing isolated checkout Python >=3.10 and matching CMake interpreter, <=28 build jobs. Preserve original placement and all safety/commit/finite checks; no unsafe multi-window overlap, threshold relaxation or fake utilization. Real-world agentic primary gate remains separate from bounded replay.

## Completion Criteria
- Integrated affected builds/tests pass or genuine blocker is documented after troubleshooting.
- Fresh before/after native runs, output/commit/lifecycle checks, phase throughput and aligned GPU telemetry available.
- Joined critical-path report with coverage and bounded GPU timeline evidence (or precise unavailable profiler blocker); vanilla comparison or documented access blocker.
- Ranked B4/B5 bottlenecks with actual evidence. No inferred speedup from source changes or CPU tests.

## Reporting
Write `.ai/reports/report-0010.md` with exact tested revisions, meaningful PASS/FAIL/UNVERIFIED, matched before/after rates and GPU utilization/power by phase, timing attribution, any fixes, limitations and raw staging path. Commit/push relevant corrections and report only to the fork branch. Return iteration/status/branch/final SHA/short summary/report path; stop editing.
