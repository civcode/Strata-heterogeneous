# Handoff 0060 — Combined native performance qualification

Read PERFORMANCE_ROADMAP.md and TEST_BUDGET.md first. Test the handed-off HEAD on feature/heterogeneous-hip-only. Source-side commits 7119b28a7c46 (compact/metadata), 9df8bc5f3845 (adaptive width), f8002c5b8029 (native completion/reset/upload), 83306575ac64 (profiling). Keep existing production/default profile unchanged. Local standalone controller tests: 5,396 assertions, exact header/test blob IDs 2d7d27e4366d2ade86183fddc5dfd5df47a52367 / 09e96270546f52ab5574a559bdd378b6c0451147; existing policy/bulk pair/accounting 86,271 / 1,040,444 / 42 passed. Launcher exact blob 7fffe3097b21f4984e577c591a3ea5f32b77ff88: sh -n and four stub-executable cases passed (ordinary sanitization, profile preserving selected traces/unsetting zero, invalid mode and invalid trace refusal). Full direct checkout unavailable here (DNS); no integrated native compilation, analyzer tests or GPU work done.

## Local verification first

- Inspect exact pinned vanilla/native sources locally, including successful Verifier::run synchronization, recursive wait_commit, Prefill::run_impl chunk hand_done/stream ordering before pinned slice reuse, and native terminal nonfinite guard. Check actual backend-specific dispatch and memory. If a guarantee is absent, repair or disable affected switch; do not add an alternative scheduler. Preserve native graph/event mechanisms, owner boundaries, all commit ACKs, finite gates, drain/reserves.
- Use current matching vanilla toolchain, venv, existing build directories, 28 jobs. Build host coordinator and required native CUDA/HIP workers. Run focused CPU final-pipeline (including compact MTP/early metadata/protocol failure tests), mtp_adaptive_width, request policy, cursor, bulk seed/pairing, prefill stage contracts, stream and Python critical-path tests. Exercise launcher profile sanitization with a harmless stub, invalid mode/flag, ordinary baseline. Include existing service fail-stop/cancel contracts as needed. No GitHub Actions.
- Confirm flags below apply only to admitted modes, stale/invalid inputs fail closed without GPU allocation when possible. In particular diagnostic full-logit cases and HIP-only mixed-only flags remain refused. Fix concrete build/test defects autonomously in this batch. Do not reinterpret prior numerical discrepancies as new bugs without evidence.

## One bounded hardware batch

GPU idle first: no competing GPU processes or active server; same known Q2_0/index/runtime, 256K configured capacity, chunk32, max_t8, INT8 KV, spec4/min_p and full resident experts, existing memory headroom from report0058. HSA_ENABLE_SDMA=0. No long-context ladder or near-capacity requests. Prefer one shared deterministic non-repetitive short coding prompt (128–513 tokens, record exact IDs) and bounded 24–64 output tokens, same admitted workload per arm. Warm each arm, collect a small number of untraced samples and record actual token counts if EOS. Use same known local harness, not alternate runtime.

Compare three arms:
A. Existing MTP baseline: all new switches off, existing full-logit/normal completion path.
B. Fixed-width optimized group, all following flags =1 except ADAPTIVE.
C. Same optimized group with adaptive-width enabled.

STRATA_HETERO_COMPACT_RESULT=1
STRATA_HETERO_FANOUT_CONTROL=1
STRATA_HETERO_MTP_EARLY_METADATA=1
STRATA_HETERO_MTP_NATIVE_COMPLETION=1
STRATA_HETERO_MTP_RESET_OVERLAP=1
STRATA_HETERO_MTP_PREFILL_UPLOAD=1
STRATA_HETERO_MTP_ADAPTIVE=1

Do not use ordinary pipeline-mode=fast or change existing fast-fence or batch-seed policy between arms. For mixed use the qualified 15/18/15 layout. If time permits, one short optimized HIP-only 24/24 regression omitting EARLY_METADATA and PREFILL_UPLOAD (explicitly unset); reuse qualified HIP config and MTP. Exercise normal completion plus cancellation/reset/recovery in same resident process; do not force a GPU kill. Observe actual adaptive cap changes when naturally reached, accepted prefix and native draft work; CPU zero-cap coverage is not proof of a GPU zero-cap run. Verify compact path, native backend state, successful commits, no draft output leakage, buffer ownership, full residency/no host routes, current reserve, cooperative HETERO_CLOSED and group retirement. If a switch causes a fault, safe cooperative drain only; isolate or fix only within test budget.

Separate one bounded trace sample with STRATA_HETERO_PERF_TRACE=1 (wrapper STRATA_HETERO_PROFILE=1). Native STRATA_VERIFY_PROFILE=1 and STRATA_PREFILL_TIMING=1 if useful and supported; collect actual backend-native selected algorithms, e.g. existing QSA/GDN batching and handoff pipeline, not blind flags. Use tools/heterogeneous_critical_path.py --stages 2 --endpoints 0,2 --trace <file> for mixed; grouped HIP uses --stages 1 --endpoints 1. Match --benchmark only if same-run compatible schema. Inspect reset, native bulk init/domain, proposal time, verification, commit, early admission overlap and uncovered gaps. These are host spans, not device occupancy. For per-device productive utilization use native profile/telemetry (disclose sampling coverage). Time traces separately; compare untraced TTFT, user-visible accepted decode tok/s, elapsed, offered/accepted and result lengths. Do not assert a speedup from insufficient samples, compare only equivalent workloads. No extended matrices, no 160 tok/s qualification in this smoke. Aim for five minutes GPU wall per batch; stop launching additional cases when budget exhausted and drain active case normally.

## Return

Write .ai/reports/report-0060.md: exact commit/build/tests, native source/dispatch audit, flags/commands, bounded raw observations and scope; include blockers, which switches actually qualified, memory/residency and lifecycle checks, evidence for next bottleneck. Report if direct GPU handoff, prompt-cache reuse or additional native attention/placement work needs more evidence; those are not completed by this batch. Commit and push report and any necessary fixes to same branch, then stop and return completion summary to Chat AI. No forced termination, no long tests, no default-profile enablement.
