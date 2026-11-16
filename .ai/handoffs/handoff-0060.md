# Handoff 0060 — revised vanilla-first source audit and performance qualification

**Supersedes the previous 0060 handoff at d723fd29.** Do not run its retired
acceptance-counter, reset-overlap or prefill-piece-upload experiment. Sync
corrected branch head handed over by Chat AI; if old work has already begun,
finish/cooperatively drain active work and preserve logs, then sync safely
without discarding unrelated changes. Report actual tested revision. Read
.ai/VANILLA_FIRST.md, .ai/VANILLA_DIVERGENCE_AUDIT.md,
.ai/PERFORMANCE_ROADMAP.md and .ai/TEST_BUDGET.md.

## CPU/native source first

Use exact pinned native vanilla source on Oberon under the authorized existing
project/reference workspace boundaries (never inspect Strata-dev). Confirm
revision 8db7ddc46cd2575514408c4ab7dd878f69df2f5f and actual effective
operational settings. Repository main is only ancestor comparison. Finish
full changed-file diff/classification against pinned source; use provided
manifest and audit, not a new speculative design. In particular actual
generate.cpp native startup/defaults, STRATA_SPEC_COST chain-rule callback,
pipeline window size vs ordinary MTP confidence submission, native catch-up,
overlap/rollback, session prefix cache/restore lifecycle, native attention
and prefill kernel dispatch. Verify exact native selector and source reuse.
Audit duplicated loaders/owner cleanup/global-QSA ordinal workaround, shared
diagnostic gating and safety hardening vs native; preserve safety and record
which differences are indispensable boundary logic. If actual source missing,
defer implementation and report blocker. No new algorithm or blind flag
sweep. Use matching toolchain, checkout .venv, pip check, 28 jobs. Rebuild
host/native CUDA/HIP workers (core prefill changed). Run focused native pipeline
window, final pipeline/compact/early/finite/protocol guards, request/cursor/bulk
seed/stream, prefill resource and Python critical-path/launcher/service tests.
Exact CPU native selector 1816 assertions and unchanged policy/bulk/accounting
86271/1040444/42 passed here. Integrated tests and native builds not yet run.
Check stale enabled switches, unsupported STRATA_SPEC_COST, new native-window
scope; avoid allocation if invalid. Local current source and builds only.
Fix concrete defects using actual native implementation, maintain test budget,
keep baseline profile unchanged; do not introduce an unverified heuristic.

## Bounded GPU batch only after source/CPU pass and GPU idle

Use qualified Q2_0/index/runtime, INT8, existing 256K configuration with short
128–513 token coding prompt and 24–64 outputs, actual exact prompt IDs/length,
spec4/min_p0.5, max_t8, chunk32, full-residency and reserves, HSA_ENABLE_SDMA=0.
Keep existing draft-batch opt-in off for both arms (native minimum now 64),
fast-fence settings constant, no pipeline_mode=fast. Compare baseline vs
optimized native-derived switches in mixed 15/18/15 arm:

- STRATA_HETERO_COMPACT_RESULT=1
- STRATA_HETERO_FANOUT_CONTROL=1
- STRATA_HETERO_MTP_EARLY_METADATA=1
- STRATA_HETERO_MTP_NATIVE_COMPLETION=1
- STRATA_HETERO_MTP_NATIVE_WINDOW=1 **only after confirming the shared native
  pipeline selector is appropriate for actual vanilla width semantics**.

If scope permits separate confidence-selector effect; if not, disclose
combined result. No invented adaptive arm; STRATA_SPEC_COST not wired, do not
enable it. Retired flags unset. Native selector uses same .5 threshold already
used to stop draft generation, not a new tuned heuristic. Inspect generated/
confidence-admitted/submitted counts in scoped trace. No high-position or
64-row batch GPU qualification needed without evidence. For short HIP-only
regression unset EARLY_METADATA; use current qualified 24/24. Warm and use
small repeated untraced samples if useful; separate short trace with native
profiler/critical path and actual dispatch. Record TTFT/emitted useful rate,
accepted/feedback counts, output lengths, overlap and native paths, not
assumed GPU occupancy. Verify actual finite guard, all-owner commits, exact
offer/anchor/stop/length, cancellation drain/reset, residency/reserve,
HETERO_CLOSED and process-group retirement. No forced kill. Keep bounded
GPU wall aim <=5 minutes; stop launching more when budget reached; no
near-capacity, baseline matrices or 160 tok/s primary qualification.
No Actions or production default changes.

## Return

.ai/reports/report-0060.md: exact source/reference revisions, remaining
vanilla divergences (source findings and actual dispatch), CPU and hardware
scope, raw measurements/flags/results, any repairs/blockers, safe teardown.
Prioritize source-grounded native extraction for next batch rather than
custom algorithms. Commit/push report and corrections; stop and return to
Chat AI. No forced termination or unrelated changes.
