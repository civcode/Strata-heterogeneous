# Handoff 0046 — bounded qualification of M5–M7 candidates

## Scope

Chat AI has implemented explicit baseline/fast pipeline profiles, early
downstream metadata dispatch, guarded direct GPU publication and a model-free
probe, a matched pipeline comparison axis, and a short HTTP smoke runner.
Read experimental/heterogeneous/docs/BOUNDED_PIPELINE.md and .ai/TEST_BUDGET.md.

## Local validation first

Use the existing vanilla Strata toolchain / venv and existing Oberon build
directories, 28 parallel jobs. Do not use GitHub Actions. Do not alter the
vanilla checkout, unrelated tools, model assets, drivers, clocks, or unrelated
services. Run the focused CPU pipeline/host protocol tests, handoff guard test,
direct graph probe Python tests, pipeline summary/CLI/server profile tests,
HTTP completion-budget regressions and quick smoke unit tests. Fix actual
build or test defects directly within scope. Native CUDA/HIP builds are
needed for new probe/profile code; reuse prior artifacts only for unchanged
targets, never mix stale components with new explicit capability tests.

## GPU checks: short and optional

First check whether the user's HTTP server or another GPU task is active.
Never stop or disrupt a running user server. If GPUs are not available for
exclusive tests, complete CPU/build work and report PARTIAL / blocked GPU.
Do not send requests to a user server unless explicitly permitted.

Total GPU test admission budget: aim for at most five minutes. No 256K
near-capacity run, context ladder, prolonged benchmark, or historical
numerical investigation. Preserve full residency and >=1 GiB reserve each
GPU. No forced GPU kills, reserve lowering or spills.

If GPUs are free and permission exists:
1. Guarded model-free CUDA->HIP0->HIP1 probe, 1 row, 4 rounds, 1 warmup,
   wait 1000ms; exercise nonfinite publication and withheld publication in
   separate fresh processes. Verify byte-exact normal payload and bit-exact
   zero quarantine on both consumers on rejection. Check both workers exit
   normally. If actual failure/cleanup uncertainty, stop GPU testing.
2. Only if time remains, short same-topology Q8 resident baseline versus
   fast (2K/32K capacity, tiny prompt, 32–64 output tokens, minimal warmup).
   Check committed counts, finite checks, full GPU residency and shutdown.
   Report output hashes and timing only as smoke observations, not robust
   throughput estimates. Do not perform the full A/B/B/A matrix unless it
   fits the short remaining budget. One authorized bounded HTTP smoke may
   replace additional benchmark work.
3. No real-model guarded asynchronous prelaunch in this handoff. The
   model-free checked primitive must pass on actual hardware first. The
   early metadata path is not GPU-side direct prelaunch.

Stop admitting tests when the budget runs out; allow in-flight work to
finish or cancel via the safe protocol. Never force-kill a GPU process.

## Return

Write .ai/reports/report-0046.md with PASS/PARTIAL/FAIL per scope, exact
commands, source/build identities, outputs, timing, unattempted cases and
remaining blockers. Preserve all failures, private logs with umask 077.
Do not label M5/M6/M7 complete or claim full throughput gains based on CPU
fixtures. Commit and push only relevant changes to feature/heterogeneous-hip-only,
report the final commit SHA and report path, then stop.
