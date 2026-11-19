# Handoff 0090

## Tier
2

Measure with the finished driver. Scope the agent may change: run configs (build paths only), small fixes to
`mixed_split/tools/agent_session.py` and `agent_table.py` (say each in the report), and `.ai/` notes it is asked to
write. A failure in the engine, the front stage or the server is reported, with Tier 1 escalation recommended, not
fixed here.

## Objective
The release-candidate qualification's long sessions (`.ai/RC.md`), resumed after the 19 November crash: one full
256K session on the mixed mirror first, then the same session on the production mirror, each with its probes judged
by the new criterion.

## Context
- **A rerun of handoff 0089**, which stopped after 3.5 minutes (report 0089): its watcher stopped on
  `workqueue: svm_range_restore_work [amdgpu] hogged CPU`, a warning, because the safety pattern `amdgpu` matched it.
  Chat AI's mistake in the handoff. `.ai/PROJECT.md` now has **stop patterns** (faults, corruption, hangs) and
  **record patterns** (warnings, recorded with their times, not stopping). RC criterion 3 now allows swap growth under
  1 GiB per session (swap stays on).
- Start fresh: new run directories `.ai/runs/0090-*`; report 0089's partial session is not continued.
- Read `.ai/PROJECT.md` (new: long runs allowed again; **the 19 November crash case** and what every long run records;
  the first rule; starting and stopping GPU processes), `.ai/RC.md` (criterion 2 changed: equal or a near-tie, and no
  non-finite logits), `.ai/STATUS.md`, reports 0087 and 0088.
- Human User's decisions (2026-11-19): swap stays on; the mixed session first; `HSA_USERPTR_FOR_PAGED_MEM` not set.
- The builds `build-0087-*` are current for the engine (only `mixed_split/tools/` and `.ai/` changed since). The
  mirrors name them.
- `agent_session.py` since report 0088: `replay` asks for top-5 logprobs and judges equal / near-tie / DIFFERENT /
  unjudged (selftest 18/18). `--reasoning-budget 0` turns thinking off. `--max-cycle-min 90` stops after a long cycle.

## Tasks
1. Safety checks (`.ai/PROJECT.md`); `free -m` and `swapon --show` at the start, between the sessions and at the end.
   `python3 mixed_split/tools/agent_session.py selftest`.
2. **A log watcher for the whole of each session** (report 0089's, with the new patterns), in the run directory:
   every kernel line matching the record patterns, and every server-log line with `paused the GPU` or `non-finite
   logits`, each with its time. Stop only on a `non-finite logits` line or a kernel line matching the **stop
   patterns**: the driver after its current request (its SIGUSR1 path), then the server with `stop_cooperatively`; run
   no further GPU job, and report. Warnings are recorded and reported, and the session goes on.
3. **Mixed full session**: a server through `serve_mixed.py` with `STRATA_IQ_MT_MIN=1`, `agent_session.py run OUT
   --url ...` (the RC defaults, seed 85, sampler on), stopped cooperatively at the end. Then replay each of its probes
   on a freshly started mixed server with the same setting, restarting between probes.
4. **Production full session**: the same with the HIP-only mirror, same seed and setting, and its probe replays.
5. `agent_table.py` over the two sessions (`--markdown` into the run directory), and the RC criteria 1 to 4, each
   passed, failed or not measured, with the numbers. Add a row per session: swap used (max), the number of
   `paused the GPU` lines and their total ms, the number of userptr/SVM workqueue lines.

## Hardware Test Budget
**About 4 hours of GPU run time, over the 30-minute allowance** (announced to the Human User; allowed unless refused).
One session may run up to 2 hours; probe replays up to 30 minutes per configuration. If time runs out, stop
cooperatively after the current request and report what was measured; the mixed session and its probes come first.

## Constraints
- **First rule: `HSA_ENABLE_SDMA=0` in every run.** All rules of `.ai/PROJECT.md`, including the start/stop rule.
- Evidence under `.ai/runs/0090-*` (commit the tables, JSON and watcher logs, not large server logs).
- No engine, server or mirror-argument changes. No two GPU jobs at once, except the two processes of one mixed run.

## Completion Criteria
- Both sessions ran to their end (or a stop rule was taken and reported), probes replayed and judged.
- `.ai/reports/report-0090.md` and all changes are committed and pushed to `feature/mixed-native`.

## Reporting
- The two sessions' tables side by side; the probe verdicts (with the gap in nats for each near-tie); the RC
  criteria 1 to 4.
- The crash-case signs per session (swap, GPU pauses, workqueue lines), with times.
- Safety lines, or "none". GPU run time used. What remains open for the soak.
