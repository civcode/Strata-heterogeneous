# Handoff 0088

## Tier
2

Measure and compare. Scope the agent may change: `mixed_split/tools/agent_session.py` and its table script (say each
change in the report), and `.ai/` notes it is asked to write. No engine changes.

## Objective
Find out whether report 0087's probe differences (live against fresh, all three) are this branch's or vanilla's
behaviour, before the probe check is used as a release criterion.

## Context
- Read `.ai/PROJECT.md` (new: **Starting and stopping GPU processes**; the tools now reset an inherited "ignore
  SIGINT" and stop with SIGINT then SIGTERM, never SIGKILL), `.ai/STATUS.md` and report 0087.
- Report 0087's probes differ early in the reasoning text: `pruned` at the 2nd token, `deep` at about the 20th,
  `compacted` at about the 40th ("was" against "is"). Near-ties are plausible: the live server reached those contexts
  through cached prefixes and chunk boundaries a fresh server does not have. Not established.
- The requests asked `reasoning_budget_tokens: 0`, yet the replies are reasoning text. Find out why (the server's
  field name or its meaning in 0.2.15) and fix the driver's request if it is the driver's mistake.
- Your worktree has uncommitted `agent_session.py` changes (a cycle limit, an AMD sampling fallback; report 0087).
  Commit them first, each described in the report, so this handoff runs committed code.

## Tasks
1. Safety checks (`.ai/PROJECT.md`), `free -m` before and after. First rule: `HSA_ENABLE_SDMA=0`.
2. **Fresh against fresh:** replay `.ai/runs/0087-scaled/probes/probe-pruned.json` on a freshly started mixed server,
   twice (restart between), with `STRATA_IQ_MT_MIN=1`. Equal to each other?
3. **HIP-only control:** the same scaled session (seed 85, `--scaled`) against the production mirror with
   `STRATA_IQ_MT_MIN=1`, then each probe replayed on a fresh HIP-only server (restart between). Equal or different,
   and where the first difference is (token or character).
4. If task 3 differs too: one more control with the HIP-only mirror and `--prompt-cache 0 --conversation-cache-slots 0`
   (the session reads every prompt from token 0): live against fresh for one probe.
5. Write a short table: config, probe, equal, first difference.

## Hardware Test Budget
20 minutes of GPU run time. Every run under 10 minutes; no context past 64K.

## Constraints
- **First rule: `HSA_ENABLE_SDMA=0` in every run.** All rules of `.ai/PROJECT.md`, including the new start/stop rule.
  Evidence under `.ai/runs/0088-*`.

## Completion Criteria
- Tasks 2 and 3 done (4 if task 3 differs); the table in the report.
- `.ai/reports/report-0088.md` and all changes are committed and pushed to `feature/mixed-native`.

## Reporting
- The table; the reasoning-budget finding; the committed driver changes; safety lines, swap, GPU time.
