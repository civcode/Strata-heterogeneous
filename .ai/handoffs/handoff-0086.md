# Handoff 0086

## Tier
2

Measure with a finished driver. Scope the agent may change: run configs (build paths only), small fixes to
`mixed_split/tools/agent_session.py` and `agent_table.py` (say each in the report), and `.ai/` notes it is asked to
write. A failure inside the engine, the front stage or the server is reported, with Tier 1 escalation recommended,
not fixed here.

## Objective
The GPU part of handoff 0085, which did not run: the RC qualification sessions of `.ai/RC.md`, part 1.

## Context
- Read `.ai/PROJECT.md`, `.ai/RC.md` (updated: the probe rule, the session size), `.ai/STATUS.md` and report 0085.
- **Done by Chat AI since report 0085:**
  - `mixed_split/tools/agent_session.py`: the workload driver (`run`, `replay`, `stub`, `selftest`; read its
    docstring). It needs only the server's OpenAI endpoint. Token IDs are not needed: the probes compare the reply's
    text and its completion token count (`.ai/RC.md` criterion 2).
    - `selftest` passes against the built-in stub: identical requests for one seed, pruning, compactions, the request
      cap, probes and replays.
    - A full-size session against the stub: 589 requests, 2 compactions, about 7 million new tokens.
  - `mixed_split/tools/agent_table.py`: the summary tables (by context bucket, compactions, probes, resources,
    errors, request identity across runs).
- The builds `build-0085-*` at `fcdc3ed6` are current for the engine (nothing outside `mixed_split/tools/` and `.ai/`
  changed since). The mirrors point at them.
- Untested on the cards: the server's handling of the driver's request form (`tools`, assistant `tool_calls`, `tool`
  messages, `reasoning_budget_tokens` 0, `stream_options`). The scaled run shows it.

## Tasks
1. **Safety checks** (`.ai/PROJECT.md`) before the first GPU run, after the last and after any pause.
2. Run `python3 mixed_split/tools/agent_session.py selftest` (no GPU).
3. **The rename on the cards** (handoff 0085 task 3, short): the mixed mirror through the launch check, one request,
   a two-turn conversation over 8,192 tokens, a vision request, a clean shutdown; pipelined against serial exactness
   on the code prompt with `STRATA_IQ_MT_MIN=1`.
4. **Scaled sessions** (`run OUT --scaled`, seed 85), each against a server started for it (the way
   `serve_session.py` starts and stops servers: through `serve_mixed.py` and `check_launch.py`, SIGINT at the end,
   drain awaited):
   - mixed, with `STRATA_IQ_MT_MIN=1`; mixed, without it; production, without it.
   - Each finishes without an error. The three request sequences are identical (`agent_table.py` says so).
   - Replay the mixed-with-`STRATA_IQ_MT_MIN=1` probes, each on a freshly started mixed server with the same
     setting (restart the server between probes): all equal.
   - **Decide the full sessions' setting:** if the two mixed runs' medians of decode and prefill tok/s differ by less
     than 2%, the full sessions run with `STRATA_IQ_MT_MIN=1` on both configurations. Otherwise they run without it,
     and a probe that differs in a full session is not a failure by itself; the scaled run with the setting is the
     correctness evidence then. Write down which.
   - If the server refuses the request form, adapt the driver as little as possible (both configurations must still
     get the same requests) and say what changed.
5. **One full session per configuration**, seed 85, mixed first, then production, with the sampler on (default
   30 s). Replay each session's three probes on fresh servers of that configuration (restart between probes). If a
   single cycle runs past 90 minutes, finish that cycle, stop cooperatively and report.
6. `agent_table.py` over the two full sessions (and separately over the scaled ones), with `--markdown` into the run
   directory.

## Hardware Test Budget
**About 4 hours of GPU run time, over the 30-minute allowance** (announced in handoff 0085, not used; allowed unless
refused).
- Task 3: up to 10 minutes. Task 4: up to 30 minutes.
- Task 5: up to 3 hours 15 minutes; one session may take up to 2 hours; probe replays up to 30 minutes in total.
- If the budget runs out: stop cooperatively after the current request and report. Priority: the mixed full session
  and its probes, then the production full session.

## Constraints
- All rules of `.ai/PROJECT.md`. Run evidence under `.ai/runs/0086-*` (requests, samples, probes, tables, server
  logs; commit the tables and JSON files, not large logs). Never kill a GPU process by force. No two GPU jobs at once,
  except the two processes of one mixed run.
- Every process runs from the `build-0085-*` builds through `check_launch.py`.
- No changes to the engine, the front stage, the server or the mirrors' arguments.

## Completion Criteria
- Self-test, task 3 and the scaled sessions pass, with the setting decision written down.
- Both full sessions ran to the end (or the stop rule was taken and reported); probes replayed.
- `.ai/reports/report-0086.md` and all changes are committed and pushed to `feature/mixed-native`.

## Reporting
- Task 3 results; the scaled sessions and the setting decision with its numbers.
- Driver changes, if any.
- The `agent_table.py` tables of both full sessions, and the RC criteria 1 to 4, each marked passed, failed or not
  measured, with the numbers behind each.
- Safety lines, or "none". GPU run time used. What remains open for the soak.
