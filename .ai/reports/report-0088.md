# Report 0088

**Status: completed.** The live/fresh differences also occur on the HIP-only production mirror (vanilla 0.2.15 plus the carried pinned-readback patch), so they are not specific to the mixed split. They are not proven to be a particular numerical effect. No unpatched vanilla build was tested.

## Probe comparison

First differences below are 1-based Unicode character positions in the reasoning text, not token IDs. Each replay used a newly started server, `STRATA_IQ_MT_MIN=1`, greedy sampling, and the saved request. Equal means the driver's comparison of content, reasoning, tool arguments and completion-token count; it does not compare token IDs.

| Configuration / comparison | Probe | Equal? | First difference |
| --- | --- | --- | --- |
| Mixed, fresh vs fresh | 0087 pruned | Yes | None |
| HIP-only, live vs fresh | pruned | No | Character 9: `The user asks` / `The user's query` |
| HIP-only, live vs fresh | deep | No | Character 35: `read` / `` `write` `` |
| HIP-only, live vs fresh | compacted | No | Character 5: `most` / `user` |
| HIP-only, prompt and conversation caches disabled, live vs fresh | pruned | Yes | None |

The 0087 mixed, 0088 HIP-only and 0088 no-cache scaled sessions sent the same 37 request SHA-256s, seed 85. Both 0088 sessions completed all requests with zero errors and one compaction. Cached-token totals were 499,911 and 0 respectively. The no-cache result is one probe, not a proof that caching causes every difference. Fresh mixed replay outputs were equal to each other but each differed from 0087's live output. Results and logs: `.ai/runs/0088-mixed-fresh*/`, `.ai/runs/0088-hip*/`, `.ai/runs/0088-nocache*/`.

## Driver and reasoning

`serve/server.py:5191` defines `reasoning_budget_tokens: 0` as **no limit**, not thinking disabled. Thus the 0087 and control requests asked for unlimited thinking and produced reasoning. After the comparable measurements, fixed `mixed_split/tools/agent_session.py`: a zero `--reasoning-budget` now also sends `chat_template_kwargs: {"enable_thinking": false}`; positive budgets continue to request the existing token cap. Added selftest coverage for both forms. This changes subsequent default request sequences; the table above uses the original requests and is not a test of the new non-thinking probes. A short HIP-only run with the fixed driver produced a probe with empty reasoning and tool calls (`.ai/runs/0088-no-thinking-probe/`). Driver selftest passed; `serve.test_effort_paths` passed 12 tests.

The previously uncommitted driver work was committed first as `71dd94250`: `--max-cycle-min` stops after a compaction when the cycle exceeds its limit, and AMD GPU-memory sampling falls back to `amd-smi` when `rocm-smi` is unavailable. No table script or engine code was changed.

## Safety

`HSA_ENABLE_SDMA=0` was set for every GPU process. Sequential GPU job wall times including startup and drain totaled 416.4 seconds, within the 20-minute budget; no run exceeded 10 minutes or a 64K context. GPU processes were launched with the prescribed signal reset and drained using `stop_cooperatively`; no processes of the run remained between jobs. No new matching kernel lines or coredumps since boot. Boot ID stayed `ffcbbb74-121f-46fa-a930-ca08f55e881c`. `free -m` reported 0 MiB swap used before and after; `swapon --show` reported 256 KiB both times. Safety snapshots and test logs: `.ai/runs/0088-safety/`.
