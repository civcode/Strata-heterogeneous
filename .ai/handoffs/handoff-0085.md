# Handoff 0085

## Tier
2

Build, write a driver, measure. Scope the agent may change: build directories, run configs, new drivers and table
scripts in `mixed_split/tools/`, and `.ai/` notes it is asked to write. A failure inside the engine or the front stage
is reported, with Tier 1 escalation recommended, not fixed here.

## Objective
The first half of the release-candidate qualification (`.ai/RC.md`, read it in full): fresh 0.2.15 builds, the
pi-like session driver, scaled smoke sessions, then one full long-context session per configuration with the
correctness probes.

## Context
- Read `.ai/PROJECT.md` (the stage is now **release candidate**; long runs are allowed when a handoff sets them),
  `.ai/RC.md`, `.ai/STATUS.md` and report 0084.
- **The branch moved back to vanilla 0.2.15** (force-push, `7e0661a7` and later). Reset the worktree to
  `origin/feature/mixed-native`; the 0.2.16 builds of report 0084 are not usable. The 0.2.15 layout: `src/`,
  `include/`, `serve/server.py`, `tools/`. The MMQ build options exist again.
- The rename to `mixed_split` was done on 0.2.15 (`5d1406fa`). Chat AI checked it without a GPU (library tests,
  `check_off_is_vanilla.py` 6 of 6 against `c775728`, syntax check, Python tools). It has not been on the cards on
  0.2.15.
- The mirrors in `mixed_split/configs/` still name `build-0071-*` paths. Point them at this handoff's builds; change
  nothing else in them except `--max-context`, which must be 262144 (production).

## Tasks
1. **Safety checks** (`.ai/PROJECT.md`): boot ID, kernel log and core dumps before the first GPU run, after the last,
   and after any pause.
2. **Builds** at the current commit, new directories `build-0085-*`, `-DSTRATA_MIXED_SPLIT=ON`, 28 jobs: CUDA (13.0,
   sm_89) and HIP (gfx1201, `-DSTRATA_PREFILL_MMQ=ON -DSTRATA_MMQ=ON`), and both vision helpers. Stamp each. Run
   `mixed_split_tests` in each tree. No vanilla GPU test suite this time.
3. **The rename on the cards** (short): the mixed mirror through the launch check, one request, a two-turn
   conversation over 8,192 tokens, a vision request, a clean shutdown. Pipelined against serial exactness on the code
   prompt with `STRATA_IQ_MT_MIN=1`.
4. **The driver `mixed_split/tools/agent_session.py`**, implementing the workload model of `.ai/RC.md` exactly:
   - every number of the model is a flag with the RC value as its default; `--seed`; `--scaled` (compaction at 92% of
     32,768, two cycles);
   - token lengths are measured with the server's tokenizer (the tokenize endpoint, or the model's tokenizer files);
   - pi's request form (tool definitions, assistant `tool_calls`, `tool` messages) if `serve/server.py` handles it in
     0.2.15; otherwise the closest form it handles, and say which;
   - a fixed number of generated tokens per request (ignore the end-of-text token if the server has an option for it;
     if not, record the real count);
   - the correctness probes of `.ai/RC.md` criterion 2: the driver saves each probe request to a file, and a second
     command replays the saved probes against a freshly started server and compares the first 64 token IDs;
   - output: one JSON line per request (context length, new and reused tokens, time to first token, prefill and decode
     tok/s, generated tokens, cache path taken where the log shows it) and a sampler every 30 s (VRAM per card,
     available RAM, the engines' resident memory, swap);
   - `mixed_split/tools/agent_table.py`: the summary of `.ai/RC.md` (by context bucket; per compaction; per session).
   - Before any GPU use, run the driver against a stub (a tiny local fake server) to check the bookkeeping: lengths,
     pruning, compaction trigger, identical request sequence for the same seed.
5. **Scaled smoke sessions**, `--scaled`, mixed then production. Both finish without `ERR`; their request sequences
   are identical; one probe each passes.
6. **One full session per configuration**, mixed first, then production, same seed, with probes after a pruning step,
   after the first compaction, and at 200K tokens or more. Replay the probes on fresh servers (mixed mode for the
   mixed probes, HIP-only for the production ones). Watch the sampler; if a single cycle runs past 90 minutes, finish
   that cycle, stop cooperatively and report rather than continue.

## Hardware Test Budget
**About 4 hours of GPU run time, over the 30-minute allowance** (announced to the Human User; allowed unless refused).
- Task 3: up to 10 minutes. Task 5: up to 20 minutes.
- Task 6: up to 3 hours 15 minutes in total. A single run (one session) may take up to 2 hours; probe replays on
  fresh servers up to 30 minutes in total.
- If the budget runs out, stop cooperatively after the current request and report what was measured. Priority: the
  mixed full session and its probes, then the production full session.

## Constraints
- All rules of `.ai/PROJECT.md`. Run evidence under `.ai/runs/0085-*`. Never kill a GPU process by force. No two GPU
  jobs at once, except the two processes of one mixed run.
- Every process runs from this handoff's builds through `check_launch.py`.
- Do not change the engine, the front stage or the mirrors' arguments (apart from the build paths and
  `--max-context`).

## Completion Criteria
- Builds and `mixed_split_tests` pass; the task 3 checks pass.
- The driver and table script are committed, with the stub check passing.
- The scaled sessions pass in both configurations.
- One full session per configuration ran to its end (or the stop rule above was taken and reported), with the probes
  replayed.
- `.ai/reports/report-0085.md` and all changes are committed and pushed to `feature/mixed-native`.

## Reporting
- Builds, tests, task 3.
- The driver: request form used, how reused tokens are read, anything in `.ai/RC.md` that could not be done as
  written.
- The summary tables of both full sessions side by side, and the RC criteria 1 to 4, each marked passed, failed or
  not measured.
- Safety lines, or "none". GPU run time used. What remains open for RC handoff 2 (the soak).
