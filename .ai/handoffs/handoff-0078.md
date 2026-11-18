# Handoff 0078

## Tier
1

Three performance steps in mixed mode. Each changes when the front stage works, not what it computes, and each is
judged by exact token IDs and timing on the cards.

## Objective
1. The front stage learns when its pipelined window has been judged wrong.
2. Find and cut the fixed per-request cost on short prompts.
3. Overlap prompt chunks across the link: the front stage reads chunk c+1 while the first R9700 reads chunk c.

## Context
- Read `.ai/PROJECT.md`, `.ai/STATUS.md`, `.ai/DIVERGENCE.md` and report 0077. With `STRATA_MIXED_SPLIT` off, the
  engine sources must stay vanilla's line for line: run `hetero/tools/check_off_is_vanilla.py` before every commit
  that touches `src/` or `include/`.
- **The mixed mirror** (`hetero/configs/mixed.example.json`) now runs `--pipeline-windows 2` with
  `STRATA_PIPELINE_AUTO=0` and `STRATA_MTP_STAGE0=0` (Human User, 2026-11-18: keep it). All comparisons in this turn
  use it as committed, except where a task says otherwise.
- **Report 0077, pipelined windows:**
  - Guesses are right 37 to 54% of the time, so about 60% of speculative windows are undone. A right guess saves 4 to
    11 ms.
  - The front stage computes a window that has already been judged wrong to its end, because it is not told.
  - Vanilla tells its own stage 0 (`doom_skip`: a doomed window gets empty plans, so it frees stage 0 sooner;
    `STRATA_PIPELINE_DOOM_SKIP=0` turns that off).
  - `STRATA_SPLIT_TIMING` does not cover pipelined windows yet.
- **Report 0076, short prompts:** mixed takes about 80 ms longer than production before the first token on the
  118-token code prompt (prefill 265 against 324 tok/s). With pipelining it is a further 6% slower (report 0077).
  Candidates, none measured:
  - the per-request session operations (Reset, Abandon, Steer, Images);
  - the front stage's prompt pass, which still runs synchronously;
  - with pipelining, the front stage handling the pipelined prompt windows one at a time.
- **Prompt chunks today** (`Prefill::set_remote_in`, `remote_fill_`):
  - for each chunk, the host asks the front stage for the chunk's rows and waits for them before its first R9700
    reads that chunk, so the CUDA card and the AMD cards never work at the same time on a prompt;
  - long prompts are read in 8,192-token chunks: a 16K prompt is two chunks, a 5K prompt one;
  - report 0075: `--prefill-pipe` (more, smaller chunks) costs 14% in mixed mode for exactly this reason. It stays off.

## Tasks
1. **Kernel log and core dumps:** record them before the first GPU run and after the last.
2. **Rebuild** the HIP and CUDA trees, and stamp them.
3. **Pipelined timing, then the wrong-guess notice.**
   - Extend `STRATA_SPLIT_TIMING` (logging only, guarded) to pipelined windows. Per request, report:
     - the front stage's GPU time for right and for undone windows;
     - how long an undo waits for the front stage;
     - how often the CUDA card is the stage that holds the next window back.
   - From that, judge whether a notice can save time. A window not yet launched on the front stage when its guess is
     judged wrong can be dropped; a launched graph cannot be stopped.
   - If it can, add the notice as the smallest link operation, mirroring vanilla's `doom_skip` on stage 0. Check:
     - token IDs equal without the notice, also with `STRATA_PIPELINE_FORCE_MISS=3`;
     - decode tok/s on the code and the long prompt, three samples each.
   - If it cannot, report the numbers and leave the code as it is.
4. **The short-prompt cost.**
   - Measure where the time before the first token goes, for the code prompt in mixed mode (pipelined and serial)
     against production: every link call per request and its round trip, the front stage's prompt pass, and the host's
     own setup. Logging only, guarded.
   - Then cut the largest part inside the boundaries. Examples:
     - send nothing that has not changed (no Steer or Images message when they equal the last request's);
     - fold per-request messages into one;
     - let the front stage's prompt pass return at launch, as its decode window does since report 0077;
     - read the pipelined prompt windows on the front stage as vanilla does on stage 0.
   - Check: token IDs unchanged, and time to first token and prefill tok/s on the code prompt, three samples each.
5. **Prompt chunks across the link.**
   - While the first R9700 reads chunk c, the front stage already reads chunk c+1. That needs two row buffers in the
     shared region (one per chunk in flight, with a ready word each), and a chunk request that returns at launch.
   - The chunk sequence and every chunk's content stay as vanilla plans them; only the front stage starts earlier.
   - Check:
     - the first 50 greedy token IDs (with `STRATA_IQ_MT_MIN=1`) equal those without the overlap, on the long and the
       16K prompt;
     - prefill tok/s on the long and the 16K prompt and on a new 32K prompt, three samples each, against the previous
       commit.
   - Report whether `--prefill-pipe` now helps in mixed mode (one R value, the 16K and the 32K prompt), but leave it
     off in the config unless it clearly wins.
6. **Final table,** untraced, three samples each, alternating: mixed (the mirror at the end of this turn) against
   production, on the code, the long and the 16K prompt. Prefill and decode tok/s, and time to first token.

## Boundaries you decide within
- As in handoff 0077. The smallest hooks in vanilla files, inside `#if defined(STRATA_MIXED_SPLIT)`. No change to
  vanilla's scheduling, chunk planning, guessing or undo logic; the front stage only starts or skips its part earlier.
- Every safety gate stays:
  - the identity check;
  - publish after the front stage's commit acknowledgement;
  - the bounded hand-off waits, with `release_gpu_waits` on an error;
  - cooperative shutdown.
- A step that cannot be done inside these boundaries is reported, not forced.

## Hardware Test Budget
40 minutes of GPU run time. That is above the project's 30-minute allowance, which the Human User was told when this
handoff was handed over. Every single run under 2 minutes. Priority order: tasks 3, 4, 5, 6.

## Constraints
- All rules of `.ai/PROJECT.md`. Run evidence goes under `.ai/runs/0078-*`; reusable drivers go into `hetero/tools/`.
  Never kill a GPU process by force. No two GPU jobs at once, except the two processes of one mixed run.
- A new GPU fault line, a crash outside the run or a reboot ends the handoff at once.
- Commit in small steps; each task is at least one commit.

## Completion Criteria
- Each of tasks 3 to 5 is either done with its check passed, or reported with the numbers showing why not.
- Token IDs are unchanged everywhere a check asks for it.
- `mixed_split_tests` (with tests for any new link operation) and `check_off_is_vanilla.py` pass.
- `.ai/reports/report-0078.md` and all changes are committed and pushed to `feature/mixed-native`.
- If stopped early: **Partial**, with the tasks reached.

## Reporting
- Every vanilla file changed, with the function and one line on why; link operations added.
- The pipelined timing; the short-prompt breakdown before and after; the prompt-chunk overlap's prefill table; the
  final table.
- Kernel log and core-dump lines, or "none". GPU run time used. What remains open.
