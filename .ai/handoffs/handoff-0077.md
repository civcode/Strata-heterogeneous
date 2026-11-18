# Handoff 0077

## Tier
1

Pipelined decode windows in mixed mode. The front stage must take part in vanilla's pipelined loop: asynchronous
windows, a second verifier, and snapshot and undo of a wrong guess. All of it happens across two processes and two GPU
runtimes, and is judged only on the cards. A wrong step shows up as a different token sequence, not as a crash.

## Objective
`--pipeline-windows 2` works in mixed mode and emits exactly the serial loop's tokens. Measure what it gains against
mixed without it and against production.

## Context
- Read `.ai/PROJECT.md`, `.ai/STATUS.md`, `.ai/DIVERGENCE.md` and reports 0075 and 0076. With `STRATA_MIXED_SPLIT`
  off, the engine sources must stay vanilla's line for line: run `hetero/tools/check_off_is_vanilla.py` before every
  commit that touches `src/` or `include/`.
- **Report 0075:** a mixed decode window is fully serial. The front stage answers the host's window call only after
  its CUDA window has finished, because vanilla's `Verifier::run` synchronises its stream. Per window on the code
  prompt: CUDA 3.3 ms (19%), first R9700 5.45, second 5.68, drafter 1.6, commit 0.7, gaps about 0.9. The link adds
  0.03 ms.
- **Report 0076:** mixed without pipelining decodes on par with production (HIP-only with `--pipeline-windows 2`):
  117.8 / 118.7 / 131.2 against 107.9 / 122.7 / 132.2 tok/s (code / long / 16K). Today mixed mode refuses
  `--pipeline-windows` above 0 and forces the default to 0 (`generate.cpp`, the mixed-split checks).
- **Vanilla's pipelined loop** (`generate.cpp`, "--pipeline-windows 2: the decode windows with the two cards
  overlapped"):
  - The front stages (`pl_nf = n_stages - 1`, every stage but the last) run window K+1 on the guess that K is accepted
    whole, while the last stage verifies K.
  - Each stage has two verifiers, one per window parity (`stage_ver_p(st, par)`), on its own stream (`pl_stream`).
  - Windows are launched and polled, not run blocking: `Verifier::pl_launch`, `service`, `done`, `pl_finish`,
    `pl_commit_async`, `commit_deferred`.
  - A wrong guess is undone by restoring the stage's GDN state from a snapshot (`PlSnap`, `snap_take`) and replaying
    K's commit with its real count (`undo_w`, `undo_keep`).
  - Vanilla states that every emitted token is the serial loop's, bit for bit with `STRATA_IQ_MT_MIN=1`.
  - `STRATA_PIPELINE_FORCE_MISS=k` makes every k-th speculative window's guess wrong, to exercise the undo path.
  - `STRATA_PIPELINE_TRACE=FILE` logs the loop's events.
- **Production** runs `--pipeline-windows 2` with `STRATA_PIPELINE_AUTO=0` and `STRATA_MTP_STAGE0=0`, so the draft
  layer stays off the first card.

## Design (decided)
- **The remote front stage is part of the host's stage 0,** as it is in the serial loop (`Verifier::set_front`).
  Vanilla's pipelined loop stays as it is: its stage 0 is the first R9700 together with the CUDA front stage, and
  `pl_nf` stays the host's count.
  - When stage 0's verifier of parity q launches a window, it first launches the front stage's window of parity q.
  - When it commits, the front stage commits. When it snapshots or undoes, the front stage does too.
- **The draft layer stays on the host,** as in production (`STRATA_MTP_STAGE0=0`). It never goes to the CUDA card.
- **First step, inside this design: the front stage answers right after launching.** The front stage's window call
  returns once its window is launched, not when it ends. The first R9700's graph waits on the shared ready word on the
  GPU, as the hand-off was built to do. The front stage's next call (commit, snapshot, the next window) waits for its
  own window first.

  This changes nothing in what is computed. It is the asynchronous launch pipelining needs, and on its own it should
  shorten the serial window by part of its gaps. Commit it separately, with its own check (task 3).
- **The front stage** gets the same per-parity machinery vanilla gives a front stage: two verifiers, a snapshot of its
  GDN state per parity, and an undo that restores it and replays a commit with the real count.
  - **Hand-off:** the shared region needs a hand-off buffer and a ready word per parity.
  - **Link:** it carries the parity, the snapshot, the undo and the asynchronous commit.
  - Extend `hetero/` with tests for each new link operation.

## Boundaries you decide within
- Vanilla's pipelined scheduling, guessing, gating and undo logic is not changed or rewritten. The hooks only repeat
  on the front stage what vanilla does on stage 0. Any change to a vanilla file is the smallest hook, inside
  `#if defined(STRATA_MIXED_SPLIT)`.
- Keep every safety gate:
  - the identity check;
  - a window is published only after the front stage acknowledged its commit (pipelined: its speculative commit is
    undone or confirmed exactly as stage 0's);
  - the bounded hand-off wait (`release_gpu_waits` on an error, as vanilla's `die` does);
  - cooperative shutdown.
- Requests vanilla runs serially in pipelined mode (repetition penalties, coupled draft sampling) stay serial.
- If the design cannot work, stop and report why. A redesign is not an acceptable outcome.

## Tasks
1. **Kernel log and core dumps:** record them before the first GPU run and after the last.
2. **Rebuild** the HIP and CUDA trees, and stamp them.
3. **The asynchronous front window,** serial loop, no pipelining:
   - greedy, MTP on, the code and the long prompt;
   - with `STRATA_IQ_MT_MIN=1`, the token IDs must equal the previous commit's exactly;
   - then, untraced, decode tok/s against the previous commit (three samples), and one traced run with
     `STRATA_SPLIT_TIMING=1` showing the first R9700 now waiting on the ready word.
4. **Pipelined windows:** implement the design.
   - **Exactness:** with `STRATA_IQ_MT_MIN=1`, greedy, the code and the long prompt, mixed with
     `--pipeline-windows 2` emits exactly the token IDs of mixed without it.
   - **Undo path:** the same with `STRATA_PIPELINE_FORCE_MISS=3`.
   - **Request types:** one sampled request (seeded), compared with the serial loop at the same seed. Vanilla promises
     the same Philox draw per position; report what you find. Also a request with a repetition penalty, which must
     fall back to serial with vanilla's message.
   - **Through the server** with the mixed mirror plus `--pipeline-windows 2`: a streaming and a non-streaming request,
     a cancelled request followed by a normal one, a two-turn conversation over 8,192 tokens (parked and reused), a
     vision request, `cvec=0`, and a clean shutdown of both processes.
5. **Measure,** untraced, three samples each, alternating, on the code, the long and the 16K prompt:
   - mixed serial (the mirror as committed);
   - mixed pipelined;
   - production (`hiponly.example.json`).

   Report prefill and decode tok/s, drafts accepted, and the share of speculative windows that were right
   (`pl_on` / `pl_spec`, or vanilla's own log line).
6. **Configs:** if pipelined mixed passes task 4, add `--pipeline-windows 2` (and, if they matter, the two
   production environment variables) to `mixed.example.json`, and say so in its comment.

## Hardware Test Budget
30 minutes of GPU run time (the project's allowance). Every single run under 2 minutes. Priority order: tasks 3, 4,
5. Task 3 alone is a useful result if the rest does not fit.

## Constraints
- All rules of `.ai/PROJECT.md`. Run evidence goes under `.ai/runs/0077-*`; reusable drivers go into `hetero/tools/`.
  Never kill a GPU process by force. No two GPU jobs at once, except the two processes of one mixed run.
- A new GPU fault line, a crash outside the run or a reboot ends the handoff at once.
- Commit in small steps; task 3 is its own commit.

## Completion Criteria
- Task 3: exact token IDs, and the timing shows the front window no longer inside the link round trip.
- Task 4: exact token IDs with and without forced misses; the server checks pass.
- Task 5: the table with three samples per configuration and prompt.
- `mixed_split_tests` (with tests for the new link operations) and `check_off_is_vanilla.py` pass.
- `.ai/reports/report-0077.md` and all changes are committed and pushed to `feature/mixed-native`.
- If stopped early: **Partial**, with the last task that works.

## Reporting
- Every vanilla file changed, with the function and one line on why; the link operations added; design decisions
  made inside the boundaries.
- The exactness results, with the first differing position if any.
- The tables of tasks 3 and 5, and the speculative-window hit rate.
- Kernel log and core-dump lines, or "none". GPU run time used. What remains open, for example `--batch` with
  pipelined groups.
