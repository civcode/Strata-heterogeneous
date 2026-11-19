# Report 0079 — read-ahead after a cancel, checkpoint overlap

**Completed.**
- **Task 3, the suspected desync:** it does not happen. I reproduced the exact situation on the cards: the front stage
  a chunk ahead when a cancel ends the read. Each time the next request went through vanilla's own path for a live
  session it cannot use, with no error. So task 4 (the fix) was skipped, as the handoff says to.
- **Task 5, the checkpoint overlap:** implemented, exact, and measured. **It gives no gain** on this rig, so I reverted it.
- `mixed_split_tests` passes in all three build trees, with the mock worker. `check_off_is_vanilla.py` reports 6 of 6.
- **Net change to the engine sources: none.** Since `8cde1265` only `hetero/tools/serve_session.py` changed (two new
  suites).
- **GPU time:** 213.5 s of the 1,200 s budget.

## Commits

- `40795177c`: `serve_session.py` gets a `cancel` suite (task 3 driver).
- `d172f54ed`: task 5, the checkpoint overlap (link protocol 5, two hooks, a CPU test).
- `bfb5d2431`: `serve_session.py` gets a `warm` suite (task 5 measurement).
- `fce1f20c8`: revert of `d172f54ed`, with the reason.
- `72d471957`: the report and the run evidence; this commit fixes the hashes after a rebase onto Chat AI's status note (`6c9d0034`). The revert's own message still names the pre-rebase hash `88a9ca70d` = `d172f54ed`.

**Builds:** `build-0071-hip`, `build-0071-cuda` and both vision builds were rebuilt and stamped three times this turn:
- at `8cde1265` for the task 3 runs;
- at `d172f54ed` for task 5;
- at `fce1f20c8` (the final state) for the last sanity run.

## Task 3: a cancel while the front stage is a chunk ahead

**Driver:** `serve_session.py RUN NAME CONFIG cancel --cancel-at P --cancel-after S`, with `STRATA_TRACE=1`. It runs
three requests on the mixed mirror (`mixed.example.json`):
1. A streamed request on the 32K prompt (32,093 tokens, chunks of 8,192). The client closes the socket S seconds after
   the host's first stage logs the chunk at P.
2. The same conversation one turn on (the 32K user turn, an assistant line, a new user line).
3. An unrelated short request.

| Run | Where the cancel landed (from the trace) | Front stage ahead? | Continuation | Unrelated | Engine |
| --- | --- | --- | --- | --- | --- |
| `r3-before-c0` (P 0, 2.0 s) | at 16,384, after its checkpoint | no: no chunk is read ahead past a checkpoint position (`pf_snapshot_due`) | 16,384 reused (checkpoint), 32 tokens | OK | up |
| `r3-before-c0b` (P 0, 1.2 s) | at 8,192: the front read [0, 8192) **and [8192, 16384)**; the host's first stage read only [0, 8192) | **yes, one chunk** | 0 reused: read from the start (Reset), 32 tokens | OK | up |
| `r3-before-c16k` (P 16384, 1.3 s) | at 24,576: the front read up to 32,088 (the last chunk, 7,512 tokens); the host up to 24,576 | **yes, one chunk** | **16,384 reused**: the checkpoint restored on both sides, 32 tokens | OK | up |

**Why it cannot happen:** vanilla already counts the live session as unusable after a read that stops partway.
- `pp_exact` is cleared while a batched part is in flight (`read_part`, `generate.cpp` near "a batched or pipelined
  part is in flight"). It is set again only when the part succeeds.
- A cancelled request keeps a live session only with `pp_exact` (#1620, "cancelled request keeps its conversation").
  Otherwise `live_ok` stays false from the request's start ("until this request has finished, the session is in
  between").
- So the next request never continues the live session. It restores a checkpoint, unparks a conversation, or reads
  from 0. On the front stage these are Restore, Unpark or Reset, and they replace its state without looking at its
  history (`f_session`, `Op::Restore` / `Op::Unpark` / `Op::Reset`).
- **No read-ahead is pending at a point where a cancel keeps the session:**
  - The #1620 path keeps it only between decode-style windows (`read_windows`), which never use PfStart.
  - A run's last chunk has `next_n` 0, so nothing is read ahead across runs.
  - The first stage's mid-prompt checkpoint stops the read-ahead (`pf_snapshot_due`).
- **The other ways a read ends between chunks:**
  - A failure in a host stage ends the engine (vanilla's `ERR`, return 1).
  - `stop_at_chunk` exists only for `--batch` (a BYIELD), which the mixed split refuses at start-up.
- **Before the Reset or Restore,** the next request sends Images and Steer. Each of them waits for the chunk in flight
  first (`settle`), so its rows land and are then ignored.

**Task 4: skipped.** Nothing in `hetero/` changed for it, so there is no desync test.

**Not run:** the 16K two-turn token check of task 4. It was tied to the fix. Task 5's exactness run includes a 16K
two-turn conversation, `long16k_t2` (below).

## Task 5: the checkpoint overlap

**What I built (`d172f54ed`, now reverted):**
- Link protocol 5:
  - **SnapStart** (`a` key, `b` n). The front stage runs Snapshot's check at once and refuses if its session does not
    hold those tokens. It then copies its state on the runner's thread and answers as soon as the copy is taken.
  - **SnapWait** (`c` keep). It returns the copy's result. The part goes into the front's book only with keep.
  - Any other operation ends the copy without keeping it.
- **Host side:** `HeteroHost::Snap`, a scope object.
  - In `checkpoint_at` (`parts == nullptr`) and in the first stage's `on_stage_chunk`, the copy starts before the host
    saves its own parts.
  - After the host's parts are saved, SnapWait(keep) carries the front's result into vanilla's
    "checkpoint could not be saved" handling.
  - On every other way out (the host's part failed or was skipped, an exception) it sends SnapWait(drop).
  - So vanilla's #3350 skip still holds on both sides.
- `STRATA_MIXED_CKPT_OVERLAP=0` turned it off. The CPU test `test_snapshot_two_step` covered:
  - the refusals;
  - keep, drop, and a copy ended by another operation;
  - a failed copy;
  - a restore of the kept part.

**Exactness:** I ran `run_engine.py` with `long32k:50, long32k_cut:30, long16k:50, long16k_t2:50`,
`STRATA_IQ_MT_MIN=1`, and the overlap off and on (`t5-exact-ov0/1`).
- **All four are identical** (`cmp_tokens.py`).
- `long32k_cut` (the first 20,000 tokens of the 32K prompt and a question) reused 16,384 tokens in both runs. In the
  "on" run that is the mid-prompt part saved through SnapStart / SnapWait.
- `long16k_t2` reused 16,073 tokens.

**Time to first token, warm code prompt:** `serve_session.py … warm`, 118 to 121 tokens, untraced, one fresh server per
sample, alternating. The table shows the server's prompt ms.

| Server | Overlap off (as before): 2nd / 3rd request | Overlap on: 2nd / 3rd request |
| --- | --- | --- |
| 1 | 337.4 / 336.3 | 336.2 / 335.5 |
| 2 | 335.9 / 336.0 | 338.1 / 334.9 |
| 3 | 365.3 / 335.1 | 337.1 / 335.0 |
| **Median** | **337.4 / 336.0** | **337.1 / 335.0** |

The first requests were 480 to 486 ms in both settings.

**Why there is no gain** (traced run against traced run, `wt-ov0` / `wt-ov1`, `STRATA_TRACE` and `STRATA_SPLIT_TIMING`):

| Warm request (2nd and 3rd) | Serial | Overlapped |
| --- | --- | --- |
| The host's own save (2 stages) | 16.7 / 15.9 ms | **24.5 / 24.3 ms** |
| The front's part | 11.5 ms (Snapshot round trip) | SnapStart 3.5 ms + SnapWait 0.02 ms on the host; the copy itself 20.4 ms on the front |
| Checkpoint wall time on the host | ≈ 28 ms | ≈ 28 ms |

- Run at the same time, the two copies slow each other down by about as much as the overlap saves. The host's save
  grows by 8 ms and the front's copy nearly doubles.
- Both copy GPU state into pageable host memory, so they share host-side resources. I did not dig further, because
  the outcome is what matters for the decision.
- **Decision:** I reverted it (`fce1f20c8`). Two hooks in a vanilla file and a protocol operation that save nothing
  should not be carried. The commit stays in history if a later rig or a pinned-memory checkpoint path changes the
  picture.

## Vanilla files changed

**None net.** `d172f54ed` changed `src/program/generate.cpp` (`checkpoint_at`, the stages' `on_stage_chunk`,
`HeteroHost`, `HeteroFrontBackend` and the front's link serving), all inside `#if defined(STRATA_MIXED_SPLIT)`. The
revert removed all of it. `check_off_is_vanilla.py` reports 6 of 6 at both commits.

## Safety and budget

- **Boot ID:** `86259517-dc02-4bae-be5f-ccc530423ae4` before the first GPU run, before each later batch of runs, and
  after the last. There was no pause in this turn. It is the same boot as at the end of report 0078.
- **Kernel log:** no new line matching the fault pattern. `kernel-before.log` and `kernel-after.log` are identical, 257
  lines, all driver initialisation at 08:03. Every run's own delta has no fault line.
- **Core dumps:** none (`cores-before.log`, `cores-after.log`).
- **GPU time:** 213.5 s. Every run was sequential and under 2 minutes; the longest was 24.8 s (engine) and 23.2 s (server session).
  - Server sessions: 163.9 s (`.ai/runs/0079-server/gpu-seconds.txt`). That covers 3 cancel runs, 6 warm runs, 2 traced
    warm runs, and 1 final sanity run of the reverted build.
  - Engine runs: 49.6 s (`.ai/runs/0079-measure/gpu-seconds.txt`).
- **Process handling:** no process was killed or signalled, except the servers' SIGINT at the end of each session.

## Evidence

- `.ai/runs/0079-server/`:
  - `r3-before-*`: task 3, with the trace lines and the log lines for each request in `*-result.json`;
  - `w-ov{0,1}-{0,1,2}`, `wt-ov{0,1}` (traced) and `w-final`: task 5.
- `.ai/runs/0079-measure/`: `t5-exact-ov{0,1}`. Its `prompts.json` is committed because it adds `long16k_t2` and
  `long32k_cut`.
- `.ai/runs/0079-build/`: build logs and test output. `.ai/runs/0079-safety/`: the boot IDs, kernel logs and core-dump
  lists.

## Open

- **Short prompts:** report 0078's options remain: an MMQ front, or fewer front layers. The checkpoint overlap is now
  measured and does not help on this rig.
- **`.ai/DIVERGENCE.md`:** no update needed for this turn. The engine sources are unchanged.
