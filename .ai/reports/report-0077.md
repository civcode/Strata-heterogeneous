# Report 0077 — pipelined decode windows in mixed mode

**Completed.** Tasks 1 to 6 are done.
- `--pipeline-windows 2` works in mixed mode. It emits exactly the serial loop's token IDs, with and without forced
  misses, and for a seeded sampled request.
- The asynchronous front window (task 3) is its own commit, with exact tokens.
- `mixed_split_tests` passes in both build trees and standalone. `check_off_is_vanilla.py` reports 6 of 6.
- GPU time: 563 s of the 1,800 s budget. No kernel fault line, no new core dump, no reboot.

Commits:
- `9d1546f59`: task 3, the asynchronous front window; `run_engine.py`.
- `7a5fab5fe`: task 4, pipelined windows; `cmp_tokens.py`.
- This commit: the configs (task 6), `serve_session.py`, the README, this report and the run evidence.

`build-0071-hip`, `build-0071-cuda` and both vision builds are stamped at the task-4 source commit. `check_launch`
accepts both example configs.

## Design as built (inside the handoff's boundaries)

- **Task 3, the asynchronous window:** the front stage answers Run once its window graph is launched.
  - Vanilla's `Verifier::run` runs unchanged on a `WindowRunner` thread (`hetero/include/strata/hetero/front_runner.hpp`).
    The existing launch-mark hook, after `cudaGraphLaunch`, signals "launched". The ready word was already reset in
    `stage_inputs` by then.
  - Every later front call (commit, prompt chunk, session op, images, the next window) first waits for the runner.
    An error the window ended with is that call's error.
  - The host's first R9700 now waits on the ready word on its GPU.
- **Task 4, pipelined windows:** the front stage is part of host stage 0, and vanilla's loop is unchanged.
  - **Hooks in `Verifier`:** stage 0's verifier of parity q calls the front stage first in `pl_launch` (link PlRun)
    and in `pl_commit_async` (PlCommit). These are the existing `FrontCalls`, plus two members.
  - **The undo:** in the decode loop's undo, for stage 0 only, the host sends PlUndo(parity of `undo_w`) before the
    replayed commit.
- **Snapshots:** the front stage takes its GDN snapshot of parity q at every PlRun.
  - Vanilla's `snap_take` is called before every stage-0 launch. In the non-overlapped mode the snapshot is taken just
    before A's commit, and stage 0 holds the same state at both points.
  - So the snapshot needs no message of its own. The copy runs on a side stream, and each commit of parity q waits for
    it.
  - `ple_prev` is saved with the snapshot and restored by the undo (vanilla's `undo_ple`). The front stage holds layer 1,
    the PLE stage.
- **Front stage with `--pipeline-windows`:** the host now passes the option on.
  - The front stage sets its own `pipeline_windows` to 0 and runs no loop of its own.
  - It gets what vanilla gives a pipelined stage 0: `ver_b` and `ver` on one stream, `set_always_publish`,
    `capture_all` before it serves the link, and 2 × 37.4 MiB of snapshots.
  - 256 MiB more VRAM reserve (vanilla's kPipeWindowMib + kPipeSnapMib).
  - A pipelined window is served on the runner thread (`service` / `done` / `pl_finish`) with `drive.d.plan` set to that
    verifier's sink. Before any non-pipelined call the stream and the side stream are synchronised (vanilla's `pl_drain`).
- **Hand-off:** a hand-off buffer and a ready word per parity (shared region slots 3 and 4; link protocol 3).
  `read_windows_pl` needs them, because it queues window K+1 on stage 0 while K is still in flight.
- **One window in flight on the front stage at a time:** a front PlRun or PlCommit first waits for the previous front
  window. Vanilla's decode loop never launches on stage 0 with a stage-0 window in flight, so decode loses nothing. The
  pipelined prompt read does serialise on it (see Open).
- **Safety gates kept:**
  - the identity check, now also on the five-slot layout;
  - a front commit is acknowledged only after its window finished without an error;
  - the bounded waits on both parities' ready words (`release_gpu_waits` covers `ver_b`'s incoming word);
  - cooperative shutdown, with the runner settled before the link ends;
  - repetition-penalty and coupled-draft requests stay serial (vanilla's own check).

## Vanilla files changed (every change inside `#if defined(STRATA_MIXED_SPLIT)`)

- **`include/strata/core/verify.hpp`:** `FrontCalls::pl_launch` and `pl_commit`. They let the pipelined calls reach
  the front stage.
- **`src/core/verify.cpp`:**
  - `Verifier::pl_launch`: launches the front stage's window of this parity before its own;
  - `Verifier::pl_commit_async`: has the front stage queue its commit first.
- **`src/program/generate.cpp`:**
  - `HeteroHost::pl_run` / `pl_commit` / `pl_undo` and `HeteroFrontBackend::f_pl_*`: the link calls.
  - The mixed refusal list:
    - `--pipeline-windows` is accepted. Only the default (−1) is still forced to 0.
    - The option is passed to the front stage. The front stage turns it into `front_pipe`, plus the VRAM reserve.
  - The shared region: slots 3 and 4, and the front stage's layout check.
  - Host:
    - `ver`'s FrontCalls get the parity-0 hooks.
    - `ver_b.set_stage` starts at `front_lb` and reads slot 3, with its ready word and parity-1 hooks. This is the
      guarded alternative of the vanilla line, which stays in `#else`.
  - Front stage:
    - `ver.set_stream` and `set_always_publish`;
    - the `ver_b` init and the snapshots (`HetSnap`);
    - `capture_all` before serving;
    - `WindowRunner`: asynchronous `f_run`, and every call settles first;
    - `f_pl_run`, `f_pl_commit`, `f_pl_undo`.
  - Pipelined decode loop, the undo block: `hetero.pl_undo` for stage 0 before the replayed commit.

**Link operations added:**
- PlRun (`a` T, `b` pos0, `c` parity, the tokens);
- PlCommit (`a` n_keep, `c` parity; may be replayed after an undo);
- PlUndo (`c` parity).

The worker refuses:
- a commit or undo without a window of that parity, or a commit longer than the window;
- PlRun while a serial window waits for its commit.

Any serial window, prompt chunk, images or window-ending session operation ends the pipelined windows.
`FrontSessionBook` tracks a window per parity: `on_pl_run`, `on_pl_commit`, which must continue the session at its
window, and `on_pl_undo`.

**New CPU tests:**
- `test_pipelined_link`: both parities' hand-off and ready word, a speculative commit, the undo, the replay, a second
  replay refused, the out-of-step checks, the rules above, and a failed launch.
- `test_window_runner`: the launch signal, an error before and after the launch, and an error carried into the next
  start.

## Task 3: the asynchronous front window (serial loop)

- **Exactness**, with `STRATA_IQ_MT_MIN=1`, greedy, MTP: code 300 / 300 and long 200 / 200 tokens, **identical** to the
  previous commit (`base-exact` against `t3-exact`).
- **Untraced decode tok/s** (`mixed.example.json` as committed then, one engine per sample, engine protocol):

  | Prompt | Previous commit (`94d11a83`) | Task 3 (`9d1546f5`) |
  | --- | --- | --- |
  | Code | 128.1 / 128.3 / 128.4, median **128.3** | 128.8 / 128.2 / 130.9, median **128.8** (+0.4%) |
  | Long | 117.1 / 116.6 / 117.5, median **117.1** | 117.7 / 117.8 / 117.7, median **117.7** (+0.5%) |

  Prefill does not change (code 168.7 against 170.6, long 2,307 against 2,307). The drafts are the same (176/253 and
  116/193).
- **Traced** (`t3-traced`, `STRATA_SPLIT_TIMING=1`), per window, against report 0075:

  | | Code | Long |
  | --- | --- | --- |
  | Link run round trip | **0.097 ms** (was 3.36) | **0.380 ms** (was 3.90) |
  | First R9700: GPU wait on the ready word | **3.74 ms** (was 0.009) | **3.96 ms** (was 0.009) |
  | Front stage: GPU window / its run call | 4.18 / 0.064 ms | 4.54 / 0.322 ms |
  | First R9700 GPU window | 5.42 ms | 5.80 ms |

  The front window is no longer inside the link round trip. The R9700 now waits for it on the GPU. The gain is small
  because the chain stays serial; only host-side launch gaps overlap.

  The front stage's GPU span reads 4.2 ms here against 3.3 ms in report 0075. Its events now sit on the runner thread
  beside the link thread. Not investigated.

## Task 4: pipelined windows (`--pipeline-windows 2`, `STRATA_PIPELINE_AUTO=0`, `STRATA_MTP_STAGE0=0`)

All runs: greedy unless stated, MTP, `STRATA_IQ_MT_MIN=1`, through the engine protocol.

| Check | Result |
| --- | --- |
| Mixed serial (this build) against task 3 | code, long: identical |
| **Exactness:** pipelined against serial | code 300/300, long 200/200: **identical**. 104 / 71 speculative windows, 67 / 43 undone |
| **Undo path:** `STRATA_PIPELINE_FORCE_MISS=3` (+`STRATA_PIPELINE_DEBUG=1`) against serial | code, long: **identical**. 81 / 52 undone |
| **Sampled request** (`temperature=0.8 top_p=0.95 seed=1234`), pipelined with forced misses, against serial | 70/70 tokens **identical** (EOS at 70). 28 of 31 speculative windows undone |
| **Repetition penalty** (`penalty_last_n=64 penalty_repeat=1.1`) | decodes serially; vanilla's line "a request with repetition penalties (penalty_last_n) decodes serially"; 20/20 identical |

No first differing position anywhere. Vanilla's promise of the same Philox draw per position holds for the sampled
request, including across undone windows.

**Through the server** (`.ai/runs/0077-server/mixed-pipe.json`: the mirror plus `--pipeline-windows 2` and both
variables):
- **`pipe-checks`:** all requests status 200.
  - Health and the model list.
  - Streaming: 120 chunks. Non-streaming and sampled requests.
  - A stream cancelled after 8 chunks, then a normal request.
  - Two short turns, the second reusing 59 tokens.
  - Fast and slow readers.
- **`pipe-features`:**
  - **16K two turns:** the first turn read 16,093 tokens. The request between the turns parked 16,156 tokens. The
    second turn restored 16,088 from a checkpoint (`cache_n` 16,088).
  - **Vision** (318-token PNG prompt) and **`cvec=0`**: both OK.
- **Shutdown:** every session ended with "the link ended cleanly" and "the front stage ended cleanly", exit status 0.

## Task 5: measurement (untraced, through the server, one fresh server per sample, alternating)

`serve_session.py … measure`. Prompts are the report 0076 ones, with a "Measurement i" prefix shared by the three
configurations of sample i. Decode limits 300 / 200 / 128.

Entries are median (min–max) tok/s. Drafts are accepted/proposed, in sample order.

| Config | Prompt | Prefill | Decode | Drafts |
| --- | --- | ---: | ---: | --- |
| Mixed serial | code | 266.3 (265.2–266.6) | 121.3 (119.6–123.8) | 161/224, 156/213, 153/209 |
| Mixed pipelined | code | 249.2 (247.2–249.4) | 118.7 (117.9–121.8) | 155/229, 145/212, 148/222 |
| Production | code | 325.3 (315.8–326.9) | 107.9 (107.3–114.1) | 150/228, 147/219, 161/229 |
| Mixed serial | long | 2,299.6 (2,296.4–2,311.9) | 121.7 (117.4–121.9) | 112/174, 114/172, 112/150 |
| Mixed pipelined | long | 2,303.5 (2,302.4–2,306.6) | 119.2 (116.3–123.5) | 109/184, 110/179, 106/149 |
| Production | long | 2,360.6 (2,350.4–2,369.4) | 114.5 (110.3–123.1) | 111/177, 110/175, 109/143 |
| Mixed serial | 16K | 3,396.0 (3,393.0–3,411.6) | 132.0 (120.2–139.3) | 85/112, 78/127, 80/105 |
| Mixed pipelined | 16K | 3,396.2 (3,394.5–3,402.1) | **146.6** (138.8–150.8) | 79/112, 78/110, 80/98 |
| Production | 16K | 3,488.6 (3,465.8–3,492.8) | 125.9 (122.3–132.2) | 77/100, 72/100, 78/94 |

**Pipelined against serial mixed:**
- **Decode:** −2% on code, −2% on long, +11% at 16K. The code and long differences are inside the sample spread; the
  16K gain is not (every pipelined sample is above the serial median).
- **Prefill:** the same on the long prompts. On the 118-token prompt it is −6%, every sample (see Open).

**Against production:** mixed pipelined decodes +10% on code, +4% on long and +16% at 16K. Prefill is −23%, −2% and
−3%.

**Speculative-window hit rate** comes from vanilla's own line (`STRATA_DECODE_TIMING=1`). It is from a separate traced
session per configuration (`t-*-0`), not from the untraced table:

| | Code | Long | 16K |
| --- | --- | --- | --- |
| Mixed pipelined: on the path / speculative | 39/98 (40%) | 25/68 (37%) | 19/35 (54%) |
| Production: on the path / speculative | 38/93 (41%) | 26/66 (39%) | 25/42 (60%) |
| Mixed: fresh / speculative window, ms | 18.5 / 14.2 | 20.3 / 15.9 | 22.3 / 10.8 |
| Production: fresh / speculative window, ms | 18.8 / 17.8 | 20.9 / 19.2 | 22.3 / 18.8 |

The guesses are right as often as in production. A right guess saves more in mixed mode (4 to 11 ms) than in
production (1 to 3.5 ms). But about 60% of speculative windows are undone, and a fresh window costs as much as a serial
one. That is why the net gain is small except at 16K.

## Task 6: configs

- **`hetero/configs/mixed.example.json`:** now has `--pipeline-windows 2`, `STRATA_PIPELINE_AUTO=0` and
  `STRATA_MTP_STAGE0=0`. The comment says so.
  - With `STRATA_MTP_STAGE0=0` the draft layer stays on the host's last card, as in production. Without it, vanilla's
    default would move the draft layer to the host's first card.
  - One server session with the new file started both verifiers on both sides and ended cleanly (`example-mixed`).
- **`hetero/configs/README.md`:** updated, including the new drivers.

**Decision for the Human User or Chat AI:** on this evidence the pipelined config is a clear win only on long contexts.
The commit adds it as the handoff asked. Dropping `--pipeline-windows 2` from the mixed example is a one-line revert if
code and 5K turns weigh more.

## Safety and budget

- **Kernel log:** the same boot (`uptime` before and after), and no new kernel line between the snapshots
  (`kernel-delta-all.log` is empty).
- **Core dumps:** none new (`cores-before.log` = `cores-after.log`).
- **GPU time:** 563 s, every run sequential and under 2 minutes. The longest run was 48.3 s (`pipe-checks`).
  - Engine runs: 224.6 s (`.ai/runs/0077-measure/gpu-seconds.txt`: 4 baseline, 4 task-3, 4 task-4 runs).
  - Server sessions: 338.3 s (`.ai/runs/0077-server/gpu-seconds.txt`: checks, features, 9 measurements, 2 traced, the
    example).
- **Process handling:** no process was killed or signalled other than the server's SIGINT.

## Evidence and drivers

- **`.ai/runs/0077-measure/`:** engine runs (`*.json` with token IDs, `.err`, kernel deltas), kernel and core-dump
  snapshots, and the pipeline trace of the forced-miss run (`t4-pipe-fm.trace`). `prompts.json` is not committed: it is
  `mkprompts.py MODEL prompts.json`.
- **`.ai/runs/0077-server/`:** server sessions (results, console, server logs, filled configs).
- **`.ai/runs/0077-build/`:** build logs.
- **New in `hetero/tools/`:**
  - `run_engine.py`: one accounted engine run from a config.
  - `serve_session.py`: one accounted server session: checks, features, or measurement.
  - `cmp_tokens.py`: compares two runs token by token.

## Open

- **The short-prompt prefill in pipelined mode (−6%):** likely cause, not confirmed: the pipelined prompt read
  (`read_windows_pl`) queues K and K+1 on stage 0 back to back, and the front stage serialises them. Each front PlCommit
  waits for its own window before it answers. Queuing the commit behind a front window still in flight would need the
  runner and the link thread to share the verifier.
- **Doomed windows:** the front stage serves a doomed window in full. It is not told, so vanilla's doom-skip (empty
  plans) does not reach it.
- **`STRATA_SPLIT_TIMING`:** it does not time pipelined windows.
- **Not run:** `--pipeline-windows 1` is accepted but untested.
- **Still refused:** `--batch` with pipelined groups.
- **The front stage's GPU span** in task 3 (4.2 against 3.3 ms) is unexplained.
- **`.ai/DIVERGENCE.md`** needs Chat AI's update:
  - "Supported since: `--pipeline-windows` (report 0077)", and remove the refusal row;
  - the verify hooks (+4 / +6 lines);
  - the line counts.
