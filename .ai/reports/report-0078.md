# Report 0078 — wrong-guess notice, short-prompt cost, prompt chunks across the link

**Completed, with one rule exception the Human User should confirm (next section).** Tasks 3 to 6 are done.
- **Task 3, the wrong-guess notice:** measured, not added. The numbers show it cannot save time.
- **Task 4, the short-prompt cost:** measured. Its largest part is the CUDA card's own prompt compute, which cannot be
  cut inside the boundaries. Two candidate cuts were measured and are not adopted.
- **Task 5, prompt chunks across the link:** done and exact. Prefill +12.7% at 16K, +16.2% at 32K.
- **Task 6, final table:** mixed now reads the 16K prompt 11% faster than production and decodes faster on all three
  prompts. The 118-token prompt is still −22.5%.
- `mixed_split_tests` passes in all three build trees (with the mock worker). `check_off_is_vanilla.py` reports 6 of 6.
- GPU time: 767 s of the 2,400 s budget.

## Reboot during the turn (rule exception)

- **What happened:** the machine was shut down cleanly at 23:37 on 18 November, while this turn was paused (the agent's
  usage limit). It was booted again at 08:01 on 19 November.
  - The previous boot's journal ends with a normal `systemd-shutdown` (filesystems synced, processes sent SIGTERM).
  - Its kernel log after this turn's "before" snapshot has no GPU fault, panic or hang line
    (`kernel-delta-prevboot.log`, 64 lines).
  - This turn's last run before the pause ended at about 23:16, 20 minutes before the shutdown.
- **The rule:** the handoff says a reboot ends the handoff at once. I did not check uptime when I resumed. I found the
  reboot only at the final kernel-log check, after about 14 minutes of further GPU runs in the new boot.
- **Those runs:** no per-run kernel delta has a fault line, there is no new core dump, and every engine and server
  ended cleanly.
  - The current boot's log has two lines that match the fault pattern. Both are driver initialisation messages
    (`vmid_mask_gfxhub`) at 08:01, before the first run.
- **The Human User should confirm** that the shutdown was theirs. If it was not, the results after 08:01 need a second
  look.

## Commits

- `830f50f75`: task 3. `STRATA_SPLIT_TIMING` for pipelined windows and for every link call (logging only).
- `2fb7f403e`: task 5. Prompt chunks across the link: link protocol 4, the hook in `Prefill`, a CPU test,
  `mkprompts.py long32k`.
- This commit: `hetero/tools/tps_table.py`, the README, this report and the run evidence.

**Builds:**
- `build-0071-hip`, `build-0071-cuda` and both vision builds are rebuilt and stamped at `2fb7f403e`.
- `build-0078-cuda-mmq` is the same commit with `-DSTRATA_MMQ=ON`. It was used for the task 4 A/B only; no committed
  config names it.

## Vanilla files changed (every change inside `#if defined(STRATA_MIXED_SPLIT)`)

| File | Function | Why |
| --- | --- | --- |
| `include/strata/prefill/prefill.hpp` | `Prefill::set_remote_in` | The fill callback also gets the size of the run's next chunk, and returns the buffer its rows are in. The `remote_in_` member is gone. |
| `src/prefill/prefill.cpp` | `Prefill::run_impl`, the chunk loop | Passes `min(C, n - c0 - T)` and takes the returned buffer as `hand_in_`. The loop and the chunk plan are unchanged. |
| `src/program/generate.cpp` | `HeteroHost`, `HeteroFrontBackend`, the front stage's link serving | Timing (task 3), the read-ahead (task 5), the sixth shared slot. |
| `src/program/generate.cpp` | pipelined decode loop, at a wrong guess's verdict | `hetero.pl_doomed(...)`: logging only; does nothing without `STRATA_SPLIT_TIMING`. |
| `src/program/generate.cpp` | after the stages' `on_stage_chunk` setup | `hetero.pf_snapshot_due`: the first stage's own checkpoint rule (`done >= part_next[0]`). No chunk is read ahead past a position where the front stage must save its part of a checkpoint. |

Against `c9934ab58`: prefill.hpp +7 / −7, prefill.cpp +2 / −2, generate.cpp +306 / −21.

**Link operations added (protocol 4):**
- **PfStart** (`a` n, `b` pos0, `c` buffer, the tokens): the front stage starts reading the chunk into prompt row
  buffer 0 (slot 2) or 1 (the new slot 5), and answers once it is taken.
- **PfWait** (`c` buffer): answers once that chunk's rows are written, or with its error.
- **The worker refuses:**
  - a second PfStart while a chunk is in flight;
  - a PfWait for a buffer with no chunk in flight;
  - a PfStart while a window waits for its commit.
- Any other operation ends a chunk in flight that nobody waited for. The backend waits for it first, and the chunk's
  error goes to that operation.

**Test:** `test_prefill_ahead` covers the refusals, a chunk read ahead into buffer 1, a chunk out of step with the
session, a failure reported by the wait, and a chunk ended by the next operation.

## Task 3: pipelined timing and the wrong-guess notice

`STRATA_SPLIT_TIMING` now adds three lines per request:
- the front stage's GPU time per pipelined window, split into kept windows and windows a wrong guess undid, and how
  long each undo waited for the front window;
- on the host, the first R9700's GPU wait on each parity's ready word, and how often that wait held it (> 0.1 ms);
- every link call's round trip (count, average, sum), on both sides.

A wrong guess is matched against the front window's end through a 16-byte record in the parity's ready-word slot
(steady clock, the same in both processes).

Traced run `t3-traced-pipe` (mirror config, MTP, `STRATA_IQ_MT_MIN=1`):

| | Code (300 tokens) | Long (200 tokens) |
| --- | --- | --- |
| Front windows: kept / undone | 131 / 67 | 90 / 44 |
| Front GPU per window: kept / undone | 4.02 / 3.80 ms | 4.43 / 4.42 ms |
| Wrong guesses whose front window had ended at the verdict | **55 of 67**, 2.26 ms before it | **39 of 44**, 2.48 ms before it |
| ... still running at the verdict | 12, ran 1.14 ms on | 5, ran 1.42 ms on |
| An undo's wait for the front window | **0.000 ms** (still running at 0 of 67) | **0.000 ms** (0 of 44) |
| First R9700's wait on the ready word | 3.78 ms per window, > 0.1 ms in 195 of 196 | 4.32 ms, in 132 of 132 |
| Link round trips per window: PlRun / PlCommit / PlUndo | 0.030 / 0.205 / 0.013 ms | 0.026 / 0.053 / 0.011 ms |

**Judgement: a notice cannot save time, so the code is left as it is.**
- **The window is always launched by the verdict.** The front stage launches its window at PlRun, immediately before
  the host's stage 0 launches the same window. The verdict comes later, so there is never a front window that could
  be dropped unlaunched.
- **It has mostly ended already.** In 82 to 89% of wrong guesses the front window had ended 2.3 to 2.5 ms before the
  verdict.
- **The rest costs nothing on the critical path.** The host's undo never waited for the front stage. The host's own
  stage 0 runs the same doomed window, which waits for the front's rows, so it ends later.
- **Upper bound:** the front GPU time a notice could free is 12 × 1.1 ms per 300 tokens (0.6% of decode), and it is
  off the critical path.
- **The CUDA card holds the first R9700 back in nearly every window** (3.8 to 4.3 ms). That is the serial chain inside a
  window (layers 0 to 14 before layer 15), not an effect of wrong guesses.

## Task 4: the short-prompt cost

**Breakdown.** The prompts are 91-token versions of the code prompt (`c118a/b/c`), read batched; `STRATA_TRACE`,
`STRATA_PREFILL_TIMING` and `STRATA_SPLIT_TIMING` were on. "Warm" is the second and third request of an engine.

| Part (84 tokens batched, warm) | Mixed | Production |
| --- | ---: | ---: |
| Host setup, request to prompt start | 15 ms | 9 to 16 ms |
| Front stage's prompt pass (CUDA, layers 0 to 14), in series before the host | **127 ms** (GPU 126) | — |
| Host stages on their GPUs | 57 + 65 ms | 86 + 89 ms (48 layers) |
| Batched read in all | **252 ms** | **178 ms** |
| Checkpoint after the prompt | 17 ms host + **11.5 ms front** (in series) | 24 ms |
| Link calls: images / steer / reset | 0.03 / 0.01 / 1.6 ms | — |
| Pipelined prompt windows (6 tokens) | 23 ms | 25 ms |

- **The gap is the CUDA card's prompt compute:** 8.5 ms per layer on the RTX 4070 Ti SUPER, against 3.6 to 3.7 ms per
  R9700 layer. Without it the mixed read would be 15 × 3.6 + 122 ≈ 176 ms, like production.
- **Link calls cost about 15 ms per request,** 11.5 ms of that the front's checkpoint part.
- **First requests are slower in both configurations.** Mixed reads 203 tok/s on the first and 302 / 312 on the next.
  Production reads 248 on the first and 378 / 383 (one sample).
- **Why report 0076 saw 80 ms:** every server measurement starts a fresh server, so its code prompt is always a first
  request.

**Pipelined against serial (report 0077's −6%), two samples each:**

| Request | Serial: prefill tok/s, first token | Pipelined |
| --- | --- | --- |
| First (cold) | 217.0, 438 ms | 199.5, 471 ms |
| Second | 307.6, 310 ms | 308.2, 309 ms |
| Third | 312.1, 306 ms | 312.3, 305 ms |

The pipelined cost is a first-request cost only (+33 ms); warm requests are equal.

**Cuts considered:**

| Candidate | Measured | Decision |
| --- | --- | --- |
| Front stage built with `-DSTRATA_MMQ=ON` (vanilla option; vanilla's CUDA default is off). Three samples each, alternating. | Short prompts: cold 202.9 → 218.9 tok/s, warm 302.0 → 342.6 and 312.1 → 348.4 (first token −30 ms). Long 2,247.5 → 2,107.9 (**−6%**), 16K 3,893.1 → 3,354.6 (**−14%**). | Not adopted: it loses on long prompts. A decision for Chat AI and the Human User if short turns weigh more. |
| No Steer / Images message when unchanged | 0.04 ms per request | Not worth a change. |
| Front prompt pass answered at launch | The host needs the rows of its single chunk before it starts; nothing overlaps for a one-chunk prompt | Not applicable to short prompts; done for multi-chunk prompts (task 5). |
| Front checkpoint part (11.5 ms) overlapped with the host's own checkpoint save (17 ms) | Would save about 11 ms per checkpoint (~4% of a warm short prompt's first token) | **Not done.** It needs an asynchronous Snapshot plus a wait, so that vanilla's "skip a checkpoint that could not be saved" (#3350) still applies. Candidate for a later turn. |

**Before and after:** nothing was cut for one-chunk prompts, so the code prompt is unchanged. In the server
measurement it is 250.6 tok/s now against 249.2 in report 0077 (mixed pipelined).

## Task 5: prompt chunks across the link

**Design.**
- The host's chunk callback (`HeteroHost::prefill`) waits for the chunk's rows (PfWait). It then starts the run's next
  chunk into the other buffer (PfStart) and returns the current buffer.
- So the front stage reads chunk c+1 while the first R9700 reads chunk c.
- **Buffer reuse is safe:** the buffer a new chunk takes was read by the chunk before, and that chunk's upload finished
  when its stream was synchronized at the chunk's end.
- **Chunks:** the sequence and every chunk's content are vanilla's. Nothing is read ahead across runs, and nothing past
  a checkpoint position.
- **Ready words:** none. The handoff suggested a ready word per buffer, but a blocking PfWait does the same job.
  - It is bounded by the link's 180 s timeout, as the synchronous chunk was.
  - It carries the front stage's error to the host at the right chunk.
  - Its cost is one round trip per chunk (0.8 to 2.8 ms for PfStart with up to 8,192 tokens of payload).
- **Memory:** the second buffer adds 320 MiB of shared pinned host memory (8,192 rows).
- **Safety:** a chunk read ahead that the host never reads (a cancel) leaves the front's book ahead of the host. Every
  later position or key mismatch is then refused by `FrontSessionBook`, so it cannot corrupt silently.
- **Off switch:** `STRATA_MIXED_PF_AHEAD=0` restores one waited chunk at a time.

**Exactness:**
- First 50 greedy token IDs (`STRATA_IQ_MT_MIN=1`) with the read-ahead on and off: **identical** on 32K, 16K and long
  (`t5-exact-ahead0/1`).
- The 32K read took its mid-prompt checkpoint at 16,384, including the front stage's part, with the read-ahead
  stopped at that chunk.

**Prefill tok/s:** untraced, one engine per sample, alternating. `STRATA_MIXED_PF_AHEAD=0` in the same build stands in
for the previous commit (the same one chunk per request, waited for).

| Prompt | Off (previous behaviour) | On | Change |
| --- | --- | --- | ---: |
| long (4,924 tokens, one chunk) | 2,291.3 / 2,272.3 / 2,264.3, median **2,272.3** | 2,273.6 / 2,275.3 / 2,267.0, median **2,273.6** | 0% |
| 16K (16,024) | 3,470.1 / 3,462.3 / 3,454.7, median **3,462.3** | 3,901.2 / 3,901.2 / 3,901.6, median **3,901.2** | **+12.7%** |
| 32K (32,024) | 3,771.7 / 3,768.6 / 3,759.3, median **3,768.6** | 4,373.4 / 4,381.6 / 4,377.3, median **4,377.3** | **+16.2%** |

Time to first token: 16K 4,645 → 4,124 ms; 32K 8,516 → 7,336 ms. Decode is unchanged.

**`--prefill-pipe 700`** (report 0075's value; the read-ahead on; three samples against the "On" column above):

| Prompt | Without | R = 700 | Change |
| --- | ---: | ---: | ---: |
| long | 2,273.6 | 2,902.1 | **+27.6%** (the 5K prompt now splits into chunks of 3,328 and 1,792 that overlap) |
| 16K | 3,901.2 | 3,904.7 | 0% |
| 32K | 4,377.3 | 4,140.2 | **−5.4%** |

- **Decode:** a 16-token decode after the 16K prompt read 125 against 143 tok/s. One pair with 128 tokens (119 without,
  127 with) did not repeat that, so decode is unaffected.
- **Decision:** it now helps 5K prompts but not 16K, and it costs 5% at 32K. That is not a clear win, so it **stays off**
  in the config. It is a one-line choice if 5K turns matter most.

## Task 6: final table

Through the server (`serve_session.py … measure`), untraced, one fresh server per sample, alternating. The mixed mirror
is `mixed.example.json` as committed (read-ahead on); production is `hiponly.example.json`. Entries are median
(samples). "Prompt ms" is the server's prompt phase, the time to the first token.

| Prompt | Config | Prefill tok/s | Decode tok/s | Prompt ms | Drafts |
| --- | --- | --- | --- | --- | --- |
| Code (118) | Mixed | **250.6** (247.8 / 250.6 / 251.3) | **123.7** (123.7 / 120.2 / 126.6) | 470.8 | 155/229, 145/212, 151/216 |
| Code (118) | Production | **323.5** (324.4 / 322.5 / 323.5) | **109.8** (109.8 / 109.0 / 115.5) | 364.8 | 150/228, 147/218, 161/229 |
| Long (5,000) | Mixed | **2,315.8** (2,320.8 / 2,315.8 / 2,309.9) | **120.7** (115.9 / 120.7 / 137.7) | 2,159.1 | 109/184, 110/179, 106/149 |
| Long (5,000) | Production | **2,371.6** (2,376.5 / 2,371.6 / 2,356.6) | **112.5** (112.5 / 112.1 / 131.7) | 2,108.3 | 111/177, 110/175, 109/143 |
| 16K (16,102) | Mixed | **3,900.7** (3,901.4 / 3,900.7 / 3,886.3) | **146.5** (146.5 / 138.8 / 150.8) | 4,128.0 | 79/112, 78/110, 80/98 |
| 16K (16,102) | Production | **3,501.0** (3,504.3 / 3,501.0 / 3,491.9) | **125.8** (125.8 / 117.0 / 142.8) | 4,599.2 | 77/100, 72/100, 78/94 |

**Mixed against production:**
- **Prefill:** −22.5% on code, −2.4% on long, **+11.4% at 16K** (report 0077: −3%).
- **Decode:** +12.7%, +7.3% and +16.5%.

## Safety and budget

- **Kernel log:** no GPU fault line in any run's delta. The reboot is described at the top. The "before" snapshot is
  from the previous boot, so the turn's delta is in two parts: `kernel-delta-prevboot.log` (previous boot, after the
  snapshot, ending in the clean shutdown) and `kernel-after.log` (the current boot).
- **Core dumps:** none new (`cores-before.log` = `cores-after.log`).
- **GPU time:** 766.5 s. Every run was sequential and under 2 minutes; the longest was 26.5 s.
  - Engine runs: 588.3 s (`.ai/runs/0078-measure/gpu-seconds.txt`).
  - Server sessions: 178.2 s (`.ai/runs/0078-server/gpu-seconds.txt`).
  - The total includes one repeated run (`t5-pf-ahead0-2`, 25.8 s). My shell call timed out after the run itself had
    finished, and I ran it again so the sample order stayed alternating.
- **Process handling:** no process was killed or signalled other than the servers' SIGINT.

## Evidence and drivers

- **`.ai/runs/0078-measure/`:** engine runs (`*.json`, `.err`, kernel deltas), kernel and core-dump snapshots, and
  `mixed-cudammq.json` (the MMQ front variant), and `prompts.json`. That file is committed this time, because it also
  holds the hand-made 91-token prompts `c118a/b/c`; the rest is `mkprompts.py` output.
- **`.ai/runs/0078-server/`:** server sessions (traced `t4-trace-*`, the final `m-*`). Its `prompts.json` is a copy
  of the one above (not committed).
- **`.ai/runs/0078-build/`:** build logs.
- **New:** `hetero/tools/tps_table.py`.

## Open

- **Short prompts:** what is left is CUDA prompt compute. Options outside this turn's boundaries:
  - an MMQ front (short +10%, long −6 to −14%);
  - fewer front layers (`--front-layers`, which also moves decode work);
  - overlapping the front's checkpoint part (~11 ms per checkpoint).
- **5K prompts are one chunk,** so the read-ahead does not reach them. `--prefill-pipe 700` would (+28%) at a 5% cost
  at 32K.
- **First-request cost with `--pipeline-windows 2`:** +33 ms on a cold server's first short prompt; not investigated.
- **`.ai/DIVERGENCE.md` needs Chat AI's update:**
  - link protocol 4 (PfStart / PfWait) and the sixth shared slot;
  - the `set_remote_in` signature;
  - the read-ahead and the `pf_snapshot_due` hook;
  - the timing hook at a wrong guess's verdict;
  - the line counts.
