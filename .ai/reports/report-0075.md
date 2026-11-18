# Report 0075

**Completed.** All four parts are done:
- **A:** R found; both A/Bs run with three samples each.
- **B:** timing lines committed; breakdown for both prompts.
- **C:** `--layer-split auto` serves in mixed mode; both design notes written below.

`check_off_is_vanilla.py` reports 6 of 6. `mixed_split_tests` and `mixed_split_cache_drop_test` pass in both build
trees. GPU time: 674 s of the 1,800 s budget.

Commits:
- `dcade0588`: part A drivers and evidence.
- `27b9062e4`: timing and `--layer-split auto`.
- `94d11a834`: queues the timed wait from the launch mark.
- This report, the README and the run evidence.

The builds `build-0071-hip`, `build-0071-cuda` and both vision builds are rebuilt and stamped at the last source commit,
and `check_launch` passes for both example configs (`.ai/runs/0075-build/`).

**Upstream push during the turn:** Chat AI pushed `53daf644` and reverted it in `dbda9bfb` (net zero) while this turn
ran. Part A ran on `c929496c` sources. I rebased onto the revert before pushing.

## One line each

- **Prompt pipeline:**
  - Not worth turning on for HIP-only on this rig. R=700 gives +3.1% on the 4.9K prompt and −0.4% at 16K. Larger R
    values do not change the plan at all: today's chunk is already 8,192.
  - In mixed mode it costs 14% at 16K.
- **Vision and the control vector:**
  - Vision costs nothing measurable on text requests.
  - The control vector costs nothing either. On the coding prompt it decodes 2.7% *faster* (98.0 against 95.4 tok/s)
    because it changes the text and the drafts accepted.
- **What overlap could hide:** the CUDA front stage is 19% of a mixed decode window, and the window is fully serial. The
  drafter is another 9 to 10%. A pipeline bounded by the slowest stage would remove up to about 68%, but speculative
  dependencies make that an upper bound, not a forecast.

## Part A: HIP-only (`hiponly.example.json`, `build-0071-hip`)

Method:
- **Prompt pipeline:** the engine runs directly over the serve protocol (`hetero/tools/engine_drive.py`, which now
  takes request keys). Each read is a fresh prompt with its own `[read i]` tag (`mkprompts.py --fresh`), sent with
  `pipe=V ckpt=0`, exactly as `tools/calibrate.py` sends it. That is calibrate's prompt-chunk step, run alone, with the
  handoff's values.
- **Warm-up:** one warm read per engine start, not counted.
- **Prompts:** 16K = 16,029 tokens, long = 4,929 tokens.

### Sweep (one sample per value, one engine; prefill tok/s)

| `--prefill-pipe` | 16K | long | chunk plan |
| --- | ---: | ---: | --- |
| off | 3,473 (the first 16K read of the start, slow) | 2,802 | 8,192 |
| 1 | 3,828 | 2,797 | 8,192 (no change: the rule keeps today's chunks) |
| 700 | 3,806 | **2,893** | 3,328 / 1,792 |
| 2000 | 3,807 | 2,788 | 8,192 (`pipeline_chunk` finds no 3% gain) |
| 5000 | 3,686 (the driver paused the GPU 434 ms) | 2,779 | 8,192 |

Only R=700 changes the plan. The others read exactly as off, so their spread is noise. Best R: **700**, and also by
calibrate's combined rate.

### A/B, off against R=700 (three samples each, alternating, one engine per prompt)

| Prompt | off | R=700 | Change |
| --- | --- | --- | ---: |
| 16K (100-token reply) | 3,813.6 / 3,802.1 / 3,798.4, median **3,802** | 3,792.6 / 3,785.3 / 3,772.2, median **3,785** | −0.4% |
| long (200-token reply) | 2,768.3 / 2,765.5 / 2,762.8, median **2,766** | 2,848.6 / 2,863.8 / 2,850.9, median **2,851** | +3.1% (each pair faster) |

Calibrate would probably keep it off: its rule needs more than 3% in the median over its 2K + 8K reads, and every round
faster. Vanilla's +36% at 16K came from rigs whose default chunk splits the prompt badly. Here `--prefill auto` already
reads 16K in two 8,192-token chunks across the two stages.

### Three arms (through the server, one start per sample, alternating, untraced; medians of three)

| Arm | Code: prefill / decode tok/s | Code: drafts | Long: prefill / decode tok/s | Long: drafts |
| --- | --- | --- | --- | --- |
| as committed | 245.6 / **98.0** | 127/187 | 2,264.6 / **105.9** | 102/178 |
| no vision | 246.3 / 97.1 | 127/187 | 2,249.6 / 105.3 | 102/178 |
| no control vector | 247.3 / 95.4 | 122/186 | 2,288.9 / 106.1 | 107/187 |

- Decode samples, code prompt:
  - committed: 98.2 / 97.3 / 98.0;
  - no vision: 97.7 / 97.1 / 96.0;
  - no control vector: 95.4 / 95.7 / 93.5.
- **Vision:** the same text and drafts as the committed arm. The differences are noise. The 83-token prefill varies 170
  to 250 tok/s in every arm.
- **Control vector:** it changes the text. On the coding prompt the steered reply accepts more drafts (67.9% against
  65.6%), and that is its decode gain. On the long prompt there is no difference.
- **Start-up:** a vision start takes about 3.5 s longer (the vision tower).

## Part B: where a mixed decode window spends its time

### Timing added (behind `STRATA_SPLIT_TIMING`, logging only)

- **Front stage:** events on its verify stream just before and after its window graph launch, giving its GPU time. It
  also logs the wall time of its run and commit calls. It writes one line per request, at the next request's image
  input (the host's first call of a request), and one at the end of the link.
- **Host:**
  - the link round trips for run and commit;
  - every local stage's window graph on its own GPU (the same launch marks);
  - the first local stage's GPU wait on the front stage's ready word. Just before that stage's graph launch, the host
    queues a bounded `wait_handoff` on the word (the same wait the graph starts with), with an event on each side.
- **Host output:** one line per request, at the request's end next to vanilla's `STRATA_SPLIT_TIMING` lines, and one at
  READY for the warm-up windows.
- **Drafter and commit:** vanilla's `STRATA_DECODE_TIMING` line covers these (`draft`, `commit/emit`).

### Breakdown (mixed config, traced runs only, two server starts, the same numbers ±0.1 ms; ms per window)

| Part | Code (300 tokens, 168 windows, avg T 2.07) | Long (200 tokens, 93 windows, avg T 2.73) |
| --- | ---: | ---: |
| Window, total (`decode timing`) | **17.7** | **20.0** |
| Link run round trip | 3.36 / 3.45 | 3.90 / 3.88 |
| — of which the front stage's GPU window | 3.33 / 3.40 | 3.87 / 3.85 |
| Link commit round trip | 0.015 | 0.015 |
| First R9700: wait on the ready word (GPU) | **0.009** | **0.009** |
| First R9700 (layers 15–32): GPU window | 5.45 | 6.13 |
| Second R9700 (33–47 + head): GPU window | 5.68 | 6.43 |
| Drafter (MTP) | 1.58 | 2.00 |
| Commit / emit | 0.71 | 0.72 |
| Rest (launch, sync, staging gaps) | ~0.9 | ~0.8 |

**Correction to the handoff's premise:** the front stage does *not* answer the window call right after the launch.
Vanilla's `Verifier::run` synchronises its stream before it returns (an earlier stage hands off after a synced window).
So the link round trip contains the whole CUDA window, and the R9700 never waits on the ready word (9 µs, the cost of the
wait kernel itself). This explains the zeros of report 0073: nothing was waiting.

Report 0073's mixed per-stage split is now measured:
- The CUDA card spends 0.22 ms per layer.
- Each R9700 spends about 0.30 ms per layer, plus the head on the second.
- Shares of a code window: front 19%, the first R9700 31%, the second 32%, drafter 9%, commit 4%, gaps 5%.

An earlier traced pair (`pretraced-*`) ran on the first timing build, which queued the wait from the link call. Its
host-side numbers are the same; its wait was not recorded.

### Task 8: mixed split, R=700 against off, 16K prompt (untraced, one engine, alternating)

| off | R=700 |
| --- | --- |
| 3,522.5 / 3,531.2 / 3,517.1, median **3,523** | 3,024.1 / 3,033.3 / 3,014.3, median **3,024** (−14%) |

R=700 did not win on 16K in part A, so strictly this task's precondition was not met. I ran it because it is cheap
(65 s), and it is informative. In mixed mode each chunk first goes to the CUDA stage, serially. Smaller chunks
(3,328 instead of 8,192) add one more round trip per chunk and more serial front-stage passes, while the HIP stages
gain nothing. Mixed 16K with today's chunks reads at 3,523 tok/s against 3,802 for HIP-only.

## Part C

### `--layer-split auto` for the host (implemented)

The vanilla planner gets a starting layer `front_lb`. With the option off, it is the vanilla code; with it on but
without `--front-exe`, `front_lb` is 0 and nothing changes. Every change is guarded:
- **Room and sizes:** `stage_left` (session carve) and `chunk_bytes` (prompt buffers) take stage 0 from `front_lb`.
  These cover the room checks, the startability gate, the #2644 share pass and the diagnosis.
- **`predict`:** stage 0's range is `[front_lb, at[0])` for its room plus trim credit, and for its layer time.
- **`stage_max`** (the tie-break) and **`parts_of`** (the gate's participants) use the same range.
- **`consider`, and the four-way loop:** a placement needs at least `max(2, #1616 floor)` layers in the host's first
  stage, counted from `front_lb`.
- **Spill walk step 1** (the early proportional split for short cards) is skipped on a mixed host. It would split from
  layer 0.
- **Refusals:** `--layer-split auto` is no longer refused. `--split-skip-if-fits` is now refused in mixed mode: it would
  drop the split and run every host layer on one card.

The profile was already filtered to the host's layers (report 0068), so nothing else in the search needed a change.

**Check** (`arm-mixed-0-auto`, `mixed.example.json` with `"layer_split": "auto"`):
- **Plan:** `layer split auto: K=31 - predicted 31.8 ms per decode window; the caches hold 16896 of 16896 profiled
  pairs (~100.0% of the routed mass), best of 31 placements`. Layers 15–30 (CUDA0), 31–47 (CUDA1). The L2 probe says
  startable.
- **Served:** the code and the long requests. Prefill / decode tok/s: 204.5 / 101.8 for code and 2,258 / 107.8 for long
  (one sample).
- **Against the fixed 33** (15–32 / 33–47): every placement holds the whole profile on these cards. So the tie-break
  picks the most even stages: 16 / 17 layers against 18 / 15. Expert slots: 7,680 + 8,704 against 7,680 + 7,680.
- **Speed:** one untraced sample, the same within noise as report 0074's medians for 33 (2,257 / 107.9 on the long
  prompt). No A/B was run.
- **Not covered:** the #1616 "tiny first stage" detection (`has_tiny_stage`) still counts the host's first stage from
  layer 0, so it cannot fire for that stage. The `consider` floor applies the minimum only once that pass runs. Four or
  more host cards go through the guarded four-way loop, which is not tested.

### Design note: `--batch` / `--slots` in mixed mode

- **What the front stage needs:**
  - one `SessionState` per batch slot for layers `[0, K)`, as vanilla builds them for every stage (`bslot_ss`, 1 +
    stages);
  - `Verifier::init_slots` on its verifier;
  - per-slot copies of what `FrontSessionBook` keeps today (tokens, steer, pictures);
  - each slot's checkpoints and parked parts keyed per slot;
  - its share of the slot memory, sized by the same `fit` rule. The host must cap `--batch` at what both processes fit,
    so a new link reply carries the front stage's count.
- **Vanilla calls involved:**
  - `run_slot_rows(rows, S, tokens, pos, …)` / `run_slots`;
  - `commit_slots` / `commit_slot_prefixes(keep)`;
  - the admission path: BGEN reads a prompt into a slot as a `GEN 1`, so `Prefill::run` on the slot's session, then a
    copy into the slot;
  - slot drop / park / move between the solo session and a slot.

  The pipelined groups (`--batch-groups`, the default on a layer split) use `batch_launch[_rows]`, `batch_poll` and
  `batch_commit_rows`. That is the same overlap problem as `--pipeline-windows`.
- **Over the link, per window:**
  - run: S rows of (slot, token, position), a few hundred bytes;
  - commit: a keep count per slot, a few bytes;
  - the hand-off: `handoff_floats × rows × 4` bytes, about 51 KB per row. With at most 8 rows per window (4 slots with
    `--batch-mtp`) that is about 410 KB, which slot 0 of the shared region already holds (8 rows).
  - admission: as a normal prompt chunk.
- **Risks:**
  - the slot bookkeeping must mirror vanilla's exactly (rotation, idle slots kept, solo↔slot moves);
  - the conversation cache per slot;
  - the front stage's VRAM for N sessions on a 16 GB card at a 196K context: int8 KV for 15 layers per slot is the
    limit, so N is probably 2 to 3;
  - the serial serving in this report: a batch window still pays the front stage's GPU time serially. The gain is only
    the batching itself, not stage overlap, unless pipelined groups are done as well.
- **Estimate:**
  - serial groups (`--batch-groups 1`, `--batch-mtp` optional): **2 Tier 1 turns**, one for the protocol, front-stage
    slots and admission, one for park, drop, the cache and the GPU checks;
  - pipelined groups on top: **+2 Tier 1 turns**, shared with the pipelined-windows work.

### Design note: DFlash2 in mixed mode

- **What the front stage needs:**
  - **Tap point:** DFlash2 reads the mean of the hc streams after layers 5, 15, 25, 35, 45. Layer 5 is in the front
    stage's range. Whether layer 15 is too depends on how vanilla numbers "after layer 15" against a stage that starts
    at 15; the per-stage `set_taps` keeps each tap with the stage that owns that layer.
  - **Windows:** the front stage turns on `set_taps({5})` (`set_batch_taps` for batch slots) and writes each window's
    tap rows into a fourth slot of the shared region.
  - **Prompts:** its prompt path writes the tap rows of each chunk for the drafter's context (`Prefill` `tap_local_`).
  - **Host:** the drafter sits on the host. The host's `gather_taps` gains a source for the front stage's share: copy
    from mapped host memory to the landing buffer, instead of a `cudaMemcpyPeerAsync` from a stage in the same process.
- **Vanilla calls involved:** `Verifier::set_taps`, `tap_dev`, `gather_taps` / `gather_taps_host`, `tapping_chain` and
  `tap_total`. The `Dflash2Drafter` calls (`context`, `context_async`, `note_rows`, deferred rows) stay local. On the
  prompt path, `Prefill`'s tap rows for the context.
- **Over the link, per window:** one tap of `n_embd` = 2,560 floats per accepted row. That is 10 KB per row, up to
  80 KB per 8-row window, through shared memory with no socket payload. The ready word already orders it, because the
  window is synced. The prompt path adds 10 KB per prompt row (82 MB for an 8,192-token chunk), so the region grows by a
  tap slot, or the taps go chunk by chunk. That is about the size of the residual hand-off slot today (8,192 rows ×
  hc × n_embd floats).
- **Risks:**
  - tap layer ownership at the boundary (an off-by-one gives a wrong context, which is silent: the drafts just get
    worse);
  - context continuity across checkpoints and parks, since the drafter's ring holds committed rows only (as vanilla);
  - `--draft auto`, which keeps the unused drafter current (more taps per commit);
  - CUDA/HIP numerical differences in the tapped activations (low priority under contract item 5);
  - the drafter's VRAM on the host.
- **Estimate:** **2 Tier 1 turns**: one for taps across the seam (windows and prompts) with an acceptance-rate
  comparison against HIP-only, one for checkpoints, parks and `--draft auto`. Plus a short Tier 2 turn for an A/B of
  drafting quality.

## Vanilla files changed (all inside `#if defined(STRATA_MIXED_SPLIT)`)

- **`include/strata/core/verify.hpp`:** `Verifier::set_launch_mark` and its member. A hook around the window graph
  launch, so events can time a stage's window on its GPU.
- **`src/core/verify.cpp`, `Verifier::run`:** calls the hook just before and just after `cudaGraphLaunch` (the
  `!pre` path).
- **`src/program/generate.cpp`:**
  - The guarded mixed-split structs:
    - `HeteroSum` / `HeteroGpuSpan`: the timing helpers;
    - `HeteroHost::run` / `commit`: round trips;
    - `timing_report`, and `time_stage` (the launch marks plus the timed ready-word wait).
  - `main`:
    - the mixed refusal list (auto allowed, `--split-skip-if-fits` refused);
    - spill walk step 1 skipped on a mixed host;
    - the planner hooks listed in part C (`stage_left`, `predict`, `stage_max`, `chunk_bytes`, `parts_of`,
      `consider`, the four-way loop);
    - the `set_front` site (installs the timing);
    - the front stage's serve block (its timing, at `f_run`, `f_commit` and `f_images`, and at the link's end);
    - the host's READY and request-end reports.

## Other changes

- **`hetero/tools/engine_drive.py`:** request keys (`name:max_new:key=v+key=v`) and LP capture, folded in from report
  0074's run-local `drive.py`.
- **`hetero/tools/mkprompts.py`:** `code`, `long16k` (16,024 tokens of `docs/*.md`), and `--fresh N` tagged variants.
- **`hetero/configs/README.md`:** auto is supported; `STRATA_SPLIT_TIMING`.
- **`.ai/DIVERGENCE.md` still lists `--layer-split auto` as refused.** It needs Chat AI's update: "Supported since:
  `--layer-split auto` (report 0075)", the new `--split-skip-if-fits` refusal, and the verify hook (+10 lines in
  `verify.hpp` / `verify.cpp`).

## Safety and budget

- **GPU time:** 674 s (11.2 min), counted per engine or server lifetime:
  - prompt pipeline: sweep 53.4 s, A/B 16K 56.4 s, A/B long 41.2 s (cut short by a missing prompt; repeated) and
    45.1 s;
  - three arms: 9 starts, 255.7 s;
  - mixed traced: 63.0 s (first build) and 63.5 s;
  - mixed pipeline A/B: 64.5 s;
  - auto: 31.5 s.

  The longest single run was 64.5 s. Runs were sequential, and no process was killed or signalled other than SIGINT to
  the server.
- **Kernel log:** the same boot, and no fault, Xid, gfxhub or GCVM line. The delta has the known `Cannot map memory …
  0x7e9152 pages` line (the shared arena) twice, two amdgpu `svm_range_*` workqueue hog warnings, and two apparmor
  denials (snap firmware-notifier).
- **Core dumps:** none new.
- **Evidence:** `.ai/runs/0075-measure/` (per-run JSON, stderr and server logs, kernel deltas, drivers, configs) and
  `.ai/runs/0075-build/`. The 5.6 MB `prompts.json` is not committed; `PROMPTS.txt` says how to rebuild it.

## Open

- The window is fully serial. Overlap needs the front stage to return before its window ends, or a second window in
  flight. The 19% CUDA share is the target for pipelined windows. For prompts, the per-chunk round trip is the target
  for pipelining chunks across the edge (task 8's −14% shows the cost of more chunks without overlap).
- `--layer-split auto` in mixed mode: no speed A/B against 33 yet (one sample, no difference). The #1616 tiny-stage
  check does not cover the host's first stage. Four-card hosts are untested.
- `.ai/DIVERGENCE.md` update (above).
