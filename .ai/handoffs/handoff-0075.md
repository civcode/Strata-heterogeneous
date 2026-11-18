# Handoff 0075

## Tier
1

Mostly measurements, but part C changes the engine in mixed mode (automatic layer placement for the host), and part B
needs GPU-side timing that only works out on the cards.

## Objective
Settle four open questions in one turn:
- whether vanilla's prompt pipeline helps this rig;
- what vision and the control vector cost;
- where a mixed decode window spends its time;
- what the remaining mixed-mode refusals need.

Implement the smallest of those refusals, `--layer-split auto`.

## Context
- Read `.ai/PROJECT.md`, `.ai/STATUS.md` and `.ai/DIVERGENCE.md`. With `STRATA_MIXED_SPLIT` off, the engine sources
  must stay vanilla's line for line: run `hetero/tools/check_off_is_vanilla.py` before every commit that touches
  `src/` or `include/`.
- **Configs (revised by Chat AI after the handoff was written, before it started):**
  - `hetero/configs/hiponly.example.json` now mirrors the Human User's production setup:
    - `--mmap-experts`, `--max-context 262144`, `--kv-resident 0`;
    - `--pipeline-windows 2` with `STRATA_PIPELINE_AUTO=0` and `STRATA_MTP_STAGE0=0`;
    - `STRATA_SPLIT_OVERLAP=1`;
    - `--prompt-cache 6 --prompt-cache-spread`, `--conversation-cache-slots 4`,
      `--conversation-cache-min-tokens 8192`;
    - vision and the control vector as before.
  - `mixed.example.json` is its counterpart, without the three pipelining settings (mixed mode refuses them).
  - Reports 0067 to 0074 ran other settings (a RAM expert arena, 196,608 tokens, no overlap, default caches), so their
    numbers are not production numbers.
  - `serve_mixed.py` no longer adds `--shared-expert-arena` to a config with `--mmap-experts`: vanilla refuses the
    two together, and with the file mapping both processes share one copy in the page cache.
  - Production has no `--prefill-pipe`, and the report 0074 runs logged `prefill_pipe=0`: the prompt pipeline has
    never been on in this project.
- **Vanilla's prompt pipeline** (`--prefill-pipe R`, `src/prefill/prefill.cpp` near `pipeline_chunk`): on a layer
  split, stage s reads chunk c while stage s+1 reads chunk c−1. It is opt-in; `tools/calibrate.py CONFIG` measures R
  per PC, but a full run also measures every other setting and takes 15 to 30 minutes. Vanilla measured +36% / +18% /
  +4% for prompts of 16K / 32K / 64K tokens on two R9700, and a wrong R cost 19 to 36% elsewhere
  (`docs/MULTI_GPU.md`).
- **In mixed mode,** the host fetches each prompt chunk's rows from the front stage and waits for them before its first
  R9700 reads that chunk. So the CUDA card does not overlap with the AMD cards, and the planner sizes chunks for the
  host's own stages only.
- **Decode timing:** `STRATA_SPLIT_TIMING` showed zero stage times in report 0073. The front stage answers the host's
  window call after launching its window, and the host's GPU then waits on the shared ready word. So the wait is on
  the GPU, not on any CPU clock.
- **Remaining refusals** (`.ai/DIVERGENCE.md`): `--layer-split auto`, `--batch` / `--slots`, DFlash2,
  `--pipeline-windows`, `--peer-device` and the helper caches.

## Part A: the production mirrors, then measurements on HIP-only (do first)
1. **Kernel log and core dumps:** record them before the first GPU run and after the last.
2. **Rebuild** the HIP and CUDA trees at the current commit if `check_launch.py` refuses the builds of report 0074.
3a. **Bring up both production mirrors** through `serve_mixed.py`:
   - one request each;
   - the engine's start line (`INFO …`): `prefill_pipe`, `pipeline_windows`, overlap, `kv`;
   - VRAM per card, and available RAM with the engine idle;
   - in mixed mode, a conversation of at least 8,192 tokens over two turns, so that it is parked and reused under
     `--conversation-cache-min-tokens 8192`.

   If the CUDA card does not fit `--max-context 262144`, or `STRATA_SPLIT_OVERLAP=1` fails in mixed mode, report it
   and fall back for the rest of the turn: to 196,608 tokens, or without the overlap. A fault inside the engine stops
   the bring-up; report it, with Tier 1 escalation recommended.
3b. **Prompts:** the short coding prompt (about 300 tokens of reply), the long prompt (about 4,950 tokens, 200 of
   reply), and a new prompt of about 16,000 tokens (`hetero/tools/mkprompts.py`; 100 tokens of reply).
4. **Prompt pipeline, HIP-only:**
   - Find this rig's R: either the prompt-chunk part of `tools/calibrate.py` alone, if it can be run without the
     other settings, or a manual sweep of `--prefill-pipe` over off, 1 and three values of R spread over its range
     (for example 700, 2000, 5000), on the 16K and the long prompt.
   - Then A/B off against the best R: three samples each, alternating, on the 16K and the long prompt.
   - All on the production mirror (`hiponly.example.json`), changing only `--prefill-pipe`.
   - Write nothing into the vanilla checkout or the Human User's configs.
5. **Vision and control vector, HIP-only, untraced:** three arms:
   - the production mirror as committed;
   - without vision (no `"vision"` section and no `--vision`);
   - without the control vector (no `--control-vector*` / `--cvec-*`).

   Three samples each, alternating, on the coding and the long prompt: prefill and decode tok/s, drafts accepted.

## Part B: where a mixed decode window spends its time
6. **Add timing behind `STRATA_SPLIT_TIMING`,** logging only. In the front stage's backend, `HeteroHost` and, if
   needed, the first local stage's wait. Use CUDA / HIP events on the streams that do the work, so the GPU-side wait
   shows. Per request, report the average per decode window of:
   - the front stage's GPU time for its window;
   - the host's link round trips (run and commit);
   - the first R9700's wait on the front stage's ready word;
   - each R9700 stage's GPU time;
   - the drafter.

   Everything goes inside the `#if defined(STRATA_MIXED_SPLIT)` guards, or in `hetero/`.
7. **Run the mixed config traced,** with the coding and the long prompt, two samples each. Report the breakdown in ms.
   Untraced and traced runs are never compared with each other.
7b. **Mixed against production, untraced:** the two production mirrors, three samples each, alternating, on the
   coding, the long and the 16K prompt: prefill and decode tok/s. Mixed has no pipelined windows, so this is the gap
   pipelining would have to close.
8. **If part A found a winning R:** run the mixed config with it, untraced, three samples on the 16K prompt, against
   mixed without it. This shows what the HIP stages gain on their own while the CUDA stage stays serial.

## Part C: the remaining refusals
9. **Implement `--layer-split auto` for the host:** vanilla's automatic placement plans the host's cards for its own
   layers `[front_layers, 48)` instead of `[0, 48)`. Extend vanilla's planner with the smallest guarded hook (a
   starting layer); do not write a second planner.
   - Check: the mixed config with `"layer_split": "auto"` starts, the log names the plan, and one request is served.
     Compare the plan with the current fixed `33`.
10. **Design notes only, no code,** for `--batch` / `--slots` and DFlash2. For each:
    - what the front stage would need;
    - which vanilla calls carry it;
    - what crosses the link per window, and how big it is;
    - the risks;
    - how many agent turns it would take, with their tier.

    The Human User asked for this estimate. Write the notes into the report.

## Hardware Test Budget
30 minutes of GPU run time (the project's allowance). Every single run under 2 minutes. Priority order if time runs
out: tasks 3a, 4, 5, 7b, 7, 9, 8.

## Constraints
- All rules of `.ai/PROJECT.md`. Run evidence goes under `.ai/runs/0075-*`; reusable drivers may go into
  `hetero/tools/` (they do not make builds stale). Never kill a GPU process by force. No two GPU jobs at once, except
  the two processes of one mixed run.
- A new GPU fault line, a crash outside the run or a reboot ends the handoff at once.
- Commit in small steps.

## Completion Criteria
- Part A: R found, and the off-against-R A/B and the three-arm A/B with three samples each.
- Part B: the timing lines are committed, and the breakdown is reported for both prompts.
- Part C: `--layer-split auto` serves in mixed mode (or the reason it cannot), and both design notes are written.
- `check_off_is_vanilla.py` and `mixed_split_tests` pass.
- `.ai/reports/report-0075.md` and all changes are committed and pushed to `feature/mixed-native`.
- If stopped early: **Partial**, with the tasks reached, in the priority order above.

## Reporting
- **Tables:**
  - the production-mirror bring-up: VRAM, RAM, start lines, and the parked 8,192-token conversation;
  - the prompt-pipeline sweep and its A/B;
  - the three-arm A/B;
  - mixed against production;
  - the mixed per-window breakdown.
- **One line each:**
  - whether the prompt pipeline is worth it on this rig;
  - what vision and the control vector cost;
  - what share of a mixed decode window overlapping could hide.
- Every vanilla file changed, with the function and one line on why.
- The two design notes.
- Kernel log and core-dump lines, or "none". GPU run time used, counted per run. What remains open.
