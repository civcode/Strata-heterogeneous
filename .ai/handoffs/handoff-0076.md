# Handoff 0076

## Tier
2

A bring-up-and-measure turn on code that is complete. Scope the agent may change: build directories, run configs,
the drivers in `hetero/tools/`. A failure inside the engine or the front stage is reported, with Tier 1 escalation
recommended, not fixed here.

## Objective
Run both configs as the Human User runs production, and measure mixed against production. All earlier numbers ran
other settings.

## Context
- Read `.ai/PROJECT.md` (the production setup) and `.ai/STATUS.md`.
- **The example configs now mirror production** (Chat AI, after report 0075):
  - `hetero/configs/hiponly.example.json` is the production setup: `--mmap-experts`, `--max-context 262144`,
    `--kv-resident 0`, `--pipeline-windows 2` with `STRATA_PIPELINE_AUTO=0` and `STRATA_MTP_STAGE0=0`,
    `STRATA_SPLIT_OVERLAP=1`, `--prompt-cache 6 --prompt-cache-spread`, `--conversation-cache-slots 4`,
    `--conversation-cache-min-tokens 8192`, vision and the control vector, `layer_split: auto`, `parallel` 1.
  - `mixed.example.json` is its counterpart, without the three pipelining settings (mixed mode refuses them), and with
    `layer_split: auto` (report 0075).
  - `serve_mixed.py` adds no `--shared-expert-arena` to a config with `--mmap-experts`: vanilla refuses the two
    together, and both processes then read the experts through one file mapping.
- **Untested in mixed mode until now:** `--mmap-experts` on the host, 262,144 tokens on the CUDA card (it held
  14.4 GB of 16 at 196,608), `STRATA_SPLIT_OVERLAP=1`, and the production cache settings.
- **Report 0075:** a mixed decode window is fully serial. Of 17.7 ms, the CUDA stage takes 3.3 ms (19%) and the link
  adds 0.03 ms. Use the same prompts (`hetero/tools/mkprompts.py`: coding, long, 16K).

## Tasks
1. **Kernel log and core dumps:** record them before the first GPU run and after the last.
2. **Builds:** use those of report 0075 if `check_launch.py` accepts them; otherwise rebuild and stamp.
3. **Bring-up,** both mirrors through `serve_mixed.py`. For each:
   - one request;
   - the engine's start lines (`INFO …`) from both processes in mixed mode;
   - VRAM per card, and available RAM with the engine idle and after a 16K request (page cache included).

   In mixed mode, also check:
   - that the log shows `STRATA_SPLIT_OVERLAP` taking effect on the host's split, or what it does there;
   - a conversation of at least 8,192 tokens over two turns, so that it is parked and reused under
     `--conversation-cache-min-tokens 8192`;
   - a vision request and a request with `cvec=0`.

   **Fallbacks:** if the CUDA card does not fit 262,144 tokens, report the largest that fits and continue with it. If
   the overlap fails in mixed mode, report it and continue without it. A fault inside the engine stops the turn
   (Partial).
4. **Mixed against production,** untraced. The two mirrors, three samples each, alternating, on the coding, the long
   and the 16K prompt: prefill and decode tok/s, drafts accepted.
5. **`--layer-split auto` against the fixed split,** in mixed mode only, untraced: `auto` (K = 31 in report 0075)
   against `"33"`, three samples each, on the coding and the long prompt.

## Hardware Test Budget
20 minutes of GPU run time (within the project's 30-minute allowance). Every single run under 2 minutes. Priority
order: tasks 3, 4, 5.

## Constraints
- All rules of `.ai/PROJECT.md`. Run evidence goes under `.ai/runs/0076-*`. Never kill a GPU process by force. No two
  GPU jobs at once, except the two processes of one mixed run.
- A new GPU fault line, a crash outside the run or a reboot ends the handoff at once.

## Completion Criteria
- Both mirrors serve (with a documented fallback if needed).
- The parked 8,192-token conversation is reused in mixed mode.
- The A/B in task 4 has three samples per config and prompt.
- `.ai/reports/report-0076.md` and all changes are committed and pushed to `feature/mixed-native`.

## Reporting
- The bring-up table: start lines, VRAM per card, RAM, overlap, and the 8,192-token reuse.
- Mixed against production per prompt (median and spread). Task 5's table.
- Kernel log and core-dump lines, or "none". GPU run time used, counted per run. What remains open.
