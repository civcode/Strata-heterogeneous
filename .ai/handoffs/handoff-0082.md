# Handoff 0082

## Tier
2

A measurement turn, with no engine change. Scope the agent may change: run configs and the drivers in `hetero/tools/`.

## Objective
Settle whether the native Q2_0 pack costs decode speed, in HIP-only and in mixed mode, with enough samples to decide
a production switch.

## Context
- Read `.ai/PROJECT.md`, `.ai/STATUS.md` and reports 0080 and 0081.
- **The two packs:**
  - Production runs `packs/q2_0`, the pack vanilla's setup builds on an AVX-512 CPU: `tools/strata_pack.py`, about
    40 GB `experts.bin`.
  - The native pack (`${STRATA_DATA}/packs/q2_0-native`, `tools/iq_pack.py --base packs/q2_0`) keeps the experts as
    the GGUF stores them.
- **So far, three samples each:**

  | Mode | Prefill, code / long / 16K | Decode |
  | --- | --- | --- |
  | HIP-only (report 0080) | −11.5% / +16.5% / +13.0% | within the spread (16K median −6%) |
  | Mixed (report 0081) | +23.7% / +15.6% / +2.7% | medians 5 to 8% lower, all ranges overlapping |

  The decode question is open.
- **Configs:** `hiponly.example.json` / `hiponly-native.example.json` and `mixed.example.json` /
  `mixed-native.example.json`. They differ only in `--pack`.
- **RAM:** 60 GB cannot page-cache both packs, so the first start after a switch is slow (report 0080: one start
  missed the server's 65 s timeout).
- **The Human User:** the NVIDIA Xid 79 of 19 November is unrelated to this project.

## Method
- **Greedy decoding** (temperature 0) with fixed lengths: 300 / 200 / 128 tokens for the code / long / 16K prompt. Both
  packs then generate (nearly) the same text, so decode differences reflect the engine, not the content. Report
  whether the two packs' outputs are identical. If they are not, give the first differing position; that is a
  finding, not a failure.
- **Blocks, not single samples:** one fresh server per block. In each block, every prompt runs once as a warm-up, then
  twice measured. The warm-up's numbers are kept but not counted.
- **Order per mode:** A B B A A B, where A is the production pack and B the native pack. That gives three blocks, so
  six measured samples, per pack and prompt.
- **Before each start after a pack switch,** warm the page cache by reading the new pack's files and the GGUF shards
  (for example `cat … > /dev/null`, which costs no GPU time). Report the start times.
- Every block must stay under 2 minutes; split a block by prompt if it would not.

## Tasks
1. **Safety checks** (`.ai/PROJECT.md`): the boot ID, kernel log and core dumps, before the first GPU run, after the
   last, and after any pause.
2. **Builds:** use the current ones if `check_launch.py` accepts them; otherwise rebuild and stamp.
3. **HIP-only:** the six blocks above, with `hiponly.example.json` and `hiponly-native.example.json`.
4. **Mixed:** the six blocks above, with `mixed.example.json` and `mixed-native.example.json`.

## Hardware Test Budget
25 minutes of GPU run time (within the project's 30-minute allowance). Every single run under 2 minutes. Priority
order: task 4 (mixed, the open case), then task 3.

## Constraints
- All rules of `.ai/PROJECT.md`. Run evidence goes under `.ai/runs/0082-*`; reusable drivers go into `hetero/tools/`.
  Never kill a GPU process by force. No two GPU jobs at once, except the two processes of one mixed run.

## Completion Criteria
- Six measured samples per pack, prompt and mode.
- `.ai/reports/report-0082.md` and all changes are committed and pushed to `feature/mixed-native`.

## Reporting
- **Per mode,** a table per prompt:
  - decode tok/s: median, minimum, maximum, and the samples;
  - prefill tok/s;
  - drafts accepted per window.
- **One line per mode:** does the native pack change decode beyond the spread of the samples, and by how much?
- Whether the two packs' greedy outputs are identical. The start times after a switch.
- Safety lines (boot ID, kernel log, core dumps), or "none". GPU run time used. What remains open.
