# Handoff 0080

## Tier
1

DFlash2 in mixed mode, turn 1 of 2: the drafter's taps cross the CUDA/HIP seam. A wrong tap does not crash anything.
It only makes the drafts worse, so it is judged by acceptance rates on the cards.

## Objective
1. Go/no-go: does vanilla's DFlash2 drafter beat the MTP drafter on this rig, HIP-only, production mirror?
2. If it does: DFlash2 runs in mixed mode for decode windows and prompts, with the drafter on the host, and its drafts
   are as good as HIP-only's.

## Context
- Read `.ai/PROJECT.md`, `.ai/STATUS.md`, `.ai/DIVERGENCE.md`, and the DFlash2 design note in report 0075 (Part C).
  With `STRATA_MIXED_SPLIT` off, the engine sources must stay vanilla's line for line: run
  `hetero/tools/check_off_is_vanilla.py` before every commit that touches `src/` or `include/`.
- **The Human User chose `--batch` and DFlash2** (2026-11-19), DFlash2 first. Production runs one conversation at a time
  (`parallel` 1) with the MTP drafter. `--batch` follows in a later handoff.
- **Vanilla DFlash2** (`generate.cpp`, `--draft dflash2`):
  - opt-in, CUDA and HIP;
  - needs `--draft-model PACK` (`tools/dflash2_pack.py`), `--mtp DIR` (its `draft_vocab.bin`) and `--spec 2` or more;
  - the MTP layer is not loaded unless `--draft-fallback mtp`;
  - on a layer split the drafter sits on the last stage's card;
  - `--draft auto` (both drafters) is for one card only.
- **Vanilla's evidence is thin:** `docs/DGX_SPARK_LINUX.md` has DFlash2 slower than MTP there (37 against 45 tok/s).
  Nothing is measured on R9700.
- **Mixed mode refuses DFlash2 today** ("the DFlash2 drafter is not supported yet").
- **The design note (report 0075):**
  - DFlash2 reads the mean of the hc streams after layers 5, 15, 25, 35 and 45. Layer 5 is the front stage's.
  - Whether "after layer 15" belongs to the front stage (layers 0 to 14) or to the host's first stage depends on
    vanilla's numbering and on the per-stage `set_taps`.
  - Per accepted row, one tap is `n_embd` = 2,560 floats (10 KB). Prompts add 10 KB per prompt row.
  - The calls are `Verifier::set_taps`, `tap_dev`, `gather_taps` / `gather_taps_host`, `tapping_chain` and
    `tap_total`, and `Prefill`'s tap rows for the drafter's context. The `Dflash2Drafter` calls stay on the host.

## Part A: go/no-go (do first)
1. **Safety checks** (`.ai/PROJECT.md`): the boot ID, kernel log and core dumps, before the first GPU run, after the
   last, and after any pause.
2. **The drafter pack:** make or find one as vanilla documents (`tools/dflash2_pack.py`; `docs/DETAILS.md` names the
   tuned drafter and the Q4_0 selector). Keep it outside both checkouts, for example under `${STRATA_DATA}`. If it
   cannot be made without the Human User (a licence step, a large download that needs approval), stop and report.
3. **HIP-only, production mirror,** untraced, three samples each, alternating, on the code, the long and the 16K prompt:
   - the mirror as committed (MTP);
   - the same with `--draft dflash2 --draft-model PACK` (and whatever vanilla needs beside it).

   Report decode tok/s, drafts accepted, and whether vanilla keeps `--pipeline-windows 2` with DFlash2 or falls back
   to serial (and why).
4. **Decision rule:**
   - If DFlash2 is not faster than MTP on any of the three prompts by more than the spread of the samples, stop here:
     **Completed, no-go**, with the numbers. Do not implement part B.
   - Otherwise continue.

## Part B: taps across the seam (only after a go)
5. **Tap ownership:** find, from vanilla's code, which stage owns each tap layer in this split. Write it into the
   report before changing code.
6. **Windows:**
   - the front stage turns on the taps it owns (vanilla's `set_taps` for its layers) and writes each window's tap rows
     into a new slot of the shared region;
   - the host's `gather_taps` takes the front stage's share from that mapped memory, where vanilla would copy from a
     stage in the same process;
   - the drafter stays where vanilla puts it on the host (the last stage's card).
7. **Prompts:** the front stage writes each prompt chunk's tap rows for the drafter's context, chunk by chunk, with the
   read-ahead of report 0078 kept working.
8. **Refusals:** let `--draft dflash2` through in mixed mode. `--draft auto` stays refused, as vanilla refuses it on a
   layer split.

## Checks (part B)
- **Draft quality:** HIP-only DFlash2 and mixed DFlash2, the same three prompts, greedy, three samples each: drafts
  accepted per window. A clearly lower rate in mixed mode means a wrong tap. Find it; do not tune around it.
- **Exactness:** with `STRATA_IQ_MT_MIN=1`, greedy, mixed DFlash2 emits the same token IDs as mixed MTP on the code and
  the long prompt. If HIP-only does not keep that equality either, report both and compare mixed with HIP-only
  instead.
- **Speed:** mixed DFlash2 against mixed MTP (the mirror), untraced, three samples each, on the three prompts.
- **Through the server:** one streaming request, one cancelled request followed by a normal one, a vision request, and
  a clean shutdown.

## Boundaries you decide within
- As in handoff 0078. The smallest hooks in vanilla files, inside `#if defined(STRATA_MIXED_SPLIT)`. No change to
  vanilla's drafter, its context ring or its window rules: the front stage only supplies its share of the taps.
- Every safety gate stays:
  - the identity check;
  - publish after the front stage's commit acknowledgement;
  - the bounded waits, with `release_gpu_waits` on an error;
  - cooperative shutdown.
- **Left for turn 2 (do not start it):** checkpoints, parked conversations and the drafter's context across them.
  Note what you see break there.

## Hardware Test Budget
30 minutes of GPU run time (the project's allowance). Every single run under 2 minutes. Part A first; a no-go after
part A is a complete result.

## Constraints
- All rules of `.ai/PROJECT.md`. Run evidence goes under `.ai/runs/0080-*`; reusable drivers go into `hetero/tools/`.
  Never kill a GPU process by force. No two GPU jobs at once, except the two processes of one mixed run.
- Commit in small steps.

## Completion Criteria
- **No-go:** part A's table and the decision. That is **Completed**.
- **Go:**
  - part B implemented;
  - draft acceptance in mixed mode within the spread of HIP-only's;
  - the exactness check;
  - the speed table and the server checks;
  - `mixed_split_tests` (with a test for the tap slot if `hetero/` changes) and `check_off_is_vanilla.py` pass.
- `.ai/reports/report-0080.md` and all changes are committed and pushed to `feature/mixed-native`.
- If stopped early after a go: **Partial**, with the last step that works.

## Reporting
- Part A's table and the decision.
- After a go:
  - tap ownership;
  - every vanilla file changed, with the function and one line on why;
  - the acceptance, exactness and speed tables;
  - what you saw of checkpoints and parks for turn 2.
- Safety lines (boot ID, kernel log, core dumps), or "none". GPU run time used. What remains open.
