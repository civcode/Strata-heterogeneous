# Handoff 0081

## Tier
1

`--batch` in mixed mode, turn 1 of 2: the front stage serves several batch slots. It is judged by exact token IDs per
slot and aggregate throughput on the cards. Starts with a go/no-go, as handoff 0080 did.

## Objective
1. **Go/no-go:** is `--batch` worth having on this rig, and can the CUDA card hold two sessions?
2. **If it is:** `--batch N --batch-groups 1` (serial groups) works in mixed mode, each slot exactly as its own solo run.
3. **Beside it** (measurement only): the native pack in mixed mode.

## Context
- Read `.ai/PROJECT.md`, `.ai/STATUS.md`, `.ai/DIVERGENCE.md`, the `--batch` design note in report 0075 (Part C) and
  report 0080. With `STRATA_MIXED_SPLIT` off, the engine sources must stay vanilla's line for line: run
  `hetero/tools/check_off_is_vanilla.py` before every commit that touches `src/` or `include/`.
- **Report 0080: DFlash2 is a no-go** on this rig (decode −5.5 / −6.9 / −16.7% against MTP on HIP-only). The Human
  User's `--batch` choice comes next.
- **Production serves one conversation at a time** (`parallel` 1). `--batch` (`--slots`, the server's `parallel`) serves
  several clients at once from one engine. On a layer split, vanilla's default is pipelined groups
  (`--batch-groups`); `--batch-groups 1` serves the slots serially.
- **The design note (report 0075):**
  - the front stage needs one `SessionState` per slot for layers `[0, K)` (vanilla's `bslot_ss`),
    `Verifier::init_slots`, per-slot copies of `FrontSessionBook`, and per-slot checkpoints and parks;
  - the calls are `run_slot_rows` / `run_slots`, `commit_slots` / `commit_slot_prefixes`, the admission (BGEN reads a
    prompt into a slot), and slot drop, park and moves between the solo session and a slot;
  - per window, S rows of (slot, token, position) and a keep count per slot cross the link; the hand-off rows fit slot
    0 of the shared region;
  - the risk is the CUDA card's VRAM: int8 KV for 15 layers per slot.
- **The native pack** (report 0080): HIP-only with `tools/iq_pack.py --base packs/q2_0` prefills 16.5% faster on the
  long prompt and 13% at 16K, with decode within the spread (`hiponly-native.example.json`). Not measured in mixed
  mode. Production is unchanged; switching is the Human User's decision.

## Part A: go/no-go (do first)
1. **Safety checks** (`.ai/PROJECT.md`): the boot ID, kernel log and core dumps, before the first GPU run, after the
   last, and after any pause.
2. **HIP-only throughput,** production mirror, untraced:
   - server `parallel` 1;
   - `parallel` 2 with vanilla's default groups;
   - `parallel` 2 with `--batch-groups 1`.

   Load: two clients at once, each sending the code prompt and then the long prompt, three rounds. Report aggregate
   decode tok/s, per-request decode tok/s and time to first token.
3. **CUDA card room:** find, from the front stage's own sizing and logs (no new code), the memory one more session for
   layers 0 to 14 needs at 262,144 tokens with int8 KV. Then say whether two sessions fit in the 16 GB card beside
   what it holds now (14.3 GB), and, if not, at which `--max-context` they would.
4. **Decision rule:** a go needs both of these:
   - `parallel` 2 with `--batch-groups 1` gives at least +20% aggregate decode over `parallel` 1 on HIP-only;
   - the CUDA card fits two sessions at a context of at least 131,072 tokens.

   Otherwise stop: **Completed, no-go**, with the numbers. Part C still runs.

## Part B: serial groups in mixed mode (only after a go)
5. **Implement the first turn of the design note:**
   - the link carries slot rows and per-slot commits;
   - the front stage builds a session per slot and serves `run_slot_rows`, `commit_slots` and the admission into a
     slot, with a per-slot book;
   - the host caps `--batch` at what both processes fit: a new link reply carries the front stage's slot count;
   - mixed mode lets `--batch N` through with `--batch-groups 1`; pipelined groups stay refused;
   - parked conversations, slot moves and the conversation cache per slot are turn 2. Keep them refused or out of
     reach, and say how.
6. **Checks:**
   - **Exactness:** with `STRATA_IQ_MT_MIN=1`, greedy, two clients at once (code and long), each slot's token IDs equal
     the same request run solo in mixed mode.
   - **Throughput:** mixed `parallel` 2 against mixed `parallel` 1 and against HIP-only `parallel` 2 with
     `--batch-groups 1`, the load of task 2.
   - **Through the server:** a cancelled request in one slot while the other continues; a vision request in a slot; a
     clean shutdown.

## Part C: the native pack in mixed mode (measurement only)
7. The mixed mirror with the native pack instead of the production pack (both processes; the CUDA front stage must
   accept it), against the mirror as committed. Untraced, three samples each, alternating, on the code, the long and
   the 16K prompt: prefill and decode tok/s. If the CUDA build refuses the native pack, report that.

   Mind RAM: report 0080 found that 60 GB cannot cache both packs, so the first start after switching may be slow.
   Count it, and do not let it skew the medians.

## Boundaries you decide within
- As in handoff 0078. The smallest hooks in vanilla files, inside `#if defined(STRATA_MIXED_SPLIT)`. No change to
  vanilla's slot scheduling, admission or commit rules: the front stage repeats per slot what the host's stage 0 does.
- Every safety gate stays:
  - the identity check;
  - publish after the front stage's commit acknowledgement, per slot;
  - the bounded waits, with `release_gpu_waits` on an error;
  - cooperative shutdown.

## Hardware Test Budget
30 minutes of GPU run time (the project's allowance). Every single run under 2 minutes. Priority order: part A,
part C, part B. A no-go after part A plus part C is a complete result.

## Constraints
- All rules of `.ai/PROJECT.md`. Run evidence goes under `.ai/runs/0081-*`; reusable drivers go into `hetero/tools/`.
  Never kill a GPU process by force. No two GPU jobs at once, except the two processes of one mixed run.
- Commit in small steps.

## Completion Criteria
- **No-go:** the tables of part A and part C, and the decision. That is **Completed**.
- **Go:**
  - part B implemented;
  - exact token IDs per slot;
  - the throughput table and the server checks;
  - `mixed_split_tests` (with tests for the new link operations) and `check_off_is_vanilla.py` pass.
- `.ai/reports/report-0081.md` and all changes are committed and pushed to `feature/mixed-native`.
- If stopped early after a go: **Partial**, with the last step that works.

## Reporting
- Part A's tables (throughput, CUDA room) and the decision. Part C's table.
- After a go: every vanilla file changed, with the function and one line on why; the link operations; the tables;
  what remains for turn 2.
- Safety lines (boot ID, kernel log, core dumps), or "none". GPU run time used. What remains open.
