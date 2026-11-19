# How this branch differs from vanilla

Reference: vanilla Strata 0.2.15 (`c775728`). Kept current by Chat AI; checked with
`python3 mixed_split/tools/check_off_is_vanilla.py` (no GPU, no build).


The subtree and namespace were called `hetero/` and `strata::hetero` until 2026-11-19 (renamed to `mixed_split/` and
`strata::mixed_split`, the build option's name). Earlier handoffs and reports keep the old names. A re-pin to vanilla
0.2.16 (reports 0083, 0084) was taken back on 2026-11-19 and is archived as `archive/mixed-native-0.2.16`.
## With `STRATA_MIXED_SPLIT` off (the default): vanilla

Every change in a vanilla engine file sits inside `#if defined(STRATA_MIXED_SPLIT)`. With the option off, the six
changed files read exactly as vanilla after the preprocessor. The branch deletes no vanilla line: it adds 1,537 lines to
7 vanilla files. The only change outside a guard is the option itself in the top-level `CMakeLists.txt` (12 lines,
default `OFF`).

## With `STRATA_MIXED_SPLIT` on: the seam

| File | What it adds | Lines |
| --- | --- | ---: |
| `CMakeLists.txt` | The option; `add_subdirectory(mixed_split)`; the define on `strata_engine`; links `strata_mixed_split` | 12 |
| `include/strata/core/verify.hpp`, `src/core/verify.cpp` | `Verifier::set_front`: the first local stage forwards each window's run and commit to the front stage. `set_launch_mark`: events around the window graph's launch, for `STRATA_SPLIT_TIMING` only. With `--pipeline-windows`, `pl_launch` and `pl_commit_async` launch and commit the front stage's window of the same parity first | 39 |
| `include/strata/prefill/prefill.hpp`, `src/prefill/prefill.cpp` | `Prefill::set_remote_out` / `set_remote_in`: a prompt chunk's rows go to, or come from, the shared hand-off instead of a stage in the same process. The fill callback gets the next chunk's size and returns the buffer its rows are in (read-ahead, report 0078) | 69 |
| `include/strata/core/conversation_cache.hpp` | `ConversationCache::on_drop`: reports each parked conversation the cache lets go (not one taken for a restore) | 25 |
| `src/program/generate.cpp` | See below | 1,392 |
| `mixed_split/` | The seam library (shared region, front link, worker process), its CPU tests, the probe, run configs and tools | 5,084 |

What `generate.cpp` adds:
- **Options:** `--front-exe`, `--front-layers`, `--front-arg` on the host; `--front-stage` on the front stage.
- **Host:** `MixedSplitHost` starts the front stage (`WorkerProcess`), sets up the shared hand-off, and repeats session
  operations on it: Reset, Abandon, Snapshot, Restore, Park, Unpark, Drop, and Steer (each request's control-vector
  switch). Before each request's first prompt chunk it sends the request's image input as vanilla built it: the
  position table, the image rows and the picture keys (link message Images, in pieces under 16 MiB).
- **Front stage:** loads layers `[0, K)`, serves the front link instead of the serve protocol, and keeps its parts of
  checkpoints and parked conversations (`FrontSessionBook`). Their keys are the tokens, the control-vector switch and
  the pictures, as vanilla's cache tells states apart. With `--vision` it uploads the host's position table and points
  its prompt path at the rows it received; it loads the host's control vector for its own layers.
- **Layer ranges:** the host's first stage starts at `front_lb`, and the front stage's process ends at `front_le`. This
  applies wherever vanilla assumed the first stage starts at layer 0 or that one card holds every layer: trimmed
  weights, session sizing, the two whole-model graph captures, PLE wiring, the output head, and the expert profile.
- **Pipelined windows** (`--pipeline-windows`, report 0077): the front stage is part of the host's stage 0. It
  mirrors stage 0's two verifiers (one per window parity), snapshots its GDN state before each launch, and on a wrong
  guess restores it and replays the commit with the real count. One hook in the decode loop's undo; link protocol 3
  adds PlRun, PlCommit and PlUndo, with a hand-off buffer and ready word per parity. Vanilla's scheduling, guessing and
  undo logic is unchanged.
- **Asynchronous front window** (report 0077): the front stage answers a window call once its window is launched.
  Vanilla's `Verifier::run` runs to its end on a helper thread (`mixed_split/front_runner.hpp`), every later call waits for
  it, and the first R9700 waits on the ready word on its GPU.
- **Prompt chunks across the link** (report 0078, link protocol 4: PfStart, PfWait): while the first R9700 reads chunk c,
  the front stage reads chunk c+1 into a second row buffer (shared slot 5, 320 MiB). Vanilla's chunk plan is
  unchanged. Nothing is read ahead across runs or past a checkpoint position. `STRATA_MIXED_PF_AHEAD=0` turns it off.
- **Timing** also covers pipelined windows and every link call (`STRATA_SPLIT_TIMING`, logging only).
- **Layer placement:** `--layer-split auto` runs vanilla's planner over the host's own layers `[front_lb, 48)`, through
  a starting layer added to the planner (report 0075).
- **Timing** (`STRATA_SPLIT_TIMING`, logging only): GPU events for the front stage's window, the link round trips, the
  host's wait on the ready word and each local stage's window.
- **Front stage settings,** all vanilla options:
  - `--no-prefill-borrow` (no cache-slot loan outside the serve loop);
  - `--mmap-experts`, or the host's `--shared-expert-arena`;
  - lookup windows capped at the hand-off's rows.

## Refused in mixed mode (the engine says so at start)

The reasons are Chat AI's reading of the code, not measured.

| Option | Why not yet |
| --- | --- |
| `--split-skip-if-fits` | The host's cards run layers past the front stage's; a one-card fallback has no meaning there. |
| `--batch` / `--slots` | The front link mirrors one session; batch slots would need one front session each (design note in report 0075). |
| DFlash2 | It drafts from taps on every stage, including the front stage's layers. |
| `--peer-device`, the helper caches | Both put experts on other cards, which the front stage's process does not see. |
| Session ops Rollback, Truncate | Vanilla's serve loop never calls them; the front stage refuses them. |

## Supported in mixed mode since

- Control vectors (Chat AI `e7d6d857`; verified in report 0074).
- `--vision` (report 0074, `0bccc402`).
- `--pipeline-windows 2` (report 0077): exactly the serial loop's tokens, also with forced misses and for a seeded
  sampled request. `--pipeline-windows 1` is accepted but untested.
- `--layer-split auto` for the host's cards (report 0075). It picks K = 31 on Oberon, against the fixed 33. Vanilla's
  check for a too-small first stage does not cover the host's first stage; hosts with four or more cards are untested.

## Not carried

- The `civcode/strata-dev-private` hardening: withdrawn on 2026-11-18 (`.ai/HARDENING.md`); archived on
  `archive/hardened-0069`.

## Known differences in output

- Greedy output can differ from HIP-only at an early token. In reports 0068 and 0074 this was a near-tie: the
  CUDA-only build picks the same token as mixed.
- With a control vector loaded but switched off, mixed output is not bit-identical to mixed without a vector. Report
  0074 traced it to layer 14, the front stage's last layer, where a covered layer's FFN write happens in another kernel.
  The likely cause is a rounding difference; not confirmed bit by bit.

## Vanilla findings (upstream candidates, not caused by this branch)

- **HIP first copy stream:** on gfx1201 with ROCm 10.1, the first conversation copy stream and event take about
  148 MiB of VRAM. `conversation_copy_streams_test` allows 16 MiB and the restore guard `kRestoreFirstUseMib`
  assumes 64 MiB (reports 0070, 0071).
- **`ple_parity`:** it needs fixtures (`bench/micro/ple_in.bin`, `ple_out.bin`) captured with `ple_layer_xcheck.cpp`.
  Without them it returns 2 (a failure), not 77 (a skip) (reports 0071, 0072).
- **`cuda_spill_guard`:** it has no sm_89 baseline. Two `native_gu_swiglu_kernel` specializations use 8 bytes of
  stack each (report 0072).
