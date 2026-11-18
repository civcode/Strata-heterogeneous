# Report 0069

Status: **Completed.** All four tasks are done:
- The experts are in RAM once (vanilla `--shared-expert-arena`).
- The front stage repeats every checkpoint and conversation-cache operation of the first stage, and both caches are on
  in mixed mode.
- The front stage no longer loads the native head.
- The run configs and drivers are committed.

Two findings for Chat AI:
- **Available RAM.** The shared arena gives one copy but no extra available RAM (below).
- **Turn 2 "same 50 tokens" check.** It holds for every checkpoint and park path. It does not hold for a turn resumed
  from the live session, and it does not hold in HIP-only (vanilla) either.

Builds: `build-0067-cuda` and `build-0067-hip`, rebuilt incrementally (`strata`, `hetero_tests`, `hetero_front_client`).
In the `STRATA_HETERO=OFF` tree (`build-0068-cuda-nohet`), `generate.cpp` compiles. `hetero_tests` passes in the host,
CUDA and HIP trees.

## Vanilla files changed

All the engine changes are in `src/program/generate.cpp`, function `main`, behind `STRATA_HETERO` and active only in a
mixed split (`hetero.on()` on the host, `front_le > 0` in the front stage).

| Where | Why |
| --- | --- |
| Mixed-split option block (host) | No longer turns the prompt checkpoints and the conversation cache off. `--prompt-cache` now reaches the front stage, so it can size its store. |
| Front-stage option block | `--mmap-experts` is forced only when no `--shared-expert-arena` is given. The store sizes come from `--prompt-cache` and `--conversation-cache-slots` before they are zeroed. |
| Native head load | Adds `&& front_le == 0`: the front stage never runs a head (task 3). |
| `HeteroHost::at` (new) | Sends a session operation with `(token_key(ids, n), n)`. |
| `checkpoint_at` | After the host's own saves, the front stage takes its part (Snapshot). Not on the mid-prompt path, which has its own row below. |
| `on_stage_chunk`, stage 0 | A split's mid-prompt checkpoint part. The front stage saves its part with the first local stage's. It is exactly at `done` then, because stage 0 asks it for each chunk before reading that chunk itself. |
| Checkpoint mount (`!from_live` branch) | Restore. `STRATA_CKPT_REREAD` also resets the front stage. |
| `park_source`, before `conversations.put` | Park, for the main session only. |
| Parked-conversation restore (`incoming`) | Unpark, after the later stages are restored. |
| Front backend | Run, commit, prefill and session operations keep the front stage's token history (below). Snapshot, Restore, Park and Unpark call `conversation_checkpoint_save`, `_restore`, `conversation_snapshot_save` and `_restore` (the `draft == nullptr` form a later stage uses) on the front stage's session. |

A failed front operation fails the way vanilla's own failure does at that place:
- checkpoint save: `ERR saving a conversation checkpoint failed: mixed split: …`;
- restore or unpark: `ERR … restoring …`, after which the engine ends, as vanilla does for a failed restore;
- park: `ERR`.

Nothing is skipped.

## Design decisions inside the boundaries

- **Keyed by token prefix, checked on every operation.** `hetero/include/strata/hetero/front_session.hpp` holds three
  things:
  - `token_key`: FNV-1a over the 32-bit token values and the length, so the host's int64 prompt buffer and its int32
    checkpoint ids give the same key.
  - `FrontSessionBook`: the front stage's token history, plus its parts and images in least-recently-used stores.
  - The front stage's checks:
    - It refuses a window or prompt chunk that does not start exactly at its history's end ("out of step").
    - It refuses a Snapshot or Park unless it holds exactly those tokens.
    - It refuses a Restore or Unpark of a key it does not keep.

  This catches a desync of the two sessions before it becomes a silently wrong continuation. None tripped in any run.
- **Park and Unpark added to `SessionOp` (7, 8).** The conversation cache parks one image per stage, including the
  stage's K/V, and a checkpoint (running state only) cannot stand in for that. Rollback and Truncate stay refused:
  vanilla's serve loop makes no such call on a stage session, because rollback is inside `Verifier::commit`, which is
  already forwarded. `hetero_tests` checks that both are refused.
- **The front stage's store.**
  - Parts: `prompt_cache × (slots + 1) + 2`, which is 32 with the defaults.
  - Images: `slots + 1`, which is 5.
  - A park copies the parts on its token path into the image, as vanilla's park copies the chain. An unpark puts them
    back.
  - Content keys mean an equal prefix overwrites; nothing is shared with the host's memory.
  - Over the server checks the front stage's anonymous RSS grew by 0.71 GiB, against 0.77 GiB on the host.
  - It is bounded by count, not bytes. With long parked conversations it can hold several GiB (open item).
- **Shared arena without new code.** `/dev/shm` holds 32.48 GB and the arena is 33.98 GB, so a tmpfs file does not
  fit. The drivers make a memfd, as `serve/replicas.py` does for replicas, and pass `/proc/<pid>/fd/<n>`. The host
  forwards the option to the front stage unchanged.
  - The HIP host's `hipHostRegister` refused the whole 34 GB memfd range ("invalid argument"). Vanilla's fallback then
    pinned all 48 per-layer slices (31 GiB).
  - CUDA registered the whole range in one call.
  - Prefill is unchanged.
- **Server launcher.** `hetero/tools/serve_mixed.py` fills the config's `${NAME}` placeholders and writes the filled
  config and the log into `RUN_DIR`. For a mixed config it adds the memfd arena. Then it `exec`s `serve/server.py`, so
  the server keeps the memfd and the pid in the path. No change to `serve/` was needed.

## RAM and VRAM

**RAM** with the engine idle at READY, both caches on (single samples, R8/R9; the server run R10 matches R9):

| | Front on `--mmap-experts` (as in 0068) | Shared arena |
| --- | --- | --- |
| Host process: anonymous / shared | 34.3 / 1.6 GiB | 2.6 / 33.2 GiB |
| Front process: anonymous / file / shared | 1.6 / 10.1 / 0.4 GiB | 1.4 / 0.3 / 32.0 GiB (the same pages) |
| System `Shmem` | 1.7 GiB | 33.3 GiB, one copy |
| `MemAvailable` | 19.0 GiB | 18.7 GiB (server: 18.5) |
| HIP-only, same build | 21.0 GiB (caches on), 23.0 GiB (caches off) | — |

- There is one copy of the experts now, but no extra headroom. The front stage's 10 GiB of experts were reclaimable
  page cache before.
- The 9.9 GiB of layers 0 to 14 still sit in the one arena. Only the fallback (an arena of the host's own layers) would
  free them; that needs a hook in the arena layout.
- The caches cost about 2 GiB at idle, in HIP-only as well (vanilla's checkpoint pool).

**VRAM**, the RTX 4070 Ti SUPER:

| Run | VRAM used |
| --- | --- |
| Report 0068 (mmap, with head) | 14.4 GiB |
| Now, mmap, no head | 13.7 GiB (14,009 MiB) |
| Now, shared arena | 14.9 GiB (15,215 MiB) |

- Without the head, the front stage's VRAM drops by about 0.7 GiB.
- In arena mode the CUDA engine leaves 763 MiB free against 1,969 MiB in mmap mode. The cache is the same 7680 slots,
  so this is vanilla's arena mode on that card, still above the 700 MiB reserve.
- With the arena, the front stage fills its cache faster: 6.5 against 4.2 GB/s.

R9700 cards while serving with MTP: 19.2 and 18.4 GB, as in report 0068. HIP-only server: 26.6 and 24.3 GB.

## Cache correctness (mixed, greedy, MTP off; the engine over the serve protocol)

R1, R5: both caches on. R2, R6, R7: both caches off, so nothing is reused. The A1 prompts are identical in every run.

| Turn 2 | Path with reuse | Reused | First 50 token IDs against the fresh read |
| --- | --- | --- | --- |
| A2ck5 (A1, another reply, a question asking for 5 sentences) | B parked A; A2ck5 unparks it and mounts A1's turn checkpoint (**Park, Unpark, Restore**) | 16 | **equal (50 of 50)** |
| A2ck (same, a shorter answer) | the same | 16 | equal (33 of 33; it stops at 33) |
| long2 (first 4,600 tokens of the long prompt, another question), `--prefill 2048 --prompt-cache-every 2048` | the checkpoint at 4,096, saved through the mid-prompt part path (**Snapshot in `on_stage_chunk`, Restore**) | 4,096 | **equal (50 of 50)** |
| A2live (A1, its answer as generated, a question) | the live session (no session operation at all) | 62 | differs at token 7 |

**A2live is not a cache defect.** HIP-only (vanilla, the same build, R3 against R4) differs on A2live too, at token 29,
while its A2ck matches. Reused from the live session, the 62 tokens were computed as a 23-token window read plus 39
one-token decode windows. Fresh, they are one 89-token prompt chunk, so the arithmetic is not the same. The live path
sends no session operation: the front stage only continued, and its position check passed. This compares like with
like only where the read geometry is equal, which is the case for every checkpoint and park path above.

Reused tokens, mixed beside HIP-only on the same conversation:

| Request | Mixed | HIP-only |
| --- | --- | --- |
| A2live (engine) | 62 | 62 |
| A2ck (engine) | 16 | 16 |
| Server two-turn chat, turn 2 | 59 | 59 (report 0067: 58) |

Turn-2 TTFT with reuse against without, mixed:

| Request | With reuse | Without |
| --- | --- | --- |
| A2live | 0.165 s | 0.341 s |
| long2 | 0.70 s | 1.90 s |
| A2ck | 0.327 s | 0.270 s |

A2ck is slower with reuse because it first parks the outgoing conversation. HIP-only shows the same: 0.330 against
0.283 s.

## Server checks (`hetero/tools/run_server_checks.sh` with `hetero/configs/mixed.example.json`)

Everything returned 200:
- health and the model list;
- streaming, non-streaming and sampled (temperature 0.8) requests;
- a streaming request cancelled after 8 chunks, then a normal request.

SIGTERM ended the server, the host and the front stage in 3 s: "the link ended cleanly", "the front stage ended
cleanly".

**Slow reader** (50 ms per chunk, 198 chunks, 10 s on the client) against a fast reader, the engine's decode rate per
segment:

| Pair | Fast reader | Slow reader |
| --- | --- | --- |
| First | 119.5 / 103.5 tok/s | 119.6 / 103.8 tok/s |
| Second | 119.5 / 103.8 tok/s | 119.6 / 103.9 tok/s |

The difference is under 0.3%. (The server sends each reply as two engine requests, because of the 64-token reasoning
budget.)

**HIP-only** (`hiponly.example.json`, the report 0067 configuration with vision and the control vector, same build):
- every check returned 200;
- turn 2 reused 59 tokens;
- the slow reader's rate was within 0.3% of the fast reader's;
- it shut down cleanly.

## Prefill and decode (single samples; the engine; MTP `--spec 4 --spec-min-p 0.5`; prompt auto; caches on)

| Request | Mixed now | Mixed 0068 | HIP-only now | HIP-only 0068 |
| --- | --- | --- | --- | --- |
| Long, 4,924 tokens, 200 generated (prefill / decode tok/s) | 2,249 / 113.0 | 2,342 / 116.3 | 2,236 / 111.4 | 2,229 / 111.2 |
| Short, 23 tokens, after long (decode tok/s) | 114.0 | 114.6 | 107.2 | 107.0 |

- The short prompt's prefill is now 112 tok/s (0068: 229), because the caches are on. Like HIP-only (132 tok/s), the
  request first parks the long conversation (about 115 ms).
- Through the server, per engine request on chat prompts (short segments, so the spread is wide): mixed decoded 84 to
  160 tok/s, HIP-only 77 to 133 tok/s.

## Safety

- **Before the first run:** 13 matching kernel lines (boot messages of today's 11:29 boot). The three newest core dumps
  are from `strata-monitor` on 17 November.
- **After the last run:** 18 new lines.
  - 16 are `svm_range_restore_work` / `amdgpu_amdkfd_restore_userptr_worker` / `svm_range_deferred_list_work [amdgpu]
    hogged CPU for >10000us`. All fall at the start or teardown of mixed engines, with and without the shared arena
    (R2 and R8 had them too).
  - 2 are `delayed_fput hogged CPU`, at teardown.
  - None fell while a server was idle or serving (R10 11:53:13–11:53:53), and none in any HIP-only run.
- No GPU fault, gfxhub, GCVM or page-fault line. No new core dump. No reboot. No GPU process was killed by force.

GPU run time: about 8.5 of the 30 minutes (17 starts, each under 70 s). Two of the starts failed on my arguments: one
engine run lacked `--spec`, and one server start lacked the placeholder variables. The second failed before any engine
started.

## Open

- **RAM headroom.** It needs the fallback (an arena of the host's own layers, 9.9 GiB) or a smaller arena. That is
  Chat AI's call: it is new code in the arena layout, not an option.
- **The front stage's store is bounded by count only.** Long parked conversations could use several GiB there. A byte
  budget would mirror vanilla's `--conversation-cache-mib` more closely.
- **The front stage's prompt chunk is still 2048**, and **the per-window edge cost** has not been measured (STATUS next
  steps).
- **The `hogged CPU` warnings** at mixed start and teardown (not at idle) remain unexplained. They never appear with
  HIP-only.
- **Not covered:** vision, control vectors, `--batch` (slot parking would need the slot-session forms of these
  operations), DFlash2, `--pipeline-windows`, `--layer-split auto`, session files (vanilla refuses them for a split).
- **Run files.** The untracked `runs69/` holds the R1–R13 results, logs and prompts.
- **Drivers committed in `hetero/tools/`:**
  - `engine_drive.py`, `mkprompts.py`, `hetero_common.py`;
  - `serve_mixed.py`, `server_checks.py`, `run_server_checks.sh`;
  - the configs and their README in `hetero/configs/`.
