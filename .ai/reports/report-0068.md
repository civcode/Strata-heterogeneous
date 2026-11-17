# Report 0068

Status: **Completed.** Milestone M2 is reached. A mixed request runs through vanilla's own generation loop, with layers
[0,15) on the RTX 4070 Ti SUPER (CUDA build, front stage), [15,33) on R9700 #0 and [33,48) on R9700 #1 (HIP build,
host). All five bring-up steps work, and both processes end cleanly in every step. `hetero_tests` passes in the CUDA,
HIP and host trees. With the mixed-split options absent, the report 0067 HIP-only configuration (vision, control vector,
`layer_split: auto`) still serves from the same HIP build and shuts down cleanly.

Builds: the existing `build-0067-cuda` (CUDA 13.0, sm_89) and `build-0067-hip` (both MMQ options) trees, both with
`STRATA_HETERO=ON`, rebuilt incrementally. A fresh tree with `STRATA_HETERO=OFF` and CUDA enabled (`build-0068-cuda-nohet`)
compiles `generate.cpp` without errors (only that object was built).

## What failed and what fixed it

| Step | Failure | Fix (commit) |
| --- | --- | --- |
| 1 | Chat AI's `33e5782c` fixed `fused_gr_read: invalid arguments`. Windows and commits then worked, but every prompt row came back NaN (`0x7fffffff`). | The front stage's prompt path borrowed expert-cache slots for its buffers. The serve loop takes and returns that loan around each prompt read (`host_res` marked missing, `relayout`), and the front stage never runs the serve loop, so the chunk was computed over experts that its own buffers overwrite. The front stage now sets vanilla's `--no-prefill-borrow` (`0bb4225b`). |
| 2 | The host stopped in the trimmed reload with `native PLE key is absent or incompatible`, then `the PLE run is not ready after construction`. | Layer 1, where the PLE block runs, is on the front stage. The host no longer wires or requires the PLE when `front_lb > 1` and clears the stale weight pointers (`4455e9a2`). |
| 2 | Both processes copied the full 31.64 GiB expert arena into RAM (2 × 31.6 GiB on a 60 GiB machine). RAM ran out, the amdgpu `svm_range_restore_work` warnings started, and the run stalled until the timeout. | The front stage reads its experts from the file mapping (vanilla `--mmap-experts`). Its expert cache holds all 7680 of its layers' pairs, so it reads the mapping only to fill the cache (`4455e9a2`). |
| 2 | (Prevention, no failure seen) The front stage captured 12-token lookup windows, larger than the 8-row shared hand-off. | Front stage: `--lookup-max-t` defaults to 0, as it does for a vanilla layer split, and `f_run` refuses T above the hand-off's rows (`4455e9a2`). |
| HIP-only | Engine exit code 1 on shutdown: `primary weight arena: invalid argument; exiting without releasing transfer owners`. | **Defect in the hardening port, not in the mixed-engine code.** The trimmed reload frees and reallocates the CUDA0 arena, but `PrimaryEarlyResources::weight_arena` kept the old pointer. The record now follows the reload (`ec8bb9c6`). The mixed host had not shown the error, probably because its new arena landed at the same address. |

## Vanilla files changed this turn

All changes are in `src/program/generate.cpp`, function `main`:
- Front-stage option block: sets `o.no_prefill_borrow`, `o.mmap_experts`, and `o.lookup_max_t = 0` (unless the user gave it). These are vanilla options set for the front stage only (reasons above).
- Trimmed CUDA0 reload: when `front_lb > 1`, `ss.ple.w = {}` and `wire_ple` is skipped. Also the hardening arena-record fix (applies to every build).
- PLE readiness check after `session_init`: skipped when `front_lb > 1`.
- Front backend `f_run`: also refuses `T > het_id.max_rows`.

`hetero/tools/front_client.cpp`: the failure message for non-finite rows now gives a count and the first raw words. A
new `--tokens a,b,c --dump PREFIX` option writes the given tokens' prompt rows and one-token window rows for offline
comparison. The `hetero/` interfaces are unchanged.

## Design decisions inside the boundaries

- The front stage uses vanilla options instead of new logic. It has no prompt-buffer loan (doing the loan would mean
  copying the serve loop's lend and refill code) and no RAM expert arena. The cost: its prompt chunk is 2048, because
  prompt auto is off without a loan. It runs each host chunk (up to 8192 rows) as several 2048-token pieces, writing
  them at their offsets in the shared slot.
- The PLE (layer 1) stays entirely in the front stage's session. Its Verifier and Prefill advance `ple_prev`
  themselves, `session_zero` resets it, and the host's `ss.ple` is not ready, so the host does no PLE read-ahead.

## `!multi_gpu` paths

- **Changed.** The session graph capture and the token graph: Chat AI's `33e5782c`, now verified on the cards. The
  `default_lookup_max_t` layer-split rule: applied to the front stage through the option block. The prompt-loan
  decision is not a `!multi_gpu` test, but it rests on the same "one card runs the serve loop" assumption.
- **Checked and left alone for the front stage:**
  - 5449: PCIe probe.
  - 5668 and 5834: HC Q8, environment-driven.
  - 6359: async commit. The next window runs on the same stream.
  - 7551: kv-grow, already guarded with `front_le == 0`.
  - 7920: MTP keep. The front stage has no MTP.
  - 8115: the native head. The front stage loads a 437 MB head it never uses; this is VRAM waste, left as is.
  - 8969, 9101, 9139: cache-auto and borrow decisions, overridden by `no_prefill_borrow`.
  - 11371, 11389, 11750, 11806–11809: CPU share. It is armed on the front stage and its numeric effect is small (rows
    below).
  - 11983, 12379.
- **Not reached in the front stage**, which does not run the serve loop: 14122, 18855, 18898, 19713, 19781, 20966,
  21039. On the host `multi_gpu` is true, so these keep their split behaviour.
- **Not used in serve mode:** 22036 onwards.
- **Host assumptions that the first stage starts at layer 0:** `front_lb` was already used for the weight reload,
  session carve, profile, prompt loan parts and window capture. The PLE was the one remaining case.

## Results

Engine driven directly over the serve protocol: the same token prompts and options for both configurations (MTP
`--spec 4 --spec-min-p 0.5`, `--kv int8`, prompt auto, no vision, no control vector). HIP-only uses `--layer-split auto`;
mixed uses `--layer-split 33 --front-layers 15`. Each figure is a single sample.

| Request | Mixed prefill / decode tok/s, drafts | HIP-only prefill / decode tok/s, drafts |
| --- | --- | --- |
| Long, 4924 tokens, 200 generated | 2342 / 116.3, 119/171 | 2229 / 111.2, 102/173 |
| Short, 23 tokens, 50 generated (after the long one) | 229 / 114.6, 28/48 | 129 / 107.0, 28/48 |
| short2, 23 tokens (stops at 20) | 236 / 88.8, 7/15 | 143 / 83.8, 7/12 |
| Greedy, MTP off, 50 tokens | 159 / 73.1 | 123 / 68.6 |

Report 0067's figures came from different prompts (decode 115 to 122, prefill 2831 at 6361 tokens), so they are not
directly comparable. Through `serve/server.py`, mixed decoded 120 to 141 tok/s on 63-token chat prompts; the HIP-only
run of the 0067 configuration decoded 96 to 127 tok/s.

Memory in use while serving:

| Card | Mixed | HIP-only (0067 configuration, with vision) |
| --- | --- | --- |
| RTX 4070 Ti SUPER | 14.4 GiB | — |
| R9700 #0 | 19.2 GB | 26.7 GB |
| R9700 #1 | 18.4 GB | 24.3 GB |

The shared hand-off region is 320 MiB (8 window rows, 8192 prompt rows).

Token IDs (finding):
- **Greedy, MTP off, first 50 tokens:** mixed differs from HIP-only at **position 0**. Mixed starts with 29108
  ("Sunlight …"); HIP-only and CUDA-only both start with 760 ("The sky …").
- CUDA-only and HIP-only serve agree on all 50 tokens.
- Evidence that position 0 is a near tie rather than a defect:
  - CUDA-only `generate` reads the prompt token by token and also picks 29108.
  - The front stage's prompt rows match that run's residual after layer 14 at every position (cosine 0.99967 to
    0.99998, `front_client --dump`).
  - Mixed with MTP equals mixed without MTP over all 50 tokens.
  - On the long prompt, mixed and HIP-only agree up to position 27.

Step 5: one streaming and one non-streaming request. A streaming request disconnected after 8 chunks (the server logged
`disconnect, cancel=True` at 10 tokens), and the next request was normal. A SIGTERM to the server (the server's
Ctrl+C path) ended the server, the host and the front stage cleanly: `the link ended cleanly`, `the front stage ended
cleanly`. Note for scripted runs: a server started with `&` from a non-interactive shell ignores SIGINT, so use SIGTERM.

## Safety

- **Before the first GPU run:** 304 matching kernel lines and 15 core records, all from `strata-monitor` before this
  handoff.
- **After the last run:** 17 new matching lines, all `workqueue: svm_range_restore_work / amdgpu_amdkfd_restore_userptr_worker
  / svm_range_deferred_list_work [amdgpu] hogged CPU for >10000us`:
  - 12 during the RAM-starved run (14:05–14:08).
  - The rest at engine start or teardown, one of them at the end of a HIP-only run (14:18:08).
- No GPU fault, page fault, gfxhub, GCVM or MES lines. No new core dumps. No reboot (boot still 10:42:15). No GPU
  process was killed by force.

GPU run time: about 13 minutes of the 60-minute budget, including about 3.5 minutes for the stalled RAM-starved run.

## Open

- The front stage's session is not snapshotted, restored or truncated. Mixed mode therefore keeps the prompt
  checkpoints and conversation cache off, and every request reads its prompt from token 0.
- The front stage has no prompt-buffer loan, so its prompt chunk is 2048. It also loads an unused native head (437 MB
  of VRAM).
- The host's RAM arena still holds the front stage's layers (about 9.9 GiB that are never read). RAM headroom with the
  server is tight: the workqueue warnings appeared at start and teardown.
- Not covered: vision, control vectors, `--batch`, DFlash2, `--pipeline-windows`, `--layer-split auto` for the host,
  and `STRATA_SPLIT_OVERLAP`.
- Local run files are in the untracked `runs68/` directory (ignored through `.git/info/exclude`): drive results,
  logs, the run configs `runs68/mixed.json` and `runs68/hiponly.json`, and the comparison dumps. The scripts used
  (`drive.py`, `client.py`, `mkprompts.py`, `cmp.py`) were in `/tmp/s68` and are not committed.
