# Report 0074

**Completed.** Mixed mode now runs vision and the control vector. All vision checks pass. The control vector switches on
both sides and parks and reuses under the right key. Mixed and HIP-only diverge at token 0 with `cvec=1`. That
token is a near tie, and the CUDA-only build picks the same token as mixed. One further finding is below: with the
vector loaded but switched off, mixed is not bit-identical to mixed without the vector. Experiments trace this to
layer 14, the last front layer, where the FFN write is rounded on CUDA instead of HIP. It is not a switching error.

Commits: `0bccc402` (vision and picture keys), `8613f5176` (`mixed.example.json`, configs README), then this report and its evidence.

## Implementation

**Vanilla files changed:** only `src/program/generate.cpp`. Every change is inside `#if defined(STRATA_MIXED_SPLIT)`.
`check_off_is_vanilla.py` reports 6 of 6 files.
- `main`, the mixed-split option check: `--vision` is no longer refused on the host.
- `main`, the `--front-stage` option check: `--vision` is no longer refused on the front stage, so it allocates
  `d_mrope` as vanilla does.
- `main`, the serve loop's per-request setup, after vanilla's vision block (`sp.embd_rows = ...`): one
  `HeteroHost::images` call per request. The front stage's layers need the same positions and rows.
- `main`, the session-operation call sites (Park, Unpark, Restore, both Snapshots, the three Drops): each also passes
  the state's picture keys (`c.imgs`, `src.imgs`, `live.imgs`, `imgs_below(req_imgs, done)`).
- `main`, the front stage's serve block: `f_images` copies vanilla's `upload_mrope`. It puts the identity table back
  for a text request and points `sp.embd_rows` at the received rows. Its checkpoints and parked copies now carry the
  session's pictures. A prompt chunk past the image request's prompt is refused.
- `HeteroHost::images` / `at` / `drop` and `HeteroFrontBackend::f_images` are additions to the guarded mixed-split
  structs in the same file.

**Design decisions:**
- **Transport:** a new link operation, `Op::Images` (link protocol 2). The serialized `FrontImages` holds the prompt
  length, the spans (start, hash, rows), the whole position table and the rows in prompt order. It is sent in pieces
  of at most `kLinkMaxPayload`, and each piece is answered. The worker checks the layout before the backend sees it:
  spans inside the prompt, in order and not overlapping, `n_embd` floats per row, and a table if and only if there
  are images.
- **Per request:** the host sends this before any session operation of every request. A text request is one empty
  frame. The front stage tracks its own "identity" flag, so it never parses the embeddings file or computes positions.
  The spans come from vanilla's `req_imgs` and `row_ptr`, so the rows are exactly what vanilla's prompt path reads.
- **Keys:** `state_key = image_key(steer_key(token_key(ids, n), steer), imgs, n)`. `image_key` mixes in the
  (start, hash) of every picture below `n`. With no picture, text keys are unchanged. `FrontSessionBook` keeps the
  request's pictures and the session state's pictures separately, as vanilla keeps `req_imgs` and `live_imgs`. A
  prompt chunk or window over a session holding other pictures than the request is refused (a new safety gate).
- **Tests:** three new tests in `mixed_split_tests`: `test_image_key`, `test_images_link` (a ~20 MB input in two
  pieces, refused layouts, a window still open) and `test_session_images`. All pass, as does
  `mixed_split_cache_drop_test`, standalone and in both build trees.

## Builds

`build-0071-hip` and `build-0071-cuda` were rebuilt incrementally at `0bccc402`. The vision builds had no work to do,
since their sources are unchanged. All four were restamped at the config commit and pass `check_launch` (evidence in
`.ai/runs/0074-build/`).

## Control vectors

Runs drive the engine directly over the serve protocol, greedy, with MTP off (no `--mtp`). They use
`.ai/runs/0074-checks/drive.py`, which is `engine_drive.py` with request keys and LP capture added.
- **`cvec=1` against `cvec=0`:** the outputs differ in mixed, as in HIP-only.
- **Park and reuse** (`cvec-mixed`): A1 (`cvec=1`), then B parks A1. The second turn with `cvec=0` reused only 16
  tokens, from a `cvec=0` checkpoint of an earlier request, not A1. The same turn with `cvec=1` restored A1's 52 live
  tokens. The front stage logged no error or key mismatch.
- **Mixed against HIP-only with `cvec=1`, first 50 IDs:** they differ at position 0. Logprobs at position 0 for the
  23-token prompt:

| Engine | no vector | `cvec=0` | `cvec=1` |
| --- | --- | --- | --- |
| HIP-only | 760 −0.706 / 29108 −0.707 | identical bits | **29108** −0.632 / 760 −0.795 |
| CUDA-only (front build, one card) | 760 −0.649 / 29108 −0.773 | identical bits | **760** −0.567 / 29108 −0.873 |
| Mixed | 29108 −0.542 / 760 −0.907 | 29108 −0.574 / 760 −0.864 | **760** −0.688 / 29108 −0.726 |

  - It is a near tie: HIP and CUDA vanilla themselves disagree with `cvec=1`.
  - Mixed agrees with CUDA-only.
  - Forced first token, `cvec=1`:

    | First token | mixed vs HIP | mixed vs CUDA | HIP vs CUDA |
    | --- | --- | --- | --- |
    | 760 | 49/49 | 49/49 | 49/49 |
    | 29108 | 23 | 23 | 35 |

  - The long prompt with `cvec=1`: mixed and HIP-only agree for the first 27 tokens, as in report 0068 without a
    vector.
- **Finding, explained:** in vanilla, `cvec=0` gives the same bits as no vector at all, on both HIP and CUDA. In mixed
  it does not. Narrowing the vector's layer range:
  - range 15–44 (host only): `cvec=0` gives the same bits as no vector;
  - range 4–13: the same bits;
  - range 14–14: the full difference.

  The likely cause, from reading the code (not proven bit by bit): a steered layer applies its own FFN write in
  `cvec_apply` (`verify.cpp:2517`). Otherwise the next layer folds the pending write into its first read
  (`verify.cpp:1503`). At layer 14 that is CUDA's rounding (front stage) against HIP's (host). So this is a CUDA/HIP
  numerical difference at the seam (contract item 5), not a switch that fails to switch.

## Vision (through the server, `mixed.example.json`)

| Request | Answer | Cached tokens |
| --- | --- | ---: |
| shapes.png | red circle, blue square, green triangle, "STRATA 42" | 0 |
| code4217.png, same text | "CODE 4217", red square, blue circle | 0 |
| shapes with red and blue swapped (same size, so the same `<|image_pad|>` tokens), same text | **blue** circle, **red** square, "STRATA 42" | 0 |
| text after the images | correct refrigerator answer | 0 |
| shapes.png again | as the first answer | 318 (its checkpoint, not the swapped image's) |
| two turns about code4217: number, then the circle's colour | "4217", then "Blue" | 0, then 410 |

The host kept two parked conversations with identical tokens, apart by their pictures. The front stage refused
nothing and logged no error.

## Long-prompt speed (`*.example.json` through `serve_mixed.py`, untraced, alternating, one server start per sample)

The long prompt is 4,963 tokens with 200 generated (preceded by an 83-token code request, as in 0073):

| Sample | Mixed prefill / decode tok/s | HIP-only prefill / decode tok/s |
| --- | ---: | ---: |
| 1 | 2220.4 / 107.7 | 2313.5 / 105.4 |
| 2 | 2261.3 / 107.9 | 2258.3 / 105.4 |
| 3 | 2257.3 / 108.0 | 2307.5 / 105.4 |
| **Median** | **2257.3 / 107.9** | **2307.5 / 105.4** |

## Safety and budget

- **GPU run time:** 755 s (12.6 min), counting each engine or server lifetime. The longest single run was 37.8 s.
  Runs were sequential and no process was killed.
- **Kernel log:** same boot, and no fault, Xid, gfxhub or GCVM line. The delta has the known
  `Cannot map memory ... 0x7e9152 pages` line (the 31.6 GiB shared arena, as in report 0072) once per mixed start,
  one `delayed_fput` workqueue hog warning, and a USB keyboard and hub re-plug.
- **Core dumps:** none new.
- Evidence is in `.ai/runs/0074-checks/`: per-run JSON, stderr, kernel deltas, the server logs and the scripts.

## Open

- `.ai/DIVERGENCE.md` still lists vision and control vectors as refused and needs Chat AI's update; `hetero/configs/README.md`
  was updated here.
- These are still refused in mixed mode: `--layer-split auto`, `--batch` / `--slots`, DFlash2, `--pipeline-windows`
  above 0, `--peer-device` and the helper caches, and the session operations Rollback and Truncate.
- The driver changes (request keys and LP capture) live only in the run directory, because a change under
  `hetero/tools/` makes the builds stale. Folding them into `engine_drive.py` is left to Chat AI.
- The CUDA rounding of the layer-14 write with a vector loaded is a seam-level numerical difference. A bit-level
  confirmation would need a dump of the hand-off rows.
