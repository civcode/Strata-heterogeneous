# Handoff 0074

## Tier
1

Vision in mixed mode moves a request's image data and image positions from the host to the front stage's process.
It touches vanilla's per-request image setup, the front link and the front stage's keys, and is judged only on the
cards. A wrong step shows up as a wrong description of an image, not as a crash.

## Objective
1. Verify control vectors in mixed mode, which Chat AI implemented (`e7d6d857`, unverified).
2. Support `--vision` in mixed mode.

At the end, the mixed config can run what the HIP-only config runs: vision and the control vector.

## Context
- Read `.ai/PROJECT.md`, `.ai/STATUS.md` and `.ai/DIVERGENCE.md`. With `STRATA_MIXED_SPLIT` off, the engine sources
  must stay vanilla's line for line: run `hetero/tools/check_off_is_vanilla.py` before every commit that touches
  `src/` or `include/`.
- **Control vectors** (Chat AI, `e7d6d857`):
  - The front stage gets the host's `--control-vector*` options, so it loads the vector.
  - `SessionOp::Steer` repeats each request's switch (`cvec=0|1`).
  - The front stage's keys carry the switch (`steer_key`), and the host passes `cvec_cached` /
    `SavedConversation::cvec` with every keyed session operation.
  - CPU-tested and syntax-checked only.
- **How vanilla does vision**, all in `generate.cpp`:
  - The server's vision helper writes the image embeddings to a file; the request names it.
  - For each request the engine builds `mrope_host`, the (t, h, w) position of every cell. Every stage's rope kernels
    read it, from its own device copy (`d_mrope`, `st->mrope`).
  - It builds `row_ptr`, which points each `<|image_pad|>` token at its embedding row; the prompt path reads them
    through `sp.embd_rows`.
  - A text request after an image request resets the table to the identity (`mrope_identity`).
  - Image keys (`req_imgs`: position and hash) are part of every checkpoint's and parked conversation's identity.
- **In mixed mode:** layer 0 and the token embedding are the front stage's, and its layers read positions too. Both the
  image rows and the position table must reach the front stage's process.
- **The front stage's keys:** today they cover tokens (and the switch). Two different images have the same
  `<|image_pad|>` tokens, so the keys must also carry the image keys below `n`, as vanilla's cache compares them.
  Otherwise a checkpoint of one image could be restored for another.

## Design guidance (decide within it)
- **The front stage computes nothing that vanilla computes on the host.** The host sends it, before the prompt's first
  chunk:
  - the position table for every cell, or an "identity" mark for a text request;
  - the image rows in prompt order (the k-th `<|image_pad|>` token reads row k);
  - the request's image keys.

  Do not re-parse the embeddings file in the front stage (that would duplicate vanilla's reader). Anything that
  exceeds the link's 16 MiB frame (`kLinkMaxPayload`) is sent in pieces, or through a slot of the shared region.
  Images can reach about 10 MiB each at 1,024 rows.
- **The front stage** uploads the table to its `d_mrope` and points its prompt path's `embd_rows` at the rows it
  received. Both processes must start with the vision tables allocated: today the front stage refuses `--vision`, and
  `d_mrope` exists only with it.
- **Keys:** extend `token_key` / `steer_key` with the image keys (position and hash), on the host and in
  `FrontSessionBook`, keeping text-only keys unchanged. Add tests to `mixed_split_tests`.
- **Boundaries:** as in handoffs 0068 and 0069. The smallest hooks in vanilla files, all inside
  `#if defined(STRATA_MIXED_SPLIT)`; no new caching or scheduling logic; no weakening of the safety gates.

## Tasks
1. **Kernel log and core dumps:** record them before the first GPU run and after the last.
2. **Rebuild** the HIP and CUDA trees and the vision helpers at the current commit, and stamp them.
3. **Control vectors,** with a mixed config that adds the HIP-only config's control-vector options:
   - greedy decoding, MTP off, the same prompt with `cvec=1` and `cvec=0`: the two outputs differ, as they do in
     HIP-only;
   - mixed with `cvec=1` against HIP-only with `cvec=1`: the first 50 token IDs. A difference is checked as in report
     0068: a near-tie, or a bug;
   - a conversation parked with `cvec=1`, then a turn with `cvec=0` (no reuse, no error), then `cvec=1` again
     (reuse). The front stage's log shows no key mismatch.
4. **Vision:** implement it as above. Then, through the server:
   - one image request: the description is right;
   - two different images with the same prompt text, one after the other: each description fits its own image, so no
     state is reused across images;
   - a text request after an image request is unaffected;
   - a two-turn conversation about one image reuses its prefix.
5. **Configs:** make `mixed.example.json` the mixed counterpart of `hiponly.example.json`, with vision and the control
   vector, and passing `check_launch.py`. Run the long-prompt request through both for prefill and decode tok/s,
   untraced, three samples each.

## Hardware Test Budget
30 minutes of GPU run time (the project's allowance). Every single run under 2 minutes.

## Constraints
- All rules of `.ai/PROJECT.md`. Run evidence goes under `.ai/runs/0074-*`. Never kill a GPU process by force. No two
  GPU jobs at once, except the two processes of one mixed run.
- A new GPU fault line, a crash outside the run or a reboot ends the handoff at once.
- Commit in small steps.

## Completion Criteria
- The control-vector checks in task 3 pass, or each difference is explained.
- The vision checks in task 4 pass.
- `mixed.example.json` serves with vision and the control vector through `check_launch.py`.
- `mixed_split_tests` (with new key tests) and `check_off_is_vanilla.py` pass.
- `.ai/reports/report-0074.md` and all changes are committed and pushed to `feature/mixed-native`.
- If stopped early: **Partial**, with what works. A redesign is not an acceptable outcome; if the design cannot work,
  stop and say why.

## Reporting
- Every vanilla file changed, with the function and one line on why; design decisions (transport, keys).
- The control-vector and vision check results; the tok/s table beside HIP-only.
- Kernel log and core-dump lines, or "none". GPU run time used. What remains open, including the refusals still in
  `.ai/DIVERGENCE.md`.
