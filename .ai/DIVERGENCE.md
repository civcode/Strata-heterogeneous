# How this branch differs from vanilla

Reference: vanilla Strata 0.2.15 (`c775728`). Kept current by Chat AI; checked with
`python3 hetero/tools/check_off_is_vanilla.py` (no GPU, no build).

## With `STRATA_MIXED_SPLIT` off (the default): vanilla

Every change in a vanilla engine file sits inside `#if defined(STRATA_MIXED_SPLIT)`. With the option off, the six
changed files read exactly as vanilla after the preprocessor. The branch deletes no vanilla line: it adds 713 lines to
7 vanilla files. The only change outside a guard is the option itself in the top-level `CMakeLists.txt` (12 lines,
default `OFF`).

## With `STRATA_MIXED_SPLIT` on: the seam

| File | What it adds | Lines |
| --- | --- | ---: |
| `CMakeLists.txt` | The option; `add_subdirectory(hetero)`; the define on `strata_engine`; links `strata_mixed_split` | 12 |
| `include/strata/core/verify.hpp`, `src/core/verify.cpp` | `Verifier::set_front`: the first local stage forwards each window's run and commit to the front stage | 19 |
| `include/strata/prefill/prefill.hpp`, `src/prefill/prefill.cpp` | `Prefill::set_remote_out` / `set_remote_in`: a prompt chunk's rows go to, or come from, the shared hand-off instead of a stage in the same process | 69 |
| `include/strata/core/conversation_cache.hpp` | `ConversationCache::on_drop`: reports each parked conversation the cache lets go (not one taken for a restore) | 25 |
| `src/program/generate.cpp` | See below | 588 |
| `hetero/` | The seam library (shared region, front link, worker process), its CPU tests, the probe, run configs and tools | 3,291 |

What `generate.cpp` adds:
- **Options:** `--front-exe`, `--front-layers`, `--front-arg` on the host; `--front-stage` on the front stage.
- **Host:** `HeteroHost` starts the front stage (`WorkerProcess`), sets up the shared hand-off, and repeats session
  operations on it: Reset, Abandon, Snapshot, Restore, Park, Unpark, Drop.
- **Front stage:** loads layers `[0, K)`, serves the front link instead of the serve protocol, and keeps its parts of
  checkpoints and parked conversations (`FrontSessionBook`).
- **Layer ranges:** the host's first stage starts at `front_lb`, and the front stage's process ends at `front_le`. This
  applies wherever vanilla assumed the first stage starts at layer 0 or that one card holds every layer: trimmed
  weights, session sizing, the two whole-model graph captures, PLE wiring, the output head, and the expert profile.
- **Front stage settings,** all vanilla options:
  - `--no-prefill-borrow` (no cache-slot loan outside the serve loop);
  - `--mmap-experts`, or the host's `--shared-expert-arena`;
  - lookup windows capped at the hand-off's rows.

## Refused in mixed mode (the engine says so at start)

The reasons are Chat AI's reading of the code, not measured.

| Option | Why not yet |
| --- | --- |
| `--layer-split auto` | The split search prices one process's cards; it does not know the front stage. Name the host's split. |
| `--vision` | Image embeddings enter at the token embedding, which is the front stage's, in the other process. |
| Control vectors | They act on a layer range that spans both processes; the front stage would need its part. |
| `--batch` / `--slots` | The front link mirrors one session; batch slots would need one front session each. |
| DFlash2 | It drafts from taps on every stage, including the front stage's layers. |
| `--pipeline-windows` above 0 | The link carries one window at a time. Planned next (see `.ai/STATUS.md`). |
| `--peer-device`, the helper caches | Both put experts on other cards, which the front stage's process does not see. |
| Session ops Rollback, Truncate | Vanilla's serve loop never calls them; the front stage refuses them. |

## Not carried

- The `civcode/strata-dev-private` hardening: withdrawn on 2026-11-18 (`.ai/HARDENING.md`); archived on
  `archive/hardened-0069`.

## Vanilla findings (upstream candidates, not caused by this branch)

- **HIP first copy stream:** on gfx1201 with ROCm 10.1, the first conversation copy stream and event take about
  148 MiB of VRAM. `conversation_copy_streams_test` allows 16 MiB and the restore guard `kRestoreFirstUseMib`
  assumes 64 MiB (reports 0070, 0071).
- **`ple_parity`:** it needs fixtures (`bench/micro/ple_in.bin`, `ple_out.bin`) captured with `ple_layer_xcheck.cpp`.
  Without them it returns 2 (a failure), not 77 (a skip) (reports 0071, 0072).
- **`cuda_spill_guard`:** it has no sm_89 baseline. Two `native_gu_swiglu_kernel` specializations use 8 bytes of
  stack each (report 0072).
