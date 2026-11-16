# The CUDA/HIP seam

Result of the stage inventory against vanilla `c7757280` (0.2.15). Source read; nothing here has been built or run.

## What vanilla already provides

- A layer split is a chain of `Verifier` objects. `Verifier::set_stage(layer_begin, layer_end, handoff_in,
  handoff_out)`: a stage that does not start at layer 0 takes its residual from `handoff_in` instead of embedding
  the tokens; a stage that does not end at the last layer writes `handoff_out` and has no head
  (`include/strata/core/verify.hpp`).
- The hand-off buffers are **mapped pinned host memory** (`cudaHostAlloc(Mapped | Portable)`,
  `src/program/generate.cpp` near "hand-off" allocation), not device-to-device copies. Per token: `R`
  (hc x n_embd), `bo` (n_embd), `inject` (hc).
- `Verifier::set_handoff_flags`: a ready word in mapped host memory. The writing stage raises it from the GPU; the
  reading stage's window graph waits on it with a bounded wait and reports a timeout (`handoff_failed`).
- `run` and `commit` continue into the next stage through `set_next`. The head, sampling, logits, `final_R` and
  the taps belong to the last stage.
- `Prefill::set_stage(layer_begin, layer_end, next)`: a later stage reads each chunk's residual rows from the
  previous stage through pinned host buffers.
- Later stages are `GpuStage` objects in `generate.cpp` (own weights, session, expert cache, verifier, prompt
  path). The loop reaches into them directly at about ten places (weights, session, expert room, prompt paths,
  verifier init, PCIe share, adaptive swaps). The first stage is not a `GpuStage`: it is the loop's own local
  objects, used throughout the file.

## Decision: the HIP process hosts the loop; the CUDA card is a remote front stage

Mixed placement is CUDA layers [0,15), HIP0 [15,33), HIP1 [33,48).

- **HIP host (chosen).** The HIP process is the ordinary `strata` program running a two-stage native split over
  [15,33) and [33,48). The head, sampler, logits, MTP drafter, DFlash taps, prompt and conversation caches and the
  serve protocol all stay local and untouched. What crosses is only what vanilla already puts in host memory
  between stages: the residual rows and the ready word.
- **CUDA host (rejected).** The first stage would stay local, but the head, the drafter and two thirds of the
  model would be remote. Every result the loop reads (`out`, logits, `final_R`, taps, draft chain) would have to
  cross the process boundary.

The remote surface for the HIP host is small because it attaches at the `Verifier` / `Prefill` level, not in the
loop:

| Operation | Vanilla call it mirrors |
| --- | --- |
| Run a window of T tokens at pos0 | `Verifier::run` / `prelaunch` on a stage `[0,15)` with `handoff_out` |
| Keep n tokens | `Verifier::commit(n_keep)`, `wait_commit` |
| Read a prompt chunk | `Prefill` stage `[0,15)` writing its rows for the next stage |
| Session reset, rollback, snapshot, restore, truncate | the same `SessionState` functions on the front's session |
| Load, admission, shutdown | the first stage's weights, session and expert cache on the CUDA card |

## New code this implies

1. **Front link in the core** (minimal hook): the first local stage, which now starts at layer 15, forwards
   run / commit / prefill / session operations to a front stage before it waits on its `handoff_in`. This is the
   mirror image of `set_next`.
2. **Shared hand-off memory**: the hand-off buffer and ready word live in a shared-memory mapping that each process
   registers with its own runtime as mapped pinned memory, in place of `cudaHostAlloc`.
3. **Front stage worker** (CUDA build): vanilla's first-stage objects (weights for [0,15), session, expert cache,
   `Verifier`, `Prefill`) driven by the forwarded operations.
4. **Loop support for a first local stage that does not start at layer 0** (weights range, session carve, no
   embedding step). This is the part inside `generate.cpp` and is the Tier 1 work of handoff 0064.
5. Transport, identity checks, all-owner commit acknowledgement and cooperative supervision of the worker.

## Open points for handoff 0064 (need a real build and run)

- Where the PLE history lives: layer 1 is on the front stage, and the loop passes `ple_prev` from the first
  stage's session.
- Whether each runtime accepts a mapped registration of the same shared mapping for the device-side wait and
  write (vanilla uses `cudaHostAlloc`; the frozen adapter did this for bounded cases, report 0047).
- How many places in `generate.cpp` assume the first stage starts at layer 0.
