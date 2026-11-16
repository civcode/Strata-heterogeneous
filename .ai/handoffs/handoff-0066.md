# Handoff 0066

## Tier
1

Part A changes vanilla's engine (`generate.cpp`, the verifier and the prompt path) so that one stage of a layer
split runs in another process on another GPU runtime. Each step is only known to be right after a CUDA build, a HIP
build and a run on the cards, and there are many such steps; written without a build it would arrive as a large
body of unverified engine code. Part B is routine work placed first at the Human User's request, so that one agent
turn covers both.

## Objective
- **Part B (first):** find out whether vanilla 0.2.15 still has the HIP snapshot readback bug, port the fix if it
  does, and measure whether the MMQ prompt kernels still pay on the R9700.
- **Part A (only after Part B is finished):** a mixed request completes through vanilla's own generation loop with
  layers [0,15) on the NVIDIA card and [15,48) on the two R9700. This is milestone M2.

## Context
- Read `.ai/PROJECT.md` and `.ai/SEAM.md` first. `.ai/STATUS.md` has the vanilla HIP-only baseline numbers
  (report 0065).
- Worktree `/home/chris/workspace/Strata-mixed-native` has host, HIP (`build-hip`) and CUDA (`build-cuda13`) trees
  that build, and a working copied run config `strata-q2_0.json` (reports 0063 to 0065).
- `hetero/` is a host-only library, CPU-tested: `FrontStage` (the calls to forward), `SharedRegion` (one memfd
  mapping with page-aligned slots), `RemoteFrontStage` / `serve_front` / `FrontBackend` (the socket link),
  `WorkerProcess` (start on descriptor 3, end by closing the socket, never by a signal). Nothing in the engine uses
  it yet.
- The shared hand-off probe showed that CUDA and HIP both register one shared mapping as mapped pinned memory and
  that a kernel of one runtime can wait on a word a kernel of the other runtime wrote (reports 0063, 0064).
- The private hardening work is commit `05134f54` in the local vanilla checkout (`/home/chris/workspace/Strata`,
  read only) and the later commits of `civcode/strata-dev-private`. Vanilla 0.2.15 contains none of it.

## Part B: snapshot readback and MMQ

1. **Is the bug still there?** Vanilla 0.2.15 saves conversation snapshots with
   `conversation_detail::copy_nonblocking` (`src/core/conversation_copy.hpp`): an async copy from the device into
   pageable host memory, then a stream sync. The private audit (`docs/TRANSFER_AUDIT.md` in `05134f54`) found that
   path returning wrong bytes on these cards, shown by the QSA reproducer in that commit's
   `src/kernels/qsa_topk_parity.cpp`. Bring only the reproducer to the unpatched 0.2.15 HIP build and run it on
   both R9700.
2. **If it fails (wrong bytes):** port the minimal fix from `05134f54` and nothing more:
   - `src/core/conversation_readback.hpp` (bounded pinned staging, 4 MiB pieces) and its test
     `conversation_readback_test`;
   - the `host_readback` argument of `copy_nonblocking` and the call sites that pass it in
     `conversation_snapshot.cpp` and `conversation_state.cpp`;
   - the early return in `conversation_park_async_begin` when parking is off;
   - the three-line AVX-512 guard in `src/kernels/cpu/native_expert.cpp`.
   Do not port the restore preflight (`conversation_copy_prepare`, `conversation_restore.hpp`), the Phase 10 or
   Phase 11B work, or any diagnostics. Show the reproducer passing with the fix, and that HIP-only still serves one
   short request with a repeated prompt (conversation cache reuse).
   **If it passes unpatched:** port nothing; report that.
3. **MMQ A/B.** On 0.2.15 the MMQ prompt kernels on HIP need `-DSTRATA_MMQ=ON` together with
   `-DSTRATA_PREFILL_MMQ=ON`; `build-hip` has none (report 0065: every chunk on the FP16 path). Build a second HIP
   tree with both options and send the same prompt of about 5000 tokens to each build once, after one warm-up
   request. Report prefill tokens per second for both. Do not change which build the rest of this handoff uses.

Part B is finished when steps 1 to 3 have an outcome. If a step cannot be completed, report **Partial** and do not
start Part A.

## Part A: the native loop with a remote front stage

### Design (decided; see `.ai/SEAM.md`)
Both processes are the ordinary `strata` program.
- **Host: the HIP build.** It runs the normal serve loop. Its layer split has two local stages, [15,33) on HIP
  device 0 and [33,48) on HIP device 1. Its first stage does not start at layer 0: it takes its residual from the
  stage hand-off, exactly as a later stage of a vanilla split does (`Verifier::set_stage` with `layer_begin > 0`).
  Head, sampler, logits, MTP drafter and the serve protocol stay as they are.
- **Front: the CUDA build**, started by the host through `WorkerProcess`. It does the normal single-card set-up
  restricted to layers [0,15) (weights, session, expert cache, `Verifier`, `Prefill`) and then, instead of the
  serve protocol, runs `serve_front` on descriptor 3 with a `FrontBackend` that calls those objects.
- **The edge** is vanilla's own stage hand-off: the hand-off buffer and its ready word are slots of one
  `SharedRegion`. Each process registers its mapping with `cudaHostRegister(Mapped | Portable)` in place of the
  `cudaHostAlloc` vanilla uses in one process. Only residual rows cross per window.
- **The forwarded calls:** the host's first local stage calls a `FrontStage` (`run`, `commit`, `prefill`,
  `session`) before it waits on its hand-off. This is the mirror image of `Verifier::set_next`.

### Where to look (line numbers at vanilla `c7757280`; `generate.cpp` is unchanged on this branch)
- Split parsing: `split_at`, `split_devs` near `src/program/generate.cpp:4487`.
- First stage session: `session_init(g, o.max_context, K, sbuf, ss, 0, hi0)` near 7304; later stages near 7336.
- Weight trimming per stage: `add_foreign` near 5593 and `load_stage_weights` near 6252.
- Hand-off buffers and ready words: `cudaHostAlloc(... Mapped | Portable)` near 12431 and 12453.
- Verifier stage ranges: `stage_ver(st).set_stage(...)` near 12463; the pipelined pair near 12534.
- Prompt path stages: `sp.set_stage(0, split_at[0], &stages[0]->sp)` near 12157 and `st.sp.set_stage` near 12136.
- `Verifier`: `set_stage`, `set_handoff_flags`, `set_next`, `prelaunch`, `handoff_failed` in
  `include/strata/core/verify.hpp`; `Prefill::set_stage` in `include/strata/prefill/prefill.hpp`.

### Boundaries you decide within
- A change to a vanilla file is a behaviour-preserving move or extraction, or the smallest hook the seam needs.
  No new scheduling, speculation, prefill, cache, sampling or synchronisation logic. If a vanilla mechanism exists
  for something, use it.
- Everything is behind the `STRATA_HETERO` build option and behind explicit command-line options. Without them the
  CUDA and HIP builds must behave as before: the HIP-only baseline must still run from the same build.
- The `hetero/` interfaces are the design. You may extend them (new session operations, more hand-off slots, a
  second buffer for the prompt path) and must keep `hetero_tests` passing with tests for what you add. If the
  design itself cannot work, stop and report why; do not replace it with another architecture.
- Keep the safety gates: identity check before the first window, a window is published only after the front stage
  has acknowledged its commit, vanilla's bounded hand-off wait and finite checks, cooperative shutdown.
- No GPU handle, stream or event crosses the process boundary. No logits cross.
- First version, refused by name at start-up when the front stage is on (add them back in later handoffs):
  `--vision`, control vectors, `--batch` and slots, DFlash2, `--pipeline-windows` above 0, peer devices, the
  all-cards split engine, `layer_split: auto`. Prompt and conversation caches: forward the session operations if
  that is straightforward; otherwise refuse or disable them and say so.

### Bring-up order
1. Front process alone: starts on the CUDA card with layers [0,15), attaches the shared region, serves the link
   (drive it with a small test client before the host exists).
2. Greedy decode, one row per window, MTP off.
3. MTP windows with native `--spec 4 --spec-min-p 0.5`.
4. A prompt of a few thousand tokens through the native prompt path.
5. Through the normal Python server (`serve/server.py --engine strata`) with a run config whose `exe` is the HIP
   build and whose arguments carry the front-stage options: one streaming and one non-streaming request, then a
   clean shutdown of both processes.

### Known open points
- The PLE block runs in layer 1, which is on the front stage, and the loop passes `ple_prev` from the first
  stage's session.
- The first-stage code path in `generate.cpp` is the loop's own local objects, not a `GpuStage`; find every place
  that assumes it starts at layer 0.
- The model identity both processes compare (`LinkIdentity::model_fingerprint`): pick something both builds compute
  the same way from the model files.

## Constraints
- All rules of `.ai/PROJECT.md`. GPU processes are never killed by force. Never two GPU jobs at once, except the
  two processes of one mixed run.
- Do not touch the frozen branch, its worktree or the local vanilla checkout.
- Commit in small steps; keep the two parts in separate commits.

## Hardware Test Budget
Up to 60 minutes of GPU run time in total for this handoff (agreed by the Human User for this bring-up), of which
at most 10 minutes for Part B. Keep every single run under 2 minutes and use short prompts except where a step
asks for a long one. When the budget is used, stop starting runs and report what was not reached.

## Completion Criteria
- Part B: steps 1 to 3 have an outcome; if the fix was ported, the reproducer and `conversation_readback_test`
  pass and HIP-only serves a request with cache reuse.
- Part A (M2): bring-up steps 1 to 5 work for CUDA [0,15) + HIP0 [15,33) + HIP1 [33,48); both processes end
  cleanly; `hetero_tests` passes in all three trees; with the front-stage options absent, the HIP-only baseline
  request still runs.
- If the budget or a genuine blocker stops Part A early, report **Partial** with the last bring-up step that works.
  That is an acceptable outcome; a redesign is not.
- `.ai/reports/report-0066.md` and all changes are committed and pushed to `feature/mixed-native`.

## Reporting
- Part B: reproducer result per card before and after; exactly what was ported; MMQ and non-MMQ prefill rates.
- Part A: every vanilla file changed, with the function and one line on why; the command-line options added; what
  is refused at start-up; how the PLE history and the session operations are handled; the design decisions you
  made inside the boundaries.
- Numbers for mixed beside the baseline in `.ai/STATUS.md`: prefill and decode tokens per second, drafts accepted,
  memory in use per card. For greedy decode with MTP off, whether the first 50 token IDs equal vanilla HIP-only's,
  and if not, the first position that differs (a difference is a finding, not a failure).
- GPU run time used, and what remains open for the next handoff.
