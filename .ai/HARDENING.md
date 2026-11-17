# Hardening carried on top of vanilla

The engine hardening from `civcode/strata-dev-private` (branch `fix/runtime-parity`) is ported onto vanilla 0.2.15
and carried as a labelled divergence from vanilla. The Human User asked for it after the 17 November host crash on a
vanilla 0.2.15 build; days of work on the patched branch had shown no such failure. No commit is known to prevent that
crash (see the incident report); this is risk reduction on transfer and ownership paths, not a proven fix.

| Private commit | What it does | Port on `feature/mixed-native` |
| --- | --- | --- |
| `05134f54` | HIP pinned snapshot readback, restore preparation and state machine, parking race | `51171e10` (conflicts resolved) |
| `8db7ddc4` | PLE reader test timing; transfer audit document | `b78b5a05` |
| `0ef75af1` | Phase 10E: pinned ring unknown completion, prefill stager, remote expert readback, Q36 | `06cd443c` (conflicts resolved) |
| `ffe0ceaf` | Phase 11B: expert cache, peer experts, remote-opt and conversation completion; prefill failure paths | `d1cc5ef0` (conflicts resolved) |
| `228161c8` | destroy layer-split stages on their owning GPU | `98be7da5` |
| `affbf7e2` | own later-stage raw GPU allocations and streams | `d85aa140` |
| `41fda0fd` | release control-vector tables and penalty history on their GPU | `b8e1087b` |
| `6c47d719` | own primary early GPU buffers and main stream | `f10f6a4d` |
| `b0010efe` | own primary serving buffers and hit event | `61a9c9a0` |
| `0f37490b` | track pipeline and refill handles through shutdown | `dee6a768` |
| `52a69928` | fence all pipeline owner GPUs before release | `bc5ce2a3` |
| (port fix) | vanilla's `GpuStage::arena` record points at the owned arena | `39e22ebc` |

Each conflicted commit's message says how its conflicts were resolved. Every line the private commits add is present
in the port, except where 0.2.15 already had the change (ISA guard names) or where the merge combines both sides
(the restore gate with vanilla's replaced-reply rule; vanilla's gated peer-expert refill with the drain-on-failure
handling).

Not ported: the 10 November diagnostics (transfer watchdog state, stall traces, readback traces), the GFXHUB recorder
and its tests, the `CMakePresets.json` preset (the MMQ build options are recorded in `.ai/PROJECT.md` instead).

Checked by Chat AI without a GPU: the 90 CPU tests that come with the patches pass; the merged `.cpp` engine files
pass a syntax check against CUDA 12.9 headers (`generate.cpp` with the same feature-define errors as unmodified
vanilla). Not checked yet: CUDA and HIP builds, the GPU tests, a run (handoff 0067).
