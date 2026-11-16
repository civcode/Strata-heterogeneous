# Vanilla Strata divergence audit — source-side 0060

## Scope and reference

Owner direction: use vanilla Strata mechanisms throughout; no independently
invented algorithms. Existing M7/implementation contract already mandated
native reuse, so the recent acceptance-counter experiment violated that
direction. This audit corrects that and inventories broader divergence.

Inspected branch feature/heterogeneous-hip-only at d723fd29cbdbea02224fa4d84f2af9c28d808b96;
corrected source through 8e93d521308c8e76ea69be69d798e01a87715a1e. Compared repository main ancestor
146df3c468739147f520ac3816de806f71232571 to the feature branch (450 commits before this audit; no
behind commits). The exact measured vanilla Oberon revision
8db7ddc46cd2575514408c4ab7dd878f69df2f5f (report 0045) was not available through
the connector. Main is a source comparison ancestor, NOT the pinned vanilla
binary/revision. Direct clone/fetch failed DNS in the local container. The
connector returns empty content for the large src/program/generate.cpp at
both source refs, even when requesting line slices/base64. Code search was
unavailable. This matters for caller policy, default options, scheduler and
cache orchestration. Do not guess those details or call this a complete
line-by-line/pinned-vanilla audit.

The full compare is capped at 300 paths; five contiguous ancestor ranges below
all have fewer than 300 paths. Their union is 527 distinct touched
paths (including historical/reverted paths), 49 in shared
src/include/kernels/serve/strata_cli, 95 in experimental
heterogeneous excluding tests/docs (includes CMake, instructions and workload
files). The [changed path manifest](audits/vanilla-audit-0060-files.txt) covers
the whole branch history up to the corrective code commits; it is not a
claim that every line was read. Inspected operational adapters, key shared core
source, changed mechanisms and source contracts; full diff against pinned
vanilla and remaining line-level source review must be finished locally.
No inference behavior was changed in the audit of unrelated historical
hardening/diagnostics and no GPU tests or Actions were run.

| Range | Commits | Paths |
|---|---:|---:|
| 146df3c46873 → 992fc289 | 249 | 196 |
| 992fc289 → 15791b8051e2 | 60 | 202 |
| 15791b8051e2 → 1adb08eeaef9 | 29 | 102 |
| 1adb08eeaef9 → b2840984af3d | 49 | 113 |
| b2840984af3d → 8e93d521308c | 65 | 106 |

## Findings and dispositions

| Area | Source evidence | Disposition |
|---|---|---|
| Adaptive MTP | main MtpDrafter::draft uses min_p and optional caller chain_rule; native header exposes set_chain_rule with STRATA_SPEC_COST and set_max_drafts. 0060 added MtpAdaptiveWidth with arbitrary counters and zero cap. | **Corrected:** deleted controller/header/test/registration. Existing shared pipeline_window_size now optional service selector; generated rows may include a low-confidence final row. Keep native producer and exact target-authoritative clipping. Refuse old adaptive flag and nonzero cost mode until exact native caller extracted. |
| Speculative scheduler | native_mtp_probe.hpp invokes native draft(1) after cursor's accepted prior-row prime, clips offer using request-local policy; live_smoke.cpp requests proposals only after all stage ACKs, uses shared greedy_accepted_prefix. Existing speculation.hpp calls pipeline_accepted_drafts. core pipeline_window.hpp was extracted historically from generate.cpp (not present on main). | Native math and acceptance reused; serial custom orchestration remains significant. Extract native multi-row draft catch-up, chain launch/ready, window pipeline and rollback, with only cross-runtime ownership seam. Do not clone lookahead. Keep existing actual successor pairing and fail-stop until replacements qualified. |
| Attention, GDN, experts | No QSA/GDN kernel implementation files changed in historical inventory. Shared verifier/prefill dispatch code changed (trace/HC/guard/external bridge); native_worker_defaults.hpp hardcodes native flags including layer_set_native_flash_attn_short(false). | **Open divergence risk:** duplicate startup policy can miss native optimized paths. Verify exact vanilla effective initialization/dispatch against both vendors. Reuse existing native attention/GDN/batch/fusion mechanisms, no new attention algorithm or tensor parallelism. |
| Local HIP overlap | hip_native_chain.hpp sets mapped portable/coherent buffers, set_handoff_flags and set_next; native main Verifier::chain_overlap/prelaunch/run uses native flags and recursion. | Preserve. Native same-runtime execution required; do not split HIP into independently scheduled processes. |
| Extra synchronization | main verify.cpp blocking stream sync after run, recursive commit/wait_commit; current CUDA and HIP adapters add event record/query polling after run/commit plus necessary transfer waits. | Native completion opt-in retained: removes only redundant completion events, uses native wait_commit, never removes DMA fences. Source supports it but integrated/native qualification pending. Audit custom polling only at external boundary; use vanilla waits, not more polling. |
| CUDA–HIP handoff | host.cpp/session.cpp/final_pipeline.cpp/shared.cpp custom request/frame/socket/memfd slots and all-owner commit; native guarded handoff adds checked wait/publish kernels in verify_kernels.cu, arm_external_handoff in Verifier. | Necessary transport/identity adaptation, not model math. Native prelaunch alone does not prove safe cross-runtime events. Retain qualified serial/ordinary compact/fanout/early metadata; MTP extensions opt-in. Guarded direct remains diagnostic bounded, not default; review against actual native wait/publish semantics, do not add a competing protocol. |
| Prefill | current Prefill::run_external validates range and invokes native run_impl + drain_pipeline; native local successor chain/HandWorker/hand_done is retained. native_prefill_domain.hpp allocates per-request scratch and uses R-only framed boundary/staging fence. | Native math/pipeline used; bridge and planning restrictive. **Corrected:** withdraw 0060 piece upload; 16MiB / (4*2560*4) = 409 rows per frame, admitted MTP chunk <=64, so no multi-piece benefit there. Retain fence, follow vanilla piece pipeline if native boundary adaptation justified. |
| Batch draft seed | main Prefill::draft_kv rounds scratch to multiples of 64 and declines B<64. 0057 changed local embedding branch minimum 32. 0058 callback savings did not demonstrate end-to-end speedup. | **Corrected:** restored exact vanilla 64-row minimum, preserve private terminal native embedding and reserve/decline guards (local bound n<=64). 32-row configuration falls back narrow even if batch switch on; prior observations remain historical. Do not change vanilla batch threshold or force unsupported HIP async draft prefill. |
| Prompt cache / reset | session.cpp exact equal blob eb91c... at main and audited branch; native session_zero called by each owner. Native MTP invokes qsa_state_zero. Resident adapter runs fresh requests; Prefill/stream allocated per bulk call. | No cache reuse; use native conversation checkpoint/KV restore and lifecycle, not a custom cache. **Corrected:** reset-overlap candidate removed; current reset semantics kept and per-stage profiling retained. |
| Stage initialization & ownership | CUDA/HIP native_stage.cpp, stage_resources.cpp, file_experts.cpp are near duplicates: normalized CUDA nonblank lines also in HIP 272/274, 206/208, 287/289 (simple cuda/hip spelling normalization, not semantic proof). They use native WeightTable, NativeDense, session_init/zero, ExpertCache and native CPU pool. Both stage_resources cleanup workaround global qsa_ord0 because main session_release iterates starting at index 0. | **Open:** extract/share actual native stage initialization, dispatch and owner cleanup where applicable. Do not remove offset workaround or collapse separate runtimes without validated native replacement. Resource/identity admission belongs to boundary; no duplicate kernel loading behavior. |
| Shared diagnostics & arithmetic | numerical_trace, layer_taps, shared_expert microtrace, MTP front trace are instrumented existing operations. native_dense.cpp adds explicit HC_WEIGHT_REQUEST; prefill hc_experiment.hpp and hc_replay.cpp introduce BF16x2/FP32 diagnostics. live_smoke explicitly refuses HC experiment/weight contract in resident/MTP. | Keep diagnostics quarantined, not optimization paths. Do not promote alternative HC arithmetic or historical numerical experiments. Review shared-core gating before upstream merge. |
| Native hardening | Compared conversation_snapshot.cpp/state.cpp, expert_cache.cpp, native_expert.cpp and q36_engine.cu against main: copy completion/owner-scoped frees/fail-stop, nullable draft snapshots, AVX512 probe guard, route completion checks. Related event_completion, transfer_terminal and conversation_restore helpers added. | These are lifecycle/correctness fixes, not new algorithms. Preserve until exact pinned upstream diff/review; do not undo safety to imitate vanilla. q36 unrelated to Q2_0 target; avoid additional feature work there. |
| Configuration, CLI, serving | strata_cli/heterogeneous.py and serve/heterogeneous.py bound spec4/minp0.5, context/prompt chunk, greedy resident metadata, service refusal/cancel/restart; serve/server.py adapts existing HTTP engine, not separate HTTP service. | Keep cross-process admission/retirement and existing APIs. Guard replicated policy against pinned native generation options. No unverified new native option defaults. Native cache/spec semantics should move to shared interfaces and leave only boundary checks in CLI/server. |
| Process/safety/tracing | process.cpp has both legacy forced-cleanup path and graceful_only mode used by current native live_smoke launch with timeout; serve retirement uses cooperative EOF/drain and verifies process group. CPU test fixtures may terminate mock processes. | Preserve verified native graceful-only path, no forced GPU termination/new supervisor. Legacy test/demo routes are not production optimization targets. Host-span instrumentation is permitted; not GPU occupancy. |
| Placement | qualified HIP24/24 and mixed15/18/15 are explicit, no evidence native auto planner reused for heterogeneous choice. | Use actual vanilla placement/calibration after P0, respect CUDA reserve. Do not invent heuristic or new tensor-parallel architecture. |

## Shared source references (exact blobs inspected)

- main verify.cpp af6179c7373b0c897118336b3b25a96a5cc76d7c vs branch 9096ba8611c6cad9b8c240ac3a66862720addaad: native run/commit/chain remains, guarded external handoff and diagnostics added.
- main prefill.cpp 0ef3cce9f90dde654e0a22d9918b79320cb5dcee vs original branch 72e2c20efcb29c1abd62d91d0c01f26f2a2b69d4: external interface, trace/HC experiments, local embedding; 64 batch rule restored in correction.
- main mtp.cpp 5403cdc29e90aa2bdd420d071a813c6be37179a8 vs original branch 0503c400dcc20369fe667009f5111e220fb3d71d: existing native draft min_p/chain_rule semantics retained, resident sizing/private embedding/front trace added.
- main session.cpp and original branch both eb91c380d1e6295d6e87f05bb1349fd28a6fbae2. No session core arithmetic change.
- main generate.cpp 03113d5a494b282ff51c9fa64b49eeb17d75e9d2 vs original branch 14b4ed6f9373a2f34153062781814dcd288ef183, contents inaccessible; cannot certify vanilla caller/defaults from blobs alone.

Shared engine/CLI/service touched-path list (historical, not net or reviewed-line claim):

- `include/strata/core/conversation_copy_resources.hpp`
- `include/strata/core/conversation_snapshot.hpp`
- `include/strata/core/event_completion.hpp`
- `include/strata/core/expert_cache.hpp`
- `include/strata/core/expert_cache_completion.hpp`
- `include/strata/core/layer.hpp`
- `include/strata/core/mtp.hpp`
- `include/strata/core/mtp_front_trace.hpp`
- `include/strata/core/numerical_trace.hpp`
- `include/strata/core/pipeline_window.hpp`
- `include/strata/core/transfer_terminal.hpp`
- `include/strata/core/verify.hpp`
- `include/strata/kernels/handoff_guard.hpp`
- `include/strata/kernels/shared_expert.hpp`
- `include/strata/kernels/verify_kernels.hpp`
- `include/strata/prefill/hc_experiment.hpp`
- `include/strata/prefill/moe_fused.hpp`
- `include/strata/prefill/prefill.hpp`
- `include/strata/program/conversation_restore.hpp`
- `serve/heterogeneous.py`
- `serve/heterogeneous_contract.py`
- `serve/heterogeneous_long_context.py`
- `serve/heterogeneous_retirement.py`
- `serve/server.py`
- `serve/test_context_overflow.py`
- `serve/test_heterogeneous_http_disconnect.py`
- `serve/test_heterogeneous_resident.py`
- `serve/test_heterogeneous_retirement.py`
- `src/core/conversation_completion.hpp`
- `src/core/conversation_copy.hpp`
- `src/core/conversation_readback.hpp`
- `src/core/conversation_snapshot.cpp`
- `src/core/conversation_snapshot_test.cpp`
- `src/core/conversation_state.cpp`
- `src/core/expert_cache.cpp`
- `src/core/layer.cpp`
- `src/core/mtp.cpp`
- `src/core/native_dense.cpp`
- `src/core/verify.cpp`
- `src/kernels/cpu/native_expert.cpp`
- `src/kernels/cuda/shared_expert.cu`
- `src/kernels/cuda/verify_kernels.cu`
- `src/prefill/hc_replay.cpp`
- `src/prefill/prefill.cpp`
- `src/program/conversation_restore_orchestration_test.cpp`
- `src/program/generate.cpp`
- `src/qwen35/q36_engine.cu`
- `strata_cli/cli.py`
- `strata_cli/heterogeneous.py`

## Corrections and validation

Two source commits withdraw separate MTP controller, reset overlap, piece
upload, introduce shared native confidence selector as opt-in, restore native
E-9 minimum. No new speculative algorithm, removal of safety gates, default
profile enablement or new kernel math. Existing qualified baseline fixed spec
and confidence-gated producer remains default (except old opt-in 32-row batch
now correctly falls back). Legacy enabled experiment flags rejected. Keep
native result/metadata/fanout/completion/profiling candidates separate.

Local GCC C++17 -O2 -Wall -Wextra -Wpedantic -Werror native pipeline-window suite:
1816 assertions PASS, source header 4c7601cc7dce6f27db2afbe1615ab7d8d571a774,
test f509dafa55b6ab41a77f2ac4eab88f20515ccd27 exact blobs. Existing isolated
MTP policy 86,271, bulk pairing 1,040,444 and seed accounting 42 assertions
pass (unchanged exact sources). Full worker integration, new source admission,
64-row native batch GPU path, native dispatch, exact pinned vanilla local
source diff and GPU performance are unverified. No claim that a single source
review or CPU tests proves speedup or pinned vanilla parity.

## Required next step

Use the exact pinned clean vanilla checkout within authorized Oberon source
boundaries and full git diff/inventory. Inspect large generate.cpp actual
native setup, cost-aware chain rule, confidence gating, per-round catch-up,
native pipeline, cache and reset; compare effective paths. Finish audit of
all remaining changed shared files, classify all adapter families and isolate
historical diagnostics. Use source-grounded extraction only; if a native
reference cannot be inspected, report instead of inventing. The revised
handoff 0060 is the sole next qualification, and replaces the old optimization
batch instructions. Do not run the old adaptive/reset/upload arms.
