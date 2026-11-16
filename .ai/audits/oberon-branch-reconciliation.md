# Oberon branch reconciliation — 2026-11-16

Authoritative branch: **feature/heterogeneous-hip-only**. Starting tip 0f0caf335f4c0309037c1cf5f1f004aef300540e. The divergent feature/oberon-launchers tip (51a37a991f09ad315532db77d84e3635e54af719) has 13 commits since 9c2719375ccc8f3d59491a2e677b0b5b956755b7; the authoritative branch had 76 unique commits. Preserve both histories; no force push, reset, wholesale merge or deletion. Never execute stale handoff 0046.

## Review of divergent commits

| Commit | Disposition |
| --- | --- |
| 2450993f | Existing 0061/0062 work owns public args, launch builder, qualified source binding and MTP policy; retain rather than copy the older greedy-only config. |
| 6c4a76e0 | Shared request_log.hpp and native log follower already on authoritative branch. Preserve MTP-aware counts. Active versus consumer-paced timing remains a comparison item, not a claimed exact match. |
| fb81959e | Existing shared pack tokenizer and source attestation from 0062 retained. |
| 7a99f257 | Existing Service progress and validated completion timings retained; no replacement of MTP BATCH record handling with single-token-only logic. |
| 9e829093 | Native application wire mode is useful direction, but this implementation predates current MTP stream/ACK semantics, validated completion and fail-stop path. Not cherry-picked. Requires source-grounded integration and native build/real runtime proof before removing adapter. |
| f1cb3709 | Older flattening selects greedy one-row coordinator, drops current qualified MTP args and metadata. Retain existing native public launcher + resident binding until native wire replacement supports all current functionality. |
| 89e708dd | Replacing entire server from an older upstream ancestor would remove newer disconnect, retirement and MTP configuration integration. Do not restore that file wholesale. |
| 17ba948b | Port explicit native first-layer boundaries and runtime-local device identities through the existing resolver and user_args precedence. Private resource/worker metadata and binding checks remain. No auto split or change in topology. Coordinator-only StrataEngine stall heuristic adjustment not applicable to current ResidentEngine, which retains its finite native-deadline-based supervision. |
| e35f0d6a | Parent-lifetime code arms SIGKILL on parent exit; not safe to port under existing no-forced-GPU-kill/cooperative-retirement contract. Existing worker preflight/capability and fail-stop supervision retained. |
| 02d9fd52 | Wire record formatter only useful with native wire; defer that extraction with wire integration. Existing shared native log formatter stays. |
| fa7dcf3b | Unbounded native request budget would remove the existing 256-token MTP admission and legacy operational caps. Retain qualified limits; do not claim unrestricted native budgets. |
| 8108d0c7 | Port inert greedy filters/seed admission and shared native config sampling defaults. Positive temperatures, active penalties, invalid values and per-request tuning refused. Current native coordinator already resets before HETERO_READY; preserve. |
| 51a37a99 | Handoff 0046 obsolete; replace with next unused actual handoff 0062 on authoritative branch (report-0062-chat.md is prior chat work, not an agent report). |

## Changes on authoritative branch
- 9f70be883a7ac64d2e27e0eb0ec5e11337eb4e92: resident greedy admission and actual run-config sampling defaults via existing native parser; tests. No sampler/algorithm change.
- 3363881eac5273293750abcb3ae612cf53f9a73e: use explicit native --layer-split and --hetero-devices with existing user_args overrides, fail closed on conflicting/invalid layouts. Preserve legacy nested plan support for diagnostics; checked-in profiles derive the exact original HIP 24/24 and mixed 15/18/15 arrangements. Mixed public MTP, --spec 4 total window, threshold 0.5, prefill 32 and max completion 256 unchanged; HIP-only prefill 128 and existing cap unchanged. No C++ inference files changed.
- Native app wire integration, full native options, wider MTP budget, non-greedy sampling, caching and complete parity remain open (see APP_COMPATIBILITY.md and report-0062-chat.md). Do not claim that the old branch's full native wire/server identity has been migrated. The pinned vanilla source object is on Oberon; no guessed algorithm or wholesale backport.

## Local validation
GitHub connector source/blob access available; git ls-remote in container fails DNS for github.com. No complete checkout or native GPU toolchain/access. Local Python unittest ran 11 tests from hash-matched modules (new sampling and placement tests plus exact updated test_mixed_runconfig using exact user_args and resolver). All pass. py_compile passed for those isolated modules. Remote blob identities verified: sampling ce8bf7f9a3b552c5fa52c7175f44d722042dade4/test 2158fd818289ed0e7c5e73c06ab6aa942dc48bdc; placement 86c791101c5d357bc59b41e2e6c4dfe018df575d/test cf7a0e7f26faac4f34d0526be185fbaed89d79db; resolver 8a83b67effa4211bc4bc0dc8e34c3ffa770d7966; native user_args f2f6c108f0a6e90bb37661981331b63b270e598a; resolver tests 3eb6cf9b3be1cdc067e8f0704ef2a5dac7c1d072. New resident test and full server/launcher integration are not run here. No GitHub Actions, GPU run or native build. Full pinned-source/net classification remains incomplete.
