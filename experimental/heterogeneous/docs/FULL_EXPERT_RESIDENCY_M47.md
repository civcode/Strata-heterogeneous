# M4.7 — Full stage expert residency

**Status: bounded M4.7 PASS on Oberon** at tested source revision `6e295857e2781cd79a59ebccd11b4680f80153eb`; [report 0004](../../../.ai/reports/report-0004.md). M4.1–M4.6 bounded PASS in [report 0003](../../../.ai/reports/report-0003.md) predates these changes.

## Why
The native v2 config previously capped each card at 1024 Q2_0 expert slots, at most 1.318 GiB of routed-expert blobs per card, independent of stage size. All other routed experts use the existing file-backed CPU pool. Mapping the complete expert file is not equivalent to loading it into device memory. Persistent resident workers need not have all experts resident. The observed low VRAM and ~21 feedback tok/s motivate testing full residency, not an assumed speedup.

For the validated plan `cuda:0:0-12,hip:0:12-30,hip:1:30-48`:

| Backend/stage | Layers | Explicit slots | Q2_0 cache bytes | GiB |
| --- | ---: | ---: | ---: | ---: |
| CUDA 0 | 12 | 6144 | 8493465600 | 7.910 |
| HIP 0 | 18 | 9216 | 12740198400 | 11.865 |
| HIP 1 | 18 | 9216 | 12740198400 | 11.865 |
| Total | 48 | 24576 | 33973862400 | 31.641 |

This is expert payload only, not total allocation. Native dense/PLE weights, compact pack weights and global duplicates, context/session state, verifier workspaces, graphs, runtime buffers and safety reserves are additional. There is no unified 80 GiB pool. Some PLE/embedding operations may still involve CPU or file-backed data (full routed-expert residency is not an all-GPU claim for every operation).

## Implementation
- V2 accepts positive slots up to the stage's actual canonical 512-per-layer capacity; v1 preflight's 1024 limit is unchanged. Python model/readiness gates now reject zero and oversized stage counts rather than estimate a silently clamped cache.
- Explicit full-stage slots demand every distinct expert for that stage from the real verified STRP profile. Incomplete or invalid pairs fail before cache allocation. Existing smaller configs retain ranked hybrid placement. No invented or substituted weights.
- Both backend providers retain existing free-memory/reserve, checked size, fill, read-back verification, host/device-residency upload and synchronization. Full mode does not relax any guard. Native verifier already checks its stage's complete table and selects its own all-resident zero-doorbell graph when enabled; heterogeneous adapters call set_stage before init. No new kernels or alternate inference path. Host-side miss in a full stage fails closed.
- Each provider logs HETERO_EXPERT_RESIDENCY at completion of cache startup (backend, ordinal, layer range, filled, total, bytes, full flag) and HETERO_EXPERT_ROUTE_STATS on normal shutdown (host callback count, routed entries, nonresident entries). The counters measure host-served callbacks, not total GPU routes or distinct CPU jobs; zero host callbacks in the native all-resident graph does not mean zero tokens. Require matching GPU graph selection, real commit/output evidence and complete slot maps; never infer total GPU activity from the callback ratio alone.

## Reported Oberon qualification

All 39 host CTests and separate CUDA worker/coordinator and HIP worker builds passed. The original configs actually requested only 16 slots per GPU (not 1024). New staging copies changed only slot counts, preserving the exact 2048-cell context, max_t 8, 512 MiB reserves, model/pack/profile identity and device plan. All three verified slot counts and payload sizes matched the table above. Each verifier reported its 100% VRAM-resident zero-doorbell graph; all three normal-shutdown counters reported zero host callbacks and zero misses.

| Matched 42-token prompt / 64-token completion, socket auto | Hybrid | Full |
| --- | ---: | ---: |
| Warm feedback decode, runs 2/3 (tok/s) | 21.31 / 21.37 | 36.82 / 36.85 |
| Warm prefill, runs 2/3 (ms) | 915.17 / 907.44 | 415.03 / 423.51 |
| Startup (ms) | 1249.4 | 7271.74 |
| Shutdown (ms) | 162.26 | 242.79 |

Sampled peak used VRAM in sequential hybrid/full telemetry: CUDA 11879317504 bytes, HIP0 17327644672 bytes, HIP1 17411874816 bytes. Observed maximum sampled utilization 87%/59%/73%; not instantaneous maxima or sustained utilization. Following normal full-server shutdown the reported idle allocations returned to baseline (468779008 bytes CUDA, 59912192 each HIP). The existing loopback server passed buffered/SSE equivalence for the tested request, actual streaming, cooperative cancellation, subsequent A and stable worker identities. No production source fixes were required.

Fresh and resident generations were reproducible within each cache mode. Both generated ` Paris.` for A. B's hybrid completion was ` Vienna, which is also the largest city`; full was ` Vienna, which is also the country's`. Long outputs also differed reproducibly. This is not proof of numerical parity or a demonstrated numerical cause. Full routed-expert residency does not mean every non-expert operation runs on GPU. Native fault recovery, abrupt parent death, broader parity, long contexts, bulk prefill and full packaging remain unqualified. Report 0004 is the evidence reviewed here; raw logs remain on Oberon at `/home/chris/tmp/strata-heterogeneous/m47-0004-Xi15bUiZ/`.

## Original validation procedure
Only the new C++ capacity helper was compiled and run locally: strict GCC C++17, byte-identical Git blob d3094661d50095f88ac75c918a4d48e19b7b61b8, 29,400 boundary assertions PASS. Full checkout retrieval could not resolve github.com; native and expanded CPU contract integration tests and actual memory/throughput were pending Oberon at handoff; the bounded results above supersede that pre-handoff status for the tested revision. Use the same 2048-cell stage configs, model, profile, original GGUF, precision, reserves, layer split and devices for matched hybrid/full comparison. Copy configs to a fresh staging directory and change only max_cache_slots for full mode. Use the existing CPU model gate, passive readiness estimator and native preflight before explicit live operation. Do not claim exact VRAM without observed telemetry.

For validation record actual fill counts and free VRAM, existing verifier's `100% VRAM resident: zero-doorbell graph` diagnostics, startup and warm timings, PCI IDs, token/commit traces, host callback/miss stats, ordinary cleanup and remaining limitations. Compare against *new* hybrid controls on the same revision, with both chunk 1 and 8 as relevant. Hybrid CPU and GPU arithmetic may differ; record real token differences rather than silently accepting or demanding bitwise parity between different execution paths. No new GPU kernels or precision changes to force an equivalence result. Repeat fresh control to distinguish reproducibility; investigate transfer/state defects only with evidence.

GPU tests within the assigned project scope are implicitly authorized when the user starts a Coding Agent task. No driver/device resets, reboots, forced worker kills, unrelated worktrees, upstream writes or GitHub Actions. Native fault recovery remains a separate gate.
