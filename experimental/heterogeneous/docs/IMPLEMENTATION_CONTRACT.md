# CUDA → HIP implementation contract (M2.18 onward)

**Status:** Active project implementation and validation contract for `experimental/heterogeneous/`. It supersedes M2.10–M2.17 instructions about the *next* investigation and about accepting the heterogeneous architecture. Their actual measurements and original numerical-parity criteria are retained as historical evidence; no prior KL test is retroactively declared passed.

## Objective and implementation boundary

Run one model across contiguous, non-overlapping layers using **separately built, isolated native Strata CUDA and HIP workers**. The target plan is `cuda:0:0-12,hip:0:12-30,hip:1:30-48`. Validate the **CUDA layer 11 → HIP layer 12 transition** first, then HIP → HIP continuation and the full pipeline.

Reuse `core::Verifier`, `set_stage` / `run` / `commit`, existing CUDA/HIP inference kernels and model execution paths. Our heterogeneous code should only own the necessary stage-local resources, worker orchestration, transfer, synchronization, failure handling and accepted-prefix coordination. **Do not implement a second inference engine.** Do not alter CUDA-only or HIP-only defaults.

The existing per-row handoff is the ordered float32 layout `R` (`hc * n_embd`), `bo` (`n_embd`), `inject` (`hc`), with no conversion or new arithmetic. This is **not** a session snapshot: K/V, GDN/recurrent, PLE and expert state remain with their owning stages. Use independent processes with backend-local device ordinals, runtime-owned allocations, host staging and explicit fences. Do not pass device pointers, stream or event handles between CUDA and HIP.

Preserve the existing model/GGUF/pack/profile provenance checks, geometry, plan/request/window/position/token identities, PCI device selection, activation bounds, finite guards, VRAM reserves, poisoning and commit acknowledgements. A caller-supplied model fingerprint is an identity check, **not a cryptographic attestation of all GPU weight bytes**.

## Required architecture acceptance — real production execution path

| Condition | Required evidence |
| --- | --- |
| Full pipeline | Native CUDA 0–11 → HIP 12–29 → HIP 30–47 executes real model windows and produces finite final logits and valid predictions |
| First boundary bytes | The actual CUDA-produced `R | bo | inject` bytes match the bytes presented to the HIP verifier **before layer-12 arithmetic**, with correct dimensions and order |
| Transfer ordering | Producer compute and D2H complete before publication; HIP H2D completes before `Verifier::run`; no stale slot or window reuse |
| Stage identity | Expected source, layer ranges, device PCI identities, plan/model metadata, request/window IDs, positions, tokens and payload size; wrong data fails closed |
| State and commit | Stage-owned persistent state is correct; all workers acknowledge the same positive accepted prefix before publication; failed or partially committed sessions are poisoned, not rolled back |
| Repeatability | Fresh sequential real runs with the same configuration complete; record predictions, finite-logit checks, commits and any variability without assuming bitwise GPU arithmetic |

**Byte equality is a transfer requirement, not a CUDA-versus-HIP arithmetic requirement.** Socket or shared-memory checksums validate payload framing but alone do not prove CUDA device → producer staging or HIP staging → device-input identity. Reuse existing buffers and fences; distinguish GPU-produced bytes, producer host buffer, wire/memfd, consumer host buffer and actual HIP device input. Where any checkpoint cannot be observed, report **UNVERIFIED**, not PASS. Add only the smallest reusable, opt-in production-boundary observation if genuinely needed, with no extra inference math or change when disabled.

Matching predictions do not alone prove a correct handoff. Matching transport bytes do not alone prove valid GPU state or successful final model execution. A short smoke does not prove prefill, long-context, performance or production readiness.

## Cross-backend numerical parity is separate

M2.10–M2.17 compared a mixed CUDA/HIP computation with a full-HIP reference. Their unchanged **KL ≤ 0.001 and prediction/top-1 agreement** is a **historical numerical-parity diagnostic**, not the architecture's transition acceptance gate. The reported M2.17 KL **0.005003 / 0.037718** FAILS that diagnostic although both windows committed and predicted **198 / 271**. The observed layer-5/6 differences occur before the first CUDA → HIP boundary; they do **not** establish a transfer defect.

Do not claim that KL passed, relax its original target, or demand bitwise equality of independent CUDA and HIP calculations. Do **not** require CUDA-to-CUDA equivalence, new layer-5/6 microprobes, isolated reference math kernels or precision toggles before testing the transition. Reopen kernel-level diagnosis only if evidence from the actual boundary or state commits requires it.

## Minimum-code and non-disposable-test policy

1. Inventory existing real `live_smoke`, CUDA/HIP verifier adapters, `WindowAdmission`, frame checksums, stage buffers, device transfer fences, STFR outputs and commit logs. **Prefer no source changes.**
2. Run the real, supervised native CUDA → HIP → HIP system with the existing model and production worker path. Test the first boundary, subsequent HIP continuation, final head and positive-prefix commits, using bounded fresh sequential sessions.
3. Reuse CPU protocol fixtures only for framing, malformed input, failure and commit logic. **Never treat synthetic fixture outputs as evidence of inference numerics or GPU handoff integrity.**
4. If required transfer evidence is missing, add the **smallest opt-in, durable, reusable observation at the existing transfer interface**. Do not add alternate floating-point operations, new quantization, a separate inference implementation, a temporary GPU runner or a new diagnostic framework for unrelated arithmetic differences.
5. Report **PASS / FAIL / UNVERIFIED** per condition, distinguishing static code inspection from hardware proof. Include tested commit, source/model/pack provenance, plan and layer layout, PCI devices, token/position/window IDs, exact comparison checkpoints, commit outcomes and raw logs. Report legacy KL separately as a **non-gating** diagnostic when available.

## Safety and historical context

Work only in `/home/chris/workspace/Strata-heterogeneous` and `/home/chris/tmp/strata-heterogeneous` on Oberon, never in either `Strata-dev` directory. No GPU reset/reboot or driver unload; keep builds separate and ≤28 parallel jobs, preserve VRAM reserves and model checks. **No GitHub Actions runs or automatic triggers.** No hardware PASS claims without actual measurements.

Historical evidence: [M2.9 mixed native run](LIVE_GPU_HANDOFF_M29.md), [M2.10 numerical parity](NUMERICAL_PARITY_M210.md), [M2.17 diagnostic](STAGE_MICRO_LOCALIZATION_M217.md). Production interfaces: [CUDA verifier](../src/cuda_verifier.cpp), [HIP verifier](../src/hip_verifier.cpp), [framing](../src/host.cpp), [window protocol](VERIFIER_WINDOW_V4.md).
