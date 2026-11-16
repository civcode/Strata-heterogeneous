# Status

Turn holder: **Coding Agent**, handoff `0066` (Tier 1) open.

## Current milestone

**M1 reached** (report 0065): HIP-only on `feature/mixed-native` is plain vanilla 0.2.15 and runs on the two R9700
with the machine's existing model files.

Vanilla HIP-only baseline (Qwen3.8-Flash-Next Q2_0, context 196608, `--kv int8`, MTP `--spec 4 --spec-min-p 0.5`,
`--prefill auto`, layer split auto K=24, `--pipeline-windows 2`, vision and control vector on):

| Request | Prompt read | Prefill tok/s | Decode tok/s | Drafts accepted |
| --- | --- | --- | --- | --- |
| Cold, 69-token prompt, 200 new | 69 | 215.5 | 85.0 | 105 / 184 |
| Repeat (64 tokens reused) | 5 | 171.4 | 84.9 | 106 / 186 |
| 5397-token prompt, 50 new | 5397 | 2283.0 | 96.1 | 25 / 45 |

Time to first token was not measured (non-streaming client). Short samples, not a qualification.

Also established: the three trees build (CUDA with `/usr/local/cuda-13.0`); the seam's CPU tests pass in each; the
shared hand-off probe passes in all CUDA/HIP pairings.

## Decisions taken by the Human User (2026-11-16)

- Reference is vanilla 0.2.15 (`c7757280`). `8db7ddc4` is private work in progress and stays unpushed.
- New work on a fresh branch; the old branch is the fallback and is frozen at `0dca627f`. No fixes there.
- The numerical probes are not carried over.
- HIP hosts the loop; the CUDA card is the remote front stage.
- Agent tiers per `.ai/WORKFLOW.md`; hardware test budget 5 minutes per agent turn unless a handoff sets another
  with the Human User's agreement.

## Open decisions for the Human User

- None.

## Private hardening work (`civcode/strata-dev-private`), reviewed 2026-11-16

Vanilla 0.2.15 contains none of it. Handoff 0066 Part B tests for the HIP snapshot readback bug and ports the
minimal fix (pinned readback, parking early return, AVX-512 guard) only if the reproducer still fails. Deferred:
restore preflight, Phase 10 (pinned ring, prefill stager, remote experts, Q36), Phase 11B (shutdown ownership),
all diagnostics. Also open: whether `-DSTRATA_MMQ=ON` still speeds up prefill on the R9700 (A/B in handoff 0066).

## Next step

Handoff 0066 (Tier 1, up to 60 minutes of GPU run time, agreed): Part B as above, then Part A, milestone M2: both
processes are the vanilla `strata` program, the CUDA build as front stage over layers [0,15), the HIP build as host
with a first stage that starts at layer 15.

After report 0066: Chat AI reviews the engine diff against the contract.

Roadmap: https://claude.ai/artifact/NXCzvkbBJ2Z9fKts6ADRNn
