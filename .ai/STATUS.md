# Status

Turn holder: **Chat AI** (no open handoff; 0063, 0064 and 0065 are closed).

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

- Hardware test budget for the Tier 1 engine bring-up (next handoff): 5 minutes is too short for repeated starts.

## Next step

Chat AI: write handoff 0066 (Tier 1). Both processes are the vanilla `strata` program: the CUDA build runs as the
front stage over layers [0,15) and serves the front link instead of the serve protocol; the HIP build runs the
ordinary loop with a first stage that starts at layer 15 and forwards to the front stage.

Roadmap: https://claude.ai/artifact/NXCzvkbBJ2Z9fKts6ADRNn
