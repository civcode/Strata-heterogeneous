# Status

Turn holder: **Coding Agent**, handoff `0063` (Tier 2) open.

## Current milestone

M1: HIP-only runs as vanilla on Oberon and the same-machine vanilla numbers exist.

Chat AI turn 1 is finished:
- Branch `feature/mixed-native` cut from vanilla `c7757280` (0.2.15); the earlier adapter is frozen and archived as
  branch `archive/heterogeneous-adapter-0062`.
- `.ai/WORKFLOW.md` (v1.2), `.ai/PROJECT.md`, `.ai/SEAM.md`.
- Host decision (agreed by the Human User): the HIP process hosts the loop; the CUDA card is a remote front stage
  attached through vanilla's own stage hand-off.
- `hetero/`: shared hand-off memory, front link, worker process, CPU tests (pass here, also under ASan/UBSan).
- `hetero/probe/shared_handoff_probe.cu`: unverified GPU probe, to be built and run by handoff 0063.

## Decisions taken by the Human User (2026-11-16)

- Reference is vanilla 0.2.15 (`c7757280`); `8db7ddc4` is archived only.
- New work on a fresh branch; the old branch is the fallback and is frozen at `0dca627f`. No fixes there.
- The numerical probes are not carried over.
- HIP hosts the loop.
- Agent tiers per `.ai/WORKFLOW.md`; hardware test budget 5 minutes per agent turn.

## Open decisions for the Human User

- None.

## Next step

After report 0063: Chat AI turn 2. Write the `FrontBackend` over vanilla's first-stage objects and the engine
hook as far as it can be checked without a GPU build, using the probe result and the baseline numbers; then
handoff 0064 (Tier 1): the engine runs with a first local stage that starts at layer 15 and a remote front stage.

Roadmap: https://claude.ai/artifact/NXCzvkbBJ2Z9fKts6ADRNn
