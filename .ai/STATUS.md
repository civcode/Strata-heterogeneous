# Status

Turn holder: **Chat AI** (no open handoff; last closed handoff is 0062 on the frozen branch).

## Current milestone

Chat AI turn 1 of the roadmap, in progress: foundation on `feature/mixed-native` (vanilla 0.2.15).

Done:
- Branch cut from vanilla `c7757280`; frozen adapter archived as `archive/heterogeneous-adapter-0062`.
- `.ai/WORKFLOW.md` (v1.2), `.ai/PROJECT.md`, this file.
- Stage inventory and host decision: `.ai/SEAM.md`. The HIP process hosts the loop; the CUDA card is a remote
  front stage that attaches through vanilla's own stage hand-off.

Not done yet in turn 1:
- Front-link interface and wire messages (header, CPU tests).
- Port of the boundary code the seam needs from the frozen adapter (frame codec, shared-memory slot, process
  supervision).

## Decisions taken by the Human User (2026-11-16)

- Reference is vanilla 0.2.15 (`c7757280`); `8db7ddc4` is archived only.
- New work on a fresh branch; the old branch is the fallback and is frozen at `0dca627f`.
- The numerical probes are not carried over.
- The stage inventory decides the host process.

## Open decisions for the Human User

- None blocking. The host decision in `.ai/SEAM.md` is reported for acknowledgement before the front link is built.

## Next step

Chat AI: finish turn 1 (front-link interface, boundary code port), then write handoff 0063 (Tier 2): build the
three trees on Oberon, push the `8db7ddc4` archive ref, and record the vanilla HIP-only baseline using the
existing known-good vanilla run config with only `exe` pointed at this branch's HIP build.

Roadmap: https://claude.ai/artifact/NXCzvkbBJ2Z9fKts6ADRNn
