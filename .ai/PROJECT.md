# Project rules

Read this file at the start of every turn. The collaboration protocol is `.ai/WORKFLOW.md`.

## What this project is

Strata running one model across an NVIDIA card (CUDA) and AMD cards (HIP) at the same time. The fork is
`civcode/Strata-heterogeneous`; vanilla is `Niko1221/Strata`.

## Implementation contract

1. Use vanilla Strata code wherever possible. Add code only for CUDA/HIP interoperability.
2. A change to a vanilla file must be a behaviour-preserving move or extraction, or the minimal hook the CUDA/HIP
   seam needs. No new scheduling, speculation, prefill, cache, sampling or synchronisation logic.
3. Builds without the heterogeneous option behave exactly as vanilla.
4. HIP-only (two AMD cards) is plain vanilla Strata with a native layer split. It uses no heterogeneous code.
5. Numerical differences between CUDA and HIP are low priority.

The target design and the plan are in `.ai/SEAM.md` and the roadmap referenced from `.ai/STATUS.md`.

## References

| What | Revision |
| --- | --- |
| Vanilla reference and base of this branch | `c77572806996349b49995e4f98d41b5294d36a21` (0.2.15) |
| Working branch | `feature/mixed-native` |
| Frozen earlier adapter (not developed, kept as fallback) | branch `feature/heterogeneous-hip-only`, archived as `archive/heterogeneous-adapter-0062` at `0dca627f` |
| Earlier local vanilla pin on Oberon | `8db7ddc4` (to be pushed as an archive ref by handoff 0063) |

Handoff and report numbers continue from the earlier branch: the next handoff is `0063`.

## Environment (Oberon)

- Cards: one NVIDIA RTX 5080 (16 GB, CUDA) and two AMD Radeon AI PRO R9700 (HIP, gfx1201).
- Use the same build toolchain as vanilla Strata and the same checkout-local `.venv` setup. Do not use a system or
  global Python for project commands.
- Build with 28 jobs.
- The local vanilla checkout is read-only reference material. Do not modify it.
- Never terminate a GPU process by force. Shut down cooperatively and wait for the drain.
- Do not dispatch, enable or add GitHub Actions.

## Project stage and hardware test budget

- Stage: **early development**.
- Hardware test budget: **5 minutes per agent turn**, short prompts, runs sequential, never two GPU jobs at once.
- Long-duration tests (soak, long context, capacity sweeps, the 160 tok/s qualification) wait for
  release-candidate state and an explicit request from the Human User.
- Never compare a traced run with an untraced one.
