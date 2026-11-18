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
6. Withdrawn (2026-11-18): the engine hardening from `civcode/strata-dev-private` is no longer carried. The 17 November
   crash is attributed to a launch config that mixed components from different builds. The hardened line is kept on
   branch `archive/hardened-0069`; see `.ai/HARDENING.md`.
7. A launch config uses the engine binary and libraries of one build directory only. Before a run, check that every
   path in the config that points into a build tree points into the same one.

The target design and the plan are in `.ai/SEAM.md` and the roadmap referenced from `.ai/STATUS.md`.

## References

| What | Revision |
| --- | --- |
| Vanilla reference and base of this branch | `c77572806996349b49995e4f98d41b5294d36a21` (0.2.15) |
| Working branch | `feature/mixed-native` |
| Frozen earlier adapter (not developed, kept as fallback) | branch `feature/heterogeneous-hip-only`, archived as `archive/heterogeneous-adapter-0062` at `0dca627f` |
| Earlier local vanilla pin on Oberon | `8db7ddc4`: a private work-in-progress branch, not pushed (report 0063) |

Handoff and report numbers continue from the earlier branch (0062 was its last).

## Environment (Oberon)

- Cards: one NVIDIA RTX 4070 Ti SUPER (16 GB, CUDA, sm_89) and two AMD Radeon AI PRO R9700 (HIP, gfx1201).
- HIP toolchain: ROCm at `/opt/rocm/core-10.1` (`ROCM_PATH`, `HIP_PLATFORM=amd`), Ninja, `-DCMAKE_HIP_ARCHITECTURES=gfx1201
  -DSTRATA_PREFILL_MMQ=ON -DSTRATA_MMQ=ON -DCMAKE_BUILD_RPATH=/opt/rocm/lib`. On 0.2.15 the MMQ prompt kernels need
  both MMQ options; report 0066 measured about 22% faster prefill with them (2641 against 2163 tok/s, one sample).
- CUDA toolchain: CUDA 13.0 at `/usr/local/cuda-13.0` (vanilla 0.2.15 is built with CUDA 13.0; driver 580.178.04).
  Configure the CUDA tree with `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=89`.
  `/usr/local/cuda` still points at CUDA 12.0, which the frozen branch was built with: leave it. nvcc 12.0 cannot
  compile vanilla 0.2.15 (report 0063). Do not change vanilla sources to fit a toolkit.
- Known-good vanilla run config: `/home/chris/workspace/Strata/strata-q2_0.json`, started by `run-q2_0.sh` in that
  checkout (port 8081, `HSA_ENABLE_SDMA=0`). Model: Qwen3.8-Flash-Next GSQ-RCO Q2_0, context 196608, `--kv int8`,
  MTP with `--spec 4 --spec-min-p 0.5`, `--prefill auto`, `layer_split: auto`, vision on, and a scaled control
  vector over layers 4 to 44. Its engine arguments are listed in report 0064.
- Use the same build toolchain as vanilla Strata and the same checkout-local `.venv` setup. Do not use a system or
  global Python for project commands.
- Build with 28 jobs.
- The local vanilla checkout (`/home/chris/workspace/Strata`) is reference material: do not change its files or
  write new ones into it. A run config copied from it must have every path it writes to (for example `log`)
  redirected into the working checkout before a server is started with it.
- Never terminate a GPU process by force. Shut down cooperatively and wait for the drain.
- Do not dispatch, enable or add GitHub Actions.

## Project stage and hardware test budget

- Stage: **early development**.
- Hardware test budget: **5 minutes per agent turn**, short prompts, runs sequential, never two GPU jobs at once.
- Long-duration tests (soak, long context, capacity sweeps, the 160 tok/s qualification) wait for
  release-candidate state and an explicit request from the Human User.
- Never compare a traced run with an untraced one.
