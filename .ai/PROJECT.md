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
7. Every process in a launch config runs from one build directory, with that build's libraries. A mixed config names
   exactly two builds, one HIP build for the host and one CUDA build for the front stage, and both are built from
   the same commit of this branch. Check before a run (`hetero/tools/check_launch.py`, handoff 0070).

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
- **Production setup** (the Human User, 2026-11-18): `hetero/configs/hiponly.example.json` mirrors it, with
  placeholders for the paths. Port 8081, `parallel` 1, `layer_split: auto`, two R9700.
  - Model: Qwen3.8-Flash-Next GSQ-RCO Q2_0, `--mmap-experts`, `--expert-cache auto`, `--prefill auto`.
  - Drafting: MTP with `--spec 4 --spec-min-p 0.5`, `--pipeline-windows 2`.
  - Context and KV: `--max-context 262144`, `--kv int8`, `--kv-resident 0`.
  - Vision on (GPU encoder, `max_tokens` 1024); a scaled control vector over layers 4 to 44 (project, per-layer).
  - Caches: `--prompt-cache 6 --prompt-cache-spread`, `--conversation-cache-slots 4`,
    `--conversation-cache-min-tokens 8192`.
  - Environment: `HSA_ENABLE_SDMA=0`, `STRATA_SPLIT_OVERLAP=1`, `STRATA_PIPELINE_AUTO=0`, `STRATA_MTP_STAGE0=0`.
  - Performance comparisons are made against this mirror. The older known-good config
    (`/home/chris/workspace/Strata/strata-q2_0.json`, context 196608, report 0064) is not the production setup.
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
- Hardware test budget: **up to 30 minutes of GPU run time per handoff** (Human User, 2026-11-18), short prompts,
  runs sequential, never two GPU jobs at once. Every single run stays under 2 minutes unless the handoff says otherwise.
- A handoff that expects more than 30 minutes states its budget, and Chat AI tells the Human User when it hands it
  over. It is allowed by default; the Human User intervenes only to refuse it. A handoff may set a lower budget.
- **Safety checks in every turn with GPU runs:**
  - Before the first GPU run and after the last, record the kernel log lines matching
    `amdgpu|gfxhub|GCVM|MES|page fault|segfault|general protection`, `coredumpctl list` since boot, and the boot ID
    (`/proc/sys/kernel/random/boot_id`).
  - **After any pause in a turn** (a usage limit, a wait for the Human User), compare the boot ID before the next GPU
    run. A different boot ID means the machine was restarted: run nothing more, and report it.
  - A new GPU fault line, a crash outside the run, or a reboot ends the handoff.
- Run evidence (logs, JSON) that a report cites goes under `.ai/runs/<iteration>-<name>/`, never at the top level.
- Long-duration tests (soak, long context, capacity sweeps, the 160 tok/s qualification) wait for
  release-candidate state and an explicit request from the Human User.
- Never compare a traced run with an untraced one.
