# Project rules

Read this file at the start of every turn. The collaboration protocol is `.ai/WORKFLOW.md`.

> **FIRST RULE — `HSA_ENABLE_SDMA=0` in EVERY run, no exceptions.** SDMA has a known bug in the AMD driver. Every
> process that touches a GPU runs with `HSA_ENABLE_SDMA=0`: servers, engines, the front stage, vision helpers, test
> binaries and CTest (`mixed_split/tools/ctest_sdma_off.sh`), probes and one-off scripts. Every run config's `"env"`
> has `"HSA_ENABLE_SDMA": "0"`; `check_launch.py` refuses a config without it, and the tools refuse any other value in
> the shell. A run without it is invalid: its results are discarded and the report says so. Every report confirms it.

## What this project is

Strata running one model across an NVIDIA card (CUDA) and AMD cards (HIP) at the same time. The fork is
`civcode/Strata-heterogeneous`; vanilla is `Niko1221/Strata`.

## Implementation contract

1. Use vanilla Strata code wherever possible. Add code only for CUDA/HIP interoperability.
2. A change to a vanilla file must be a behaviour-preserving move or extraction, or the minimal hook the CUDA/HIP
   seam needs. No new scheduling, speculation, prefill, cache, sampling or synchronisation logic.
3. Builds without the mixed-split option behave exactly as vanilla.
4. HIP-only (two AMD cards) is plain vanilla Strata with a native layer split. It uses no mixed-split code.
5. Numerical differences between CUDA and HIP are low priority.
6. One labelled divergence from vanilla is carried (Human User, 2026-11-19, after the 19 November crash): private
   commit `05134f54`, the HIP pinned snapshot readback and conversation-restore preparation (`.ai/HARDENING.md`). It
   applies to every build, HIP-only included, with or without the mixed-split option. "Vanilla" in points 3 and 4 and
   in `check_off_is_vanilla.py` means vanilla 0.2.15 plus that commit: branch `base/vanilla-0.2.15-pinned-readback`.
   The rest of the 2026-11-17 hardening stays withdrawn (`archive/hardened-0069`).
7. Every process in a launch config runs from one build directory, with that build's libraries. A mixed config names
   exactly two builds, one HIP build for the host and one CUDA build for the front stage, and both are built from
   the same commit of this branch. Check before a run (`mixed_split/tools/check_launch.py`, handoff 0070).

The target design and the plan are in `.ai/SEAM.md` and the roadmap referenced from `.ai/STATUS.md`.

## References

| What | Revision |
| --- | --- |
| Vanilla reference and base of this branch | `c77572806996349b49995e4f98d41b5294d36a21` (0.2.15). Production stays on 0.2.15 with MMQ (Human User, 2026-11-19) |
| Re-pin to 0.2.16, taken back | reports 0083 and 0084, archived as `archive/mixed-native-0.2.16`: vanilla 0.2.16 removed the MMQ prompt kernels, and its prefill was 15 to 20% slower on HIP-only (report 0084) |
| Working branch | `feature/mixed-native` |
| Frozen earlier adapter (not developed, kept as fallback) | branch `feature/heterogeneous-hip-only`, archived as `archive/heterogeneous-adapter-0062` at `0dca627f` |
| Earlier local vanilla pin on Oberon | `8db7ddc4`: a private work-in-progress branch, not pushed (report 0063) |

Handoff and report numbers continue from the earlier branch (0062 was its last).

## Layout (vanilla 0.2.15)

- The engine is at the top level (`src`, `include`, `CMakeLists.txt`); configure with `cmake -S . -B <build dir>`. The
  server is `serve/server.py`; vanilla's tools are `tools/` (`calibrate.py`, `iq_pack.py`, `strata_pack.py`, the vision
  helpers in `tools/vision/`).
- The mixed split's own subtree is `mixed_split/` (library, tests, tools, configs); the top-level `CMakeLists.txt`
  takes it in behind `STRATA_MIXED_SPLIT`. Its code lives in the namespace `strata::mixed_split`, and its headers are
  under `strata/mixed_split/`. The name was `hetero/` / `strata::hetero` until 2026-11-19; history in `.ai/` keeps the
  old name.
- Check the option-off build with `python3 mixed_split/tools/check_off_is_vanilla.py` (default reference `c775728`).

## Environment (Oberon)

- Cards: one NVIDIA RTX 4070 Ti SUPER (16 GB, CUDA, sm_89) and two AMD Radeon AI PRO R9700 (HIP, gfx1201).
- HIP toolchain: ROCm at `/opt/rocm/core-10.1` (`ROCM_PATH`, `HIP_PLATFORM=amd`), Ninja, `-DCMAKE_HIP_ARCHITECTURES=gfx1201
  -DSTRATA_PREFILL_MMQ=ON -DSTRATA_MMQ=ON -DCMAKE_BUILD_RPATH=/opt/rocm/lib`. The MMQ prompt kernels need both MMQ
  options; report 0066 measured about 22% faster prefill with them (2641 against 2163 tok/s, one sample).
- CUDA toolchain: CUDA 13.0 at `/usr/local/cuda-13.0` (vanilla 0.2.15 is built with CUDA 13.0; driver 580.178.04).
  Configure the CUDA tree with `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=89`.
  `/usr/local/cuda` still points at CUDA 12.0, which the frozen branch was built with: leave it. nvcc 12.0 cannot
  compile vanilla 0.2.15 (report 0063). Do not change vanilla sources to fit a toolkit.
- **Production setup** (the Human User, 2026-11-18): `mixed_split/configs/hiponly.example.json` mirrors it, with
  placeholders for the paths. Port 8081, `parallel` 1, `layer_split: auto`, two R9700.
  - Model: Qwen3.8-Flash-Next GSQ-RCO Q2_0, `--mmap-experts`, `--expert-cache auto`, `--prefill auto`.
  - Drafting: MTP with `--spec 4 --spec-min-p 0.5`, `--pipeline-windows 2`.
  - Context and KV: `--max-context 262144`, `--kv int8`, `--kv-resident 0`.
  - Vision on (GPU encoder, `max_tokens` 1024); a scaled control vector over layers 4 to 44 (project, per-layer).
  - Caches: `--prompt-cache 6 --prompt-cache-spread`, `--conversation-cache-slots 4`,
    `--conversation-cache-min-tokens 8192`.
  - Environment: `HSA_ENABLE_SDMA=0`, `STRATA_SPLIT_OVERLAP=1`, `STRATA_PIPELINE_AUTO=0`, `STRATA_MTP_STAGE0=0`.
    `STRATA_SPLIT_OVERLAP=1` has no effect next to `--pipeline-windows 2` (vanilla turns it off; report 0084).
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

- Stage: **release candidate** (Human User, 2026-11-19). The qualification is defined in `.ai/RC.md`.
- Hardware test budget: **up to 30 minutes of GPU run time per handoff** (Human User, 2026-11-18), short prompts,
  runs sequential, never two GPU jobs at once. Every single run stays under 2 minutes unless the handoff says otherwise.
- A handoff that expects more than 30 minutes states its budget, and Chat AI tells the Human User when it hands it
  over. It is allowed by default; the Human User intervenes only to refuse it. A handoff may set a lower budget.
- **Safety checks in every turn with GPU runs:**
  - Before the first GPU run and after the last, record the kernel log lines matching the record patterns below,
    `coredumpctl list` since boot, and the boot ID (`/proc/sys/kernel/random/boot_id`).
  - **Two kinds of kernel line** (report 0089 stopped on a warning because one pattern, `amdgpu`, matched both):
    - **Stop patterns** (a fault, corruption or hang): `gfxhub|GCVM|page fault|segfault|general protection|Bad page|
      Bad swap|Oops|BUG:|rcu.*stall|amdgpu.*(fault|reset|timeout|hang)|ring .* timeout|MES failed|Xid|NVRM`. A new
      line ends the run cooperatively at once, and the handoff.
    - **Record patterns** (warnings, the crash case's early signs): everything the stop patterns match, plus
      `amdgpu|workqueue:.*\[amdgpu\]|restore_userptr_worker|svm_range_restore_work|MES`. Recorded with their times and
      reported; they do not stop a run. The engine's `paused the GPU` lines are recorded the same way.
  - The first rule (top of this file): `HSA_ENABLE_SDMA=0`. Each report confirms it, and records swap use (`free -m`)
    before and after.
  - **After any pause in a turn** (a usage limit, a wait for the Human User), compare the boot ID before the next GPU
    run. A different boot ID means the machine was restarted: run nothing more, and report it.
  - A new GPU fault line, a crash outside the run, or a reboot ends the handoff.
- **Starting and stopping GPU processes** (report 0087: a session's end hung because the server ignored SIGINT):
  - A command started in the background by a shell (`cmd &`, `nohup`, a tool runner's background job) inherits SIGINT
    ignored, and so does every process it starts; the server's cooperative stop then does nothing. Start servers and
    engines only through the tools in `mixed_split/tools/` (they reset that first: `reset_ignored_signals`), or from a
    Python script that calls `mixed_split_common.reset_ignored_signals()` before starting them. Never start a server
    or engine directly with `&` or `nohup`.
  - Stop them with `mixed_split_common.stop_cooperatively`: SIGINT and the drain, then SIGTERM. Never SIGKILL, and never
    a `subprocess` timeout that kills (`subprocess.call/run(timeout=...)` sends SIGKILL to the child when it expires).
  - Every command that runs a GPU job returns when the job ends: run it in the foreground with a timeout longer than
    the job, not as a background job you poll. After it returns, `pgrep -af 'strata|server.py'` must show nothing
    of the run before the next GPU job starts.
  - A command that seems to hang: check the processes first (`pgrep -af`, `ps -o pid,stat,cmd`). Stop what is left
    cooperatively as above. If something still runs after SIGINT and SIGTERM, start no further GPU job and report it.
- Run evidence (logs, JSON) that a report cites goes under `.ai/runs/<iteration>-<name>/`, never at the top level.
- Long-duration tests (soak, long context, capacity sweeps) are allowed again for the release-candidate qualification
  (`.ai/RC.md`; Human User, 2026-11-19, after the 19 November crash), with each long run's length and the turn's total
  budget in the handoff. Swap stays as configured (on).
- **The 19 November crash case** (documented, Human User's decision to keep swap on): the host crashed in a long mixed
  session at a 214K-token context, while Linux was swapping (about 8 GB written out, about 39 GB still available) and
  the AMD driver kept restoring GPU-mapped host memory. Its signs, in the order they appeared (the first three are
  warnings, recorded; the last two are stop lines):
  - kernel: `workqueue: amdgpu_amdkfd_restore_userptr_worker ... hogged CPU` and `svm_range_restore_work ... hogged CPU`;
  - engine: `the graphics driver paused the GPU for N ms ... while Linux moved memory the GPU uses` (up to 23 s);
  - swap use growing during the run;
  - engine: `non-finite logits` after a conversation restore;
  - kernel: `BUG: Bad page map`, an `Oops`, RCU stalls; the boot ended uncleanly.

  Every long run records the first three (the kernel lines, the engine's pause lines, swap use from the sampler) and
  reports them with their times. A `non-finite logits` line, or a kernel line matching the stop patterns, ends the run
  cooperatively at once. The carried pinned-readback patch (`.ai/HARDENING.md`) is this project's response;
  `HSA_USERPTR_FOR_PAGED_MEM=0` (vanilla's `docs/AMD_HIP.md`) is held in reserve.
- Never compare a traced run with an untraced one.
