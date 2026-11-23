# Handoff 0094

## Tier
1

Why Tier 1: the result is an analysis. Hot spots have to be read from CPU samples, GPU kernel traces and the
engine's own timers together, and judged as to where a fix would belong (the mixed split's code, or vanilla's).

## Objective
Profile the Human User's real pi agent workload on their optimal mixed-split configuration and find the
bottlenecks and hot spots, with the exact same requests in every run. **No code changes in this handoff**: it
measures and explains; fixes come in later handoffs.

## Context
- Read `.ai/PROJECT.md` (the first rule; starting and stopping GPU processes; stop and record patterns),
  `.ai/DIVERGENCE.md`, `.ai/STATUS.md`, reports 0093 and 0092.
- **The configuration** (Human User, 2026-11-23, tuned with 15-minute real pi sessions): the IQ3_XXS mixed config,
  build `build-0093-*`, with `--spec 3`, `--mtp-window 16384`, `--pipeline-windows 2`, `--front-layers 15`,
  `--front-arg --vram-reserve-mib --front-arg 450`, 256K context, `--kv-resident 0`, `--mmap-experts`, env
  `STRATA_SPLIT_OVERLAP=1`, `STRATA_PIPELINE_AUTO=0`, `STRATA_MTP_STAGE0=0`, `HSA_ENABLE_SDMA=0`,
  `"layer_split": "auto"`. It is in this checkout as `strata-iq3_xxs-mixed-maxperf.json` (the Human User's file:
  copy it, do not edit it). Check the copy's `args` and `env` against this list and report any difference.
- **The Human User's results with it:** 142.2 tok/s engine decode, 76.8% draft acceptance (59 requests, 80,104
  tokens); phase table (`phase_tps.py`, the Human User's measure): thinking 112.5, answering 107.5, writing tool
  calls 161.1, overall 141.2 tok/s. Another session: 131.6 overall.
- **The recording:** `/home/chris/workspace/strata-local-archive/recordings/session-20261123T170506Z.jsonl`,
  107 requests through `record_proxy.py` (all methods). It holds the Human User's prompts, tool output and code:
  **it never goes into the repository**, and nothing derived from its text does either (the replay outputs hold
  numbers only).
- **The tools** (`mixed_split/tools/`, read their docstrings): `replay_session.py run|summary` (greedy and with the
  recorded pauses by default; `max_tokens` = what each request generated when recorded; checks each prompt's size
  against the recording; runs `phase_tps.py` beside it and writes the phase table and `phase_tps.html`),
  `phase_tps.py` (the Human User's `/metrics` benchmark, unchanged logic).

## Tasks
1. Safety checks; the first rule; the start/stop rule. `git pull`; the builds `build-0093-*` are current for the
   engine (only tools and `.ai/` changed since `39bb4deb`): confirm with `check_launch.py`.
2. **The recording:** count its chat requests, tokens generated, cancelled and failed requests, and compactions if
   recognisable (a request whose prompt is much shorter than the one before). Numbers only in the report.
3. **Baseline:** three replays of the whole recording, each on a freshly started server (the copy of the config,
   started like the run script: `serve/server.py --engine strata --config ... --port 8081`, from a Python parent
   that calls `reset_ignored_signals()`, stopped with `stop_cooperatively`). `replay_session.py summary` over the
   three. Report the spread, and compare with the Human User's numbers (greedy changes acceptance a little: say so).
   Every prompt's size must match the recording (`prompt mismatches` 0).
4. **The engine's own timers** (one replay): `STRATA_DECODE_TIMING=1`, `STRATA_VERIFY_PROFILE=1`,
   `STRATA_SPLIT_TIMING=1` in the copy's `env`. Aggregate the per-request lines into one table per phase
   (requests classified by their dominant phase from the replay's chunk counts): per window, host wall time
   (adapt wait, verify, commit/emit, draft, other), each card's GPU stage time (CUDA front, HIP0, HIP1), the front
   window, the link calls, the pipelined hit/miss counts and the cost of a miss. Which card or step sets the pace,
   per phase?
5. **CPU sampling** (one replay): a profiling build, `build-0094-prof-hip` and `build-0094-prof-cuda`, the same
   options as `build-0093-*` plus `-DCMAKE_BUILD_TYPE=RelWithDebInfo` and `-fno-omit-frame-pointer` (C, C++ and the
   device compilers' host side); first check that a short replay (`--limit 15`) on it decodes within 3% of
   `build-0093-*`, otherwise profile `build-0093-*` with `perf record --call-graph dwarf` instead. Then
   `perf record -F 499 -g` attached to the host engine and the front stage for the whole replay, and `py-spy record`
   on `server.py`. Report per process and per thread (host thread, pool workers, front runner, link, server): the
   top functions by self and by inclusive time, spin/wait versus work.
6. **AMD GPU kernels** (one replay, `--limit 40`): `rocprofv3 --kernel-trace --stats` through a wrapper script used
   as `exe` in another copy of the config. Per card: the top kernels by total time, kernels per decode window, idle
   gaps. Report the replay's own decode speed beside it (the profiler's overhead).
7. **NVIDIA kernels** (one replay, `--limit 40`): `nsys profile -t cuda,osrt --sample=none` through a wrapper used as
   `--front-exe`. The top kernels, the CUDA API calls that wait (synchronisation, copies), idle gaps.
8. **The analysis:** the five largest hot spots or bottlenecks, each with its evidence (which runs and numbers), an
   estimate of what removing it could gain in tok/s per phase, and where a fix would belong: the mixed split's own
   code (inside the seam), its configuration, or vanilla's code (an upstream candidate, not ours to change).

## Hardware Test Budget
**About 2.5 hours of GPU run time, over the 30-minute allowance** (announced to the Human User; allowed unless
refused). Each full replay takes about the recording's length (about 15-20 minutes with pauses). Order: tasks 3, 4,
5, 6, 7. If time runs out, stop cooperatively and report what was measured.

## Constraints
- **First rule: `HSA_ENABLE_SDMA=0` in every run**, profiler wrappers included. All rules of `.ai/PROJECT.md`.
- No engine or tool code changes (a profiling build and wrapper scripts are not code changes). A tool bug that
  blocks the work: the smallest fix, said in the report.
- Large traces and profiles stay local: `/home/chris/workspace/strata-local-archive/profiles/0094/`. Commit the
  summaries, tables, `phase_tps.html` files and the replay summaries under `.ai/runs/0094-*`.
- The Human User's run files are theirs: work on copies.

## Completion Criteria
- Baseline, timers and at least the CPU profile done; the GPU kernel profiles done or the reason they are not.
- The hot-spot analysis in the report.
- `.ai/reports/report-0094.md` and all changes committed and pushed to `feature/mixed-native`.

## Reporting
- The recording's numbers; the baseline table and spread; the per-phase timer table; the CPU, AMD and NVIDIA
  findings; the five hot spots with evidence, estimated gains and where each fix belongs.
- Safety lines, swap, GPU time.
