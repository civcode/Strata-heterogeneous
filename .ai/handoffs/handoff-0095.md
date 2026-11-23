# Handoff 0095

## Tier
1

Why Tier 1: a design study. It has to model the pipelined decode loop across three cards from measured numbers,
judge three layouts against the implementation contract, and design the chosen one well enough to build it later.

## Objective
Answer one question with numbers: **how do we take load off HIP1 and use the NVIDIA card, which is idle 65% of the
time?** (report 0094, hot spots H2 and H3). Compare three candidates on the same data, recommend one, and sketch its
design. **Nothing is built in this handoff**: estimates, short measurements, a design.

## Context
- Read `.ai/PROJECT.md` (the implementation contract: vanilla code wherever possible, interop only inside
  `#if defined(STRATA_MIXED_SPLIT)`; the first rule), `.ai/DIVERGENCE.md` (the front stage, the link protocol,
  pipelined windows, the CPU placement of report 0093), `.ai/STATUS.md`, and reports 0094, 0093 and 0077.
- **Report 0094's numbers** (the Human User's IQ3_XXS maxperf config, `build-0093-*`, a replay of a real pi session):
  - speculative on-path window 14.7 ms; fresh window 24.0 ms; 46.5% of kept windows fresh; a miss costs ~9.3 ms;
  - GPU per window: CUDA front 4.76 ms (15 layers), HIP0 6.73 ms (16 layers), HIP1 9.92 ms (17 layers + the head,
    1.17 ms); the MTP drafter chain runs on HIP1 (`STRATA_MTP_STAGE0=0`), late in 43% of windows;
  - HIP0 waits 4.5 ms per window for the front's ready word; the CUDA card's GPU is busy 35%;
  - prompt chunks across the link: 28% of prompt time; tokens per window 2.27 (thinking) and 2.50 (tool calls).
- **The candidates:**
  - **A. The draft model on the NVIDIA card.** The host keeps everything else; the drafter chain runs on the CUDA
    card between windows (it needs the last hidden state and the head's input; the draft tokens come back). A new
    link message, and the drafter's state (its KV, its window) lives in the front stage.
  - **B. Role swap: the order HIP0 -> HIP1 -> CUDA** (Human User, 2026-11-23). The CUDA engine becomes the vanilla
    host and runs the last layers, the head, the drafter, sampling, acceptance, the caches and the server link; the
    two AMD cards become the front stage, which then runs its own layer split across HIP0 and HIP1. It reuses the
    front-stage mechanism in the other direction. Open points to settle: a front stage on two cards (today one
    process, one card; with pipelining it mirrors stage 0); the VRAM budget of a 16 GB host card (head, drafter about
    1.1 GB, two verify windows, KV of its layers at 256K, an expert cache); the host-side features moving to CUDA
    (conversation and prompt caches, parks; the carried pinned-readback patch is HIP-only); prompt reading then paced
    by which card; the card driving the display (Xorg on the NVIDIA card).
  - **C. Today's layout, configuration only** (layer split, `--spec`, `--spec-min-p`, front layers, reserves): the
    baseline the others have to beat.

## Tasks
1. Safety checks; the first rule; the start/stop rule (only if task 2 runs anything).
2. **Short measurements on the NVIDIA card** (as little as needed): the GPU time of the output head and of the MTP
   drafter chain (per draft step and per window at `--spec 3`) on the RTX 4070 Ti SUPER, for IQ3_XXS; also the time of
   N decode layers there, if report 0094's 4.76 ms for 15 does not suffice. For example a CUDA-only run with the
   experts left mostly to the CPU, traced with `nsys` kernel statistics (the head and drafter kernels do not depend on
   where the experts are); or any cheaper way. Say how each number was obtained and how far it can be trusted.
3. **A model of the pipelined decode loop** for each candidate: per window, each stage's GPU time, the hand-offs and
   link calls, the on-path and fresh window times, and the expected tok/s per phase (thinking, tool calls) at report
   0094's miss rates and tokens per window. State the model, check it against report 0094's measured numbers for
   today's layout first (it must reproduce 14.7 and 24.0 ms within reason), then apply it to A, B and C. Give ranges
   where inputs are uncertain. Include prompt reading (time to first token) and VRAM for each.
4. **What each candidate costs:** the code to write and where (the mixed split's files, the link protocol, vanilla's
   files inside the seam), what it needs from vanilla's code beyond the seam (the contract: name every such place),
   the risks (correctness, the memory-pressure case of `.ai/PROJECT.md`, the pinned-readback patch, the display card),
   and the tests that would prove it.
5. **The recommendation**, and for the recommended candidate a design sketch: components, data flow per window, link
   messages, state ownership, failure handling, the order of the build in handoff-sized steps, and the measurements
   that would confirm each step. If C wins, say which configuration sweep would find its best point.

## Hardware Test Budget
**Up to 45 minutes of GPU run time, over the 30-minute allowance** (announced to the Human User; allowed unless refused).
Runs under 10 minutes each. None at all is fine if the numbers can be had without.

## Constraints
- **First rule: `HSA_ENABLE_SDMA=0` in every run.** All rules of `.ai/PROJECT.md`.
- No changes to the engine, the mixed split's code or the tools. Measurement scripts and wrappers go under
  `.ai/runs/0095-*` or `/home/chris/workspace/strata-local-archive/profiles/0095/`.
- The Human User's run files are theirs: work on copies. Do not start anything while the Human User's server runs
  (check first; ask if it does).

## Completion Criteria
- The model reproduces today's layout; estimates for A, B and C with ranges; costs and contract impact for each;
  a recommendation with a design sketch.
- `.ai/reports/report-0095.md` (and a design file, `.ai/design/drafter-placement.md` or similar, if the sketch is
  long) committed and pushed to `feature/mixed-native`.

## Reporting
- The measurements and how they were taken; the model and its check; the comparison table (tok/s per phase, time to
  first token, VRAM, code size, contract impact, risk) for A, B and C; the recommendation and the design sketch;
  open questions for the Human User. Safety lines and GPU time if anything ran.
