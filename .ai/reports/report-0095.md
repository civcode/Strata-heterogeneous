# Report 0095: unloading HIP1 / using the NVIDIA card (Tier 1 design study). Completed

**Short answer:** neither A (drafter on the NVIDIA card) nor B (role swap) is worth building now. HIP1 does not pace
speculative windows; the serial sum of the drafter's chain, stage 0 and HIP1 does. The biggest loss found is a
**state flip on HIP1** that slows the chain after the first compaction and lasts for the rest of the session (80% of
report 0094's windows). The ROCm setting **`GPU_MAX_HW_QUEUES=2`** removed it in two short runs, with no code change.
**Recommendation: C**, starting with a queue-limit sweep and a long run. A stays the fallback; its design is in
`.ai/design/drafter-placement.md`. Nothing in the engine, the mixed split or the tools was changed.

## Measurements (all `build-0093-*`, copies of the maxperf config, port 8081, sequential)

| run | what | how far to trust it |
|---|---|---|
| `0095-cuda-only` | CUDA-only engine on the RTX 4070 Ti SUPER, serial, `--spec 3`, three short requests; nsys (graph level), `STRATA_VERIFY_PROFILE`, `STRATA_DECODE_TIMING` | head **0.77 ms** per window (GPU events; HIP1: 1.16-1.18). Drafter GPU per graph: round 0.77 ms, one draft step **0.69 ms** (nsys graph trace, 277 windows). Serial draft host time 1.83-1.93 ms per window at avg T 2.58 (R9700 serial, report 0093: 1.68 ms at T 2.35): the 4070 drafts **no faster** than the R9700. Layers with resident experts: ~0.28-0.30 ms each (the front: 0.30; HIP0 0.40) |
| `0095-trace` | first 15 recorded requests, `STRATA_PIPELINE_TRACE` (host clock), timers | the model's calibration; 131.7 tok/s; on-path 12.27, fresh 23.26 ms |
| `0095-stage0` | same, `STRATA_MTP_STAGE0=1` (drafter on HIP0) | chain beside HIP0's window: stage 0 +1.57 ms on speculative windows; 117.2 tok/s. One 18.8 s driver pause in this run |
| `0095-tracelong` | requests 89-97 (107-122K context), fresh server | on-path 12.94, fresh 24.64: context alone does not give 0094's 14.7 |
| `0095-compaction` | requests 36-41 (compaction at 38), fresh server | reproduces the flip: on-path 12.0 / 11.2 before, 15.1 / 15.5 / 15.5 after |
| `0095-compaction-hiponly` | same, HIP-only (vanilla + carried patch) | flips too, the other way: 18.6 / 18.0 before, 13.4 / 13.3 / 13.3 after |
| `0095-compaction-hwq16` | `GPU_MAX_HW_QUEUES=16` | no flip, but on-path 14.0-14.8 throughout (chain 7.9 ms); fresh 21.7-22.1 |
| `0095-compaction-hwq2` (+ `-repeat`) | `GPU_MAX_HW_QUEUES=2` | no flip: on-path **11.0-11.6** in every request, fresh 21.3-21.8; after the compaction 132.9 / 142.7 / 152.3 and 147.3 / 145.1 / 153.0 tok/s against default 116.2 / 125.7 / 128.5 |
| `0095-trace-hwq2` | first 15 requests, `=2` | same as default in the good state: 131.2 tok/s, on-path 12.27, fresh 23.35, prompt 47.4 s (default 47.7) |

Outputs differ between runs (greedy content varies, report 0094); the compaction runs after request 38 happen to
generate the same token counts with `=16` and `=2` (494 / 595 / 773), default 494 / 575 / 639. All hwq numbers are
one or two short runs: **a lead, not a qualified setting.**

**Trace intervals** (`pltrace_stats.py`, 15-request run): chain to next decision c1 = 4.9 ms; fresh decision
c2a = 1.9 ms; stage 0 (front + HIP0) 11.1 ms; HIP1 9.35 ms; HIP1 idles **2.9 ms per on-path window** waiting for
stage 0. In the bad state: c1 7.4-8.0 ms (p90 10.7), HIP1 window 10.6-11.1 ms (p90 15). With `=2`: c1 4.3-5.0,
HIP1 8.4-8.9, stage 0 10.1-10.3 ms.

## The model and its check

`.ai/runs/0095-safety/plmodel.py` replays the real verdict sequence with vanilla's rule of two windows in flight
(window n+2 is made by the chain at window n's verdict): on the path the period is ~(c1 + S0 + s1)/2, a fresh
window is ~c2a + S0 + s1 plus the wait for a doomed window. Check: 15-request run 12.27 / 22.67 ms against measured
12.27 / 23.26; drafter-on-HIP0 run 13.31 / 22.9 against 13.43 / 24.0. Report 0094's 14.7 / 24.0 is a mix: its own
logs give 12.50 / 23.58 ms before the first compaction (requests 1-39, 20% of windows) and 15.13 / 24.07 after (80%).
The model, fed means, gives 13.9 for the bad state against measured 15.1-15.5 (coinciding tails); it understates
whatever removes the bad state.

## Comparison (per phase with report 0094's class mix and tokens per window)

| | on-path / fresh ms | thinking tok/s | tool calls tok/s | time to first token | VRAM | code | contract | risk |
|---|---|---:|---:|---|---|---|---|---|
| today, good state | 12.27 / 23.26 | 122.6 | 147.6 | | | | | |
| today, bad state (80% of 0094) | 15.3 / 23.6 | 113.4 | 132.8 | | | | | |
| **A** drafter on CUDA (model) | 11.7-12.1 / 23.3-25.0 | 117-124 | 142-150 | +2 to +6% | front +1.05 GiB: a front layer less or a short cache | ~1.5-2.5K lines, link protocol 5 | `mtp.cpp/hpp` remote mode, `generate.cpp` (~10 chain, 6 prefill, ~19 `kv_state` sites), `prefill.cpp` `draft_kv` | drafter KV across processes in snapshots/parks; fresh windows slower (chain tail on the front card) |
| **B** role swap (model) | 14.9-15.9 / 24.1-25.4 | 107-113 | 126-133 | −10 to +5% | CUDA host fits ~12-13 layers | ~3-5K lines | PROJECT.md rule 7 (host must be HIP), check_launch; all host features on CUDA | pinned-readback patch is HIP-only (CUDA snapshots at 6 GiB/s pageable); host on the display card (Xorg uses 4 MiB there) |
| **C** `GPU_MAX_HW_QUEUES=2` (measured, short) | 11.5-12.3 / 21.4-23.4 | 122-133 | 147-159 | unchanged | none | none | none (runtime env) | queue sharing with spin-wait kernels; needs a long run |

A gains in the bad state (+3 to +9% thinking, +7 to +13% tool calls) but loses up to 5% in the good one; if C holds,
A adds about nothing. B loses: as host the CUDA card holds fewer layers than today, so the serial sum grows.

## Recommendation and next steps (design: `.ai/design/drafter-placement.md`)

1. Sweep `GPU_MAX_HW_QUEUES` = 1 / 2 / 3 / 4: three untraced full replays per arm plus one traced compaction replay.
   Pass: no flip, on-path ≤ 12.5 ms after a compaction, prompt unchanged, no hang.
2. RC long run of the chosen value (fewer queues change concurrency for prompt streams, conversation copies and the
   vision helper).
3. Then `--spec 2/3/4` x `--spec-min-p 0.3/0.5/0.7` (one draft step ≈ 0.7 ms of c1, counted half per on-path window).
4. Upstream candidate: the flip also happens on the HIP-only build.

## Open questions for the Human User

- May the sweep and a long run of `GPU_MAX_HW_QUEUES` go into the next handoff? Full replays take 16-18 min each.
- Which streams share HIP1's queues in the bad state is not known. Finding out needs a rocprof queue trace: useful
  for the upstream report, not needed for the setting.

## Safety

- **First rule:** `HSA_ENABLE_SDMA=0` in every GPU process (runner forces it, `check_launch.py` on the mixed and
  HIP-only configs; the CUDA-only config set it in `env` and the shell; the nsys wrapper exports it).
- **Boot** `f94c87ce-2f4b-412e-b02c-09dfdb04f155` before (after the usage-limit pause) and after. No core dumps. **No
  new record or stop kernel lines** between 22:30 and 22:57 (`0095-safety/before.txt`, `after.txt`); the gfxhub
  fault at 12:31 predates this turn (reported in 0094). No non-finite logits. Every server stopped cooperatively (rc 0); no leftovers.
- **Driver pause lines:** trace 6 (largest 1.6 s), stage0 5 (**18.8 s**), tracelong 8 (1.6 s), compaction 1, hwq16 3,
  hwq2 1, trace-hwq2 1, hiponly 0, hwq2-repeat 0, cuda-only 0.
- **Swap:** 682 MiB before, 865 MiB after (16,383 total).
- **GPU time:** 10 runs, 945 s server lifetime in total (**15.8 min**, each under 3 min), of the 45 approved.
