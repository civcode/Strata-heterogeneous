# Report 0094: profiling the recorded pi session (Tier 1). Completed, including the perf profile

No engine or tool code was changed. Every run used `build-0093-*` (`check_launch.py` OK; BUILD_COMMIT `3fb31013`; the perf runs used `build-0094-prof-*`, built from `c9802cc5`, the same engine source
everywhere; only tools and `.ai/` have changed since). Each run used a copy of `strata-iq3_xxs-mixed-maxperf.json`
on a freshly started server on port 8081, launched like the run script, with the safety watchers. Runner, wrappers and
analysis scripts: `.ai/runs/0094-safety/`. Numbers: `.ai/runs/0094-*`. Large traces stay in
`strata-local-archive/profiles/0094/`.

**The config copy against the handoff's list:** everything on the list matches. The file also has `--vision`,
`--vram-reserve-mib 700` (host), `--spec-min-p 0.5`, `--expert-cache auto`, `--prefill auto`, `--kv int8` and the
MTP / expert-profile paths. None of these conflicts with the list. `layer_split: auto` gives HIP0 15-30 and HIP1
31-47; the front runs 0-14.

## 1. The recording (numbers only)

- 107 chat requests. 1 was cancelled by the client: the last one, after 1,085 reasoning chunks, with no usage. No
  request failed otherwise (all status 200).
- The 106 complete requests generated 130,173 tokens in 943 s of decode: **138.0 tok/s**, 75.2% draft acceptance
  (sampled, live).
- Chunks: thinking 36,120, answering 4,636, tool calls 76,061. Dominant phase per request: 53 tool calls,
  49 thinking, 5 answering.
- Prompts ran from 9,529 to 135,682 tokens (median 59,996). 1.07 M prompt tokens were read in 353 s; 6.03 M tokens
  came from the cache.
- **5 compactions**, at requests 38, 68, 82, 88 and 99 (70.8K→24.1K, 74.9K→28.3K, 95.1K→16.0K, 107.1K→9.5K,
  123.7K→13.9K). Each is a single-message, reasoning-off, tool-less request. These are exactly the 5
  answering-dominant requests.
- The session lasted 1,323 s. The pauses between requests add up to only 9.4 s, so a replay takes about as long as
  the generation itself.

**The replays use `--limit 106`.** `replay_session.py` sets `max_tokens` only from recorded usage. The cancelled
#107 has none, so it would decode up to its `max_completion_tokens` of 49,152. This is a tool limitation, not fixed.

## 2. Baseline: three full replays, greedy, recorded pauses

| run | generated | engine decode tok/s | acceptance % | ttft median s | overall (/metrics) | thinking | answering | tool calls | prompt mismatches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline1 | 91,470 | 140.69 | 77.12 | 3.54 | 140.19 | 119.38 | 106.49 | 154.19 | 0 |
| baseline2 | 85,358 | 138.40 | 76.86 | 3.59 | 137.56 | 115.82 | 106.49 | 154.66 | 0 |
| baseline3 | 86,618 | 134.75 | 75.35 | 3.51 | 133.75 | 108.42 | 106.49 | 147.78 | 0 |
| **median** | | **138.4** | **76.86** | 3.54 | **137.56** | **115.82** | **106.49** | **154.19** | |

- **Spread:** engine decode −2.6 / +1.7% around the median, overall −2.8 / +1.9%, thinking −6.4 / +3.1%, tool calls
  −4.2 / +0.3%. Answering is 106.5 in all three runs (it decodes serially; see H5). The three answering values
  differ in their last digits; equal to 2 decimals is a coincidence.
- **Against the Human User:** 142.2 engine decode / 76.8% acceptance, phases thinking 112.5, answering 107.5, tool
  calls 161.1, overall 141.2. Those numbers come from another session (59 requests). This recording measured
  138.0 / 75.2% live. The replays come within 3% on decode; thinking is higher and tool calls lower, and acceptance
  is close.
- **Greedy is not the same work in each run.** The prompts are byte-identical (0 mismatches), but the outputs differ
  between runs: 85,358 to 91,470 generated tokens, and greedy replies stop earlier than the sampled recording's
  130,173. So the spread above includes different content, and A/B tests on this replay need several runs per arm.
  Greedy raises acceptance by about 1.5 points over the live session (76.9 against 75.2%).
- **Two requests ran as two engine calls each** (the second reads 19 new tokens; seen in the timer log). The server
  printed "reached max tokens while still thinking" 4 to 5 times per full replay. The cause of the split was not
  identified. It has no effect on the totals.

## 3. Engine timers (one full replay; tracing cost 7% on the first 40 requests: 128.7 against 137.7 tok/s untraced)

Per-request lines aggregated (`timers.py`; `0094-timers/timers.json`). Each request is classed by its dominant phase
in the stream.

| per window (weighted) | thinking (47 req) | answering (5 req) | tool calls (54 req) | all |
|---|---:|---:|---:|---:|
| engine tok/s in this run | 115.5 | 97.2 | 133.3 | 126.1 |
| mode | pipelined | **serial** (all 1,781 windows) | pipelined | |
| ms per kept window / tokens per window | 19.6 / 2.27 | 23.4 / 2.28 | 18.7 / 2.50 | 19.0 / 2.43 |
| fresh windows (share, ms) | 56.6%, 23.6 | — | 42.4%, 24.2 | 46.5%, 24.0 |
| speculative on the path (ms) | 14.4 | — | 14.7 | 14.7 |
| rolled back / launched speculatively | **48.4%** (3,990/8,247) | — | 32.6% (6,697/20,544) | 37.1% |
| held below the gate | 1,513 | — | 3,461 | 4,974 |
| cost of a miss (fresh − on-path) | 9.1 ms | — | 9.4 ms | 9.3 ms |
| drafter chain late | 35% of windows | — | 47% | 43% |
| GPU per window: CUDA front / HIP0 / HIP1 (head) | 4.69 / 6.62 / 9.68 (1.16) | 4.64 / 6.86 / 8.76 | 4.79 / 6.78 / 10.12 (1.17) | 4.76 / 6.73 / 9.92 (1.17) |
| HIP0's wait for the front's ready word | 4.51 | 4.27 | 4.58 | 4.56 |
| link PlRun / PlCommit / PlUndo (host) | 0.05 / 0.05 / 0.04 | run 0.08, commit 0.04 | 0.05 / 0.04 / 0.03 | |

Outside decode (whole replay): prompt chunks across the link waited 179 × 545 ms = **97.6 s of the 347.8 s of
prompt time**. Session link calls (snapshot, park and the like): 326 × 38 ms = 12.4 s. Conversation parks: 29.5 s.
GPU pauses from the driver: 16.2 s.

**Host time split per window:** not available in pipelined mode. Vanilla's pipelined loop reports the whole window
as verify, with commit, draft and adapt at 0 (adaptive overlap is off). The host's share comes from the CPU samples
(section 4).

**What sets the pace:**
- **Fresh windows** pay all three cards in series: 4.76 + 6.73 + 9.92 = 21.4 GPU ms, plus about 2.6 ms of
  hand-offs and host work. That makes 24.0 ms.
- **Speculative on-path windows** take 14.7 ms. That is more than stage 0 (front + HIP0 in series: 11.5 ms) and more
  than HIP1's window alone (9.9 ms). It matches HIP1's window plus the MTP drafter's chain, which also runs on HIP1
  (`STRATA_MTP_STAGE0=0`). The chain is late in 43% of windows, and the AMD trace below shows HIP1 as the
  busiest card. So HIP1 sets the pace of speculative windows, and the cards in series set the pace of fresh windows.
- **Phases:** thinking is slower than tool calls because of more misses (48 against 33%) and fewer tokens per window.
  Answering is slower because it decodes serially.

## 4. CPU samples (perf, after the Human User enabled it; and the earlier sampler)

**Profiling build:** `build-0094-prof-hip` and `build-0094-prof-cuda` have the same `STRATA_*` options as
`build-0093-*`, plus `RelWithDebInfo`, `-fno-omit-frame-pointer` and `-g`. On the device side `-g` and the frame
pointer go to the host compiler only (`-Xarch_host`, `-Xcompiler`). The build-type flags are kept at **`-O3`**,
because `RelWithDebInfo`'s default `-O2` would change the code being measured. Logs: `0094-build/`.
- **The 3% check** (`--limit 15`, one run each): 139.7 tok/s on `build-0093-*` against 144.2 on the profiling build.
  The profiling build is not slower; content differences are larger than any build difference.
- **perf, full replay:** `perf record -F 499 -g` (frame-pointer call graphs) and `py-spy` on `server.py`.
- **Fix:** the first attempt attached with `-p`. That misses threads started later: the front's window runner and
  the per-run prompt threads. It was redone with perf as the host's `exe` wrapper (`wrap-perf.sh`), which inherits
  every new thread and the front stage it spawns. The analysis is restricted to the replay window. That run decoded
  141.4 tok/s (the `-p` run 143.6), inside the baseline spread.
- **Results:** `0094-perf-inherit/perf-self.txt`, `perf-dso-kernel.txt`, `perf-inclusive.txt`. The first run with
  the LD_PRELOAD sampler (`0094-cpu/`), which unwinds with eh_frame, gives the better inclusive attribution: perf's
  frame-pointer chains break inside `libamdhip64`.

perf, 690K samples over the 957 s replay:

| thread | share of all samples | where |
|---|---:|---|
| host main (decode loop) | 57.2% (about 83% of a core) | User space, polling: `libamdhip64` 27.4%, `libhsa-runtime64` 9.0% (`InterruptSignal::WaitRelaxed` 8.5%), `__tls_get_addr` 7.0% (TLS lookups in `hipSetDevice`, `hipGetDevice` and `hipEventQuery`), vdso clock 3.9%, the engine's own code 4.8% (`Verifier::done`, `main`, the pipeline lambdas). Kernel 0.9%. |
| front WindowRunner | 25.8% | `libcuda` 13.5%, **kernel 6.7% (`sched_yield`)**. Inclusive: prompt `cudaStreamSynchronize` 10.9%, `Verifier::done` → `cudaEventQuery` + `sched_yield` 7.5%. |
| host HSA `AsyncEventsLoop` | 8.1% | ROCr's event thread, all in `libhsa-runtime64` |
| front main (link) | 1.2% | snapshot and checkpoint copies (`memmove`), session operations |
| short-lived prompt threads (host) | ~7.7% | `hipStreamSynchronize` waits |

Kernel time is 8.5% of all samples, mostly from the front's `sched_yield` loop. The AMD driver accounts for 0.02% of
these processes' time. Its userptr-restore workqueues run in kernel workers, which this per-process profile does not
see; that would need system-wide `-a`, which `perf_event_paranoid = 1` does not allow. server.py had 10.9 s of
active Python in this run (11.5 s in the first).

The earlier sampler (eh_frame, `build-0093-*`), for inclusive attribution:

| process / thread | CPU s over 931 s | in polling / wait leaves | where |
|---|---:|---:|---|
| host main | 681 | 88% | `Verifier::done` 54%, prompt 12.5%, `MtpDrafter::chain_poll` + `chain_outputs_ready` 19%, `Verifier::run` 4.9%, snapshot/restore copies ~2% |
| host prompt async threads (176) | 113 | 100% | `hipStreamSynchronize` (27% under `MtpDrafter::prefill`) |
| host pool workers (15) | ~0 | | idle in decode |
| front main (link) | 14 | 51% | `conversation_snapshot_save` 57%, checkpoint copies 13% |

**Both measures agree: there is no CPU work hot spot.** The host's decode thread and the front's window runner
spend their time polling. Each takes most of a core, and about a quarter of the front runner's time is in the kernel
(`sched_yield`). Real host work per window is small. The polling cores matter only under CPU contention (report
0093's case). The cost per poll is about 12% of the host thread in TLS lookups inside the HIP API (`hipSetDevice` /
`hipGetDevice` around every `hipEventQuery`).

## 5. AMD kernels (rocprofv3, `--limit 40`; decode 121.3 tok/s against 137.7 untraced on the same 40 requests: −12%)

`0094-amd/kernels.txt`, `decode.txt`, `hip1-only.txt`. Agent 1 = HIP0, agent 2 = HIP1.

- **Whole run:** both cards are busy 47% of 333 s (kernel union). On HIP0, `wait_handoff_kernel` (the spin on the
  front's ready word) is **53.5 s, 34% of its kernel time**: 12,659 × 4.23 ms.
- **Steady decode** (170 s of 50 ms bins with decode windows and no prompt kernels; 72 HIP0 windows per second):

| | HIP0 | HIP1 |
|---|---:|---:|
| real kernel work | 38.8% | 66.2% |
| spinning on the front (`wait_handoff`) | 30.1% | 0 |
| gaps < 100 µs between kernels (launch bubbles) | 11.2% (3.3 µs × 475 kernels per window) | 14.1% (3.6 µs × 610 per HIP0 window) |
| idle (rest) | ~20% | ~20% |

- **HIP1's extra load:**
  - the 150K-vocabulary Q5_K head: `native_q5_k_mmvq` at 304 µs, 59,787 calls, 18.2 s; plus the multi-row Q5_K
    calls at 182 to 191 µs, about 9 s;
  - kernels that run only on HIP1 (MTP layer, `mtp_select`, sampling): 1.9 ms per HIP0 window, 13.8% of decode
    time;
  - attention: 93 µs per call against HIP0's 48 µs, twice as many calls (the drafter's attention too).
- The launch bubbles are inflated by tracing (−12% decode). Their untraced size is unknown and is an upper bound
  here.
- `__amd_rocclr_copyBuffer` blit copies (from SDMA off) are small in decode.

## 6. NVIDIA kernels (nsys on the front stage, `--limit 40`; decode 133.9 against 137.7 untraced: −3%)

`0094-nvidia/front-gpu.txt`.
- **Decode windows are CUDA graphs:** 21,777 launches. In steady decode (136.5 s) the front runs 76 windows per
  second, 4.61 ms each, and its **GPU is busy 35%**. The median idle gap between windows is 6.6 ms.
  `cudaGraphLaunch` takes 36 µs (median) on the host.
- **Prompt path:** 174,083 kernels, 39.3 s of GPU time for the 40 requests' prompts. The rocprof run, a different
  run of the same requests, shows HIP0 at 32.3 s and HIP1 at 34.0 s of prompt kernels. So the front is the slowest
  prompt stage. The largest items are cutlass f16 GEMM (4.9 s), the fused MoE (`native_kernel<17/16/42>`, 8.9 s) and
  `prompt_attn_i8v3` (2.0 s).
- **API waits:**
  - `cudaStreamSynchronize` 45.7 s in 6,129 calls (the prompt chunks; the WindowRunner spins in it);
  - `cudaEventQuery` 1.77 M calls (polling);
  - `cudaMemcpy` / `cudaMemcpyAsync` 3.3 s.
- **Copies:** device-to-host 48.9 GiB in 8.0 s at **6.1 GiB/s**, host-to-device 20.0 GiB at 6.2 GiB/s. These are the
  conversation snapshots and checkpoints at park, restore and checkpoint time. The rate looks like pageable memory.

## 7. One configuration probe (traced, `--limit 40`, not a fix)

`"layer_split": 33` puts 18 layers on HIP0 and 15 on HIP1 (HIP0 cache 9,216 slots, full). Compared with the timer
run's first 40 requests:
- stage times: HIP0 6.61 → 7.40 ms, HIP1 9.11 → 7.81 ms;
- **speculative window 12.95 → 12.41 ms** (−4%), which supports HIP1 as the pacer of speculative windows;
- but fresh windows 23.6 → 24.3 ms, more misses, a different output (23,733 against 18,536 tokens), and net decode
  128.5 → 122.8.

One sample of each, with the content changed, so this is **inconclusive** as a configuration change.

## 8. The five largest hot spots

Gains are estimated from the timer run's window classes: phase time = fresh windows × fresh ms + on-path windows × on-path ms.

| # | Hot spot | Evidence | Estimated gain | Where a fix belongs |
|---|---|---|---|---|
| H1 | **Wrong guesses: half the windows are fresh.** 46.5% of kept windows run fresh at 24.0 ms against 14.7 ms on the path. 37% of speculative windows are rolled back (thinking 48%, tool calls 33%), and 4,974 are held below the gate. A miss costs about 9.3 ms. | Timers §3 | Half the misses: thinking **+10%**, tool calls **+7.5%**. No fresh windows at all (ceiling): thinking +36%, tool calls +27%. | Vanilla: the pipeline's guess (K+1 is launched on the guess that K is accepted whole) and the drafter's acceptance. An upstream candidate. Mixed-specific part: a fresh window crosses three cards in series (see H3). |
| H2 | **HIP1 carries verify, the MTP drafter chain and the 150K head.** It paces speculative windows. | HIP1 real work 66% + bubbles 14% against HIP0's 39%. Head 18.2 + ~9 s, HIP1-only drafter kernels 1.9 ms per window. Chain late in 43% of windows. Speculative 14.7 ms > HIP1's window of 9.9 ms. Split-33 probe: speculative −0.5 ms. | Chain off HIP1 (about 2-3 ms per window): speculative windows to about 12-12.5 ms, so thinking **~+6%**, tool calls **~+8%**. | Configuration: report 0093 measured the drafter on HIP0 as slower, and layer balance is inconclusive (§7). The seam: a drafter on the CUDA card, which is 65% idle, would be a new design. Vanilla: a cheaper drafter head (upstream). |
| H3 | **Stage 0 is two cards in series.** HIP0 spins 4.2-4.6 ms per window on the front's ready word, and the front is idle between windows. | `wait_handoff` 30% of HIP0's decode time, 34% of its kernel time. Front GPU busy 35% (nsys). Fresh window = 21.4 GPU ms + 2.6 ms of hand-off and host. | Every 1 ms off the serial chain: fresh windows −4%, so thinking ~+3%, tool calls ~+2.3%. Shortening stage 0 helps on-path windows only once H2 is relieved. | The mixed split's seam (how the front's window and HIP0's overlap) and configuration (front layers; the front's VRAM is full at a 450 MiB reserve). |
| H4 | **AMD launch bubbles:** 475 (HIP0) and 610 (HIP1) kernels per window with 3.3-3.6 µs between them. | rocprof §5: 11% and 14% of decode time. Inflated by tracing (−12%). | Upper bound 1.5 / 2.0 ms per window. If a third of that is real: HIP1 −0.7 ms, so on-path windows ~−5%, thinking ~+1.6%, tool calls ~+2.2%. | Vanilla (kernel count, graph dispatch on ROCm): upstream candidate, not ours. |
| H5 | **Prompt and between-request costs** (TTFT, not decode tok/s). The front paces prompt chunks. Parks and snapshots copy slowly. Compactions decode serially. | Prompt-chunk wait 97.6 of 347.8 s of prompt time (28%). Front prompt kernels 39 s against 32-34 s on the HIP cards. Front D2H 6.1 GiB/s; parks 29.5 s, link session calls 12.4 s per replay. Answering (the 5 compactions) runs serial at 97 tok/s: the server adds the model card's `presence_penalty` 1.5 to reasoning-off requests, and vanilla decodes "a request with repetition penalties" serially. | Prompt time up to −28% (TTFT median 3.5 s). Pipelined compactions: answering about +15-25%, which is about +1% overall (4.7% of tokens). | Prompt pacing: configuration (front layers) or the seam (chunk split). Snapshot copies: vanilla (pinned buffers; our carried pinned-readback patch is HIP-only). Serial compactions: vanilla (penalties in pipelined windows) or the config's `sampling` block, which changes outputs (the Human User's choice). |

Not counted as hot spots:
- **CPU:** no work hot spot (§4).
- **GPU pauses from the driver** ("Linux moved memory the GPU uses"): 2.2 to 30.4 s per full replay, **one of 18.8 s**
  (baseline2). This is a system matter (userptr / mmap'd experts); `HSA_USERPTR_FOR_PAGED_MEM=0` is still held in
  reserve.

## Safety

- **The first rule:** `HSA_ENABLE_SDMA=0` was set for every GPU process, including both profiler wrappers. The config
  `env` was checked by `check_launch.py`, and the runner forces it.
- **Boot:** `f94c87ce-2f4b-412e-b02c-09dfdb04f155` before and after. No core dump. No new stop-pattern line. No
  non-finite logits. No process left after any run; every server was stopped cooperatively (rc 0).
- **Before this turn:** this boot has a **gfxhub page fault at 12:31:44 from client SDMA1, strata PID 9157**, 6 hours
  before this turn's first run, from a process this turn did not start. It is recorded here because SDMA should be
  off everywhere; the Human User may want to know which run it was.
- **Record lines during the turn:** 20:11:16 `amdgpu_amdkfd_restore_userptr_worker` hogged CPU 259 times, and
  20:11:22 `svm_range_restore_work` 515 times (both during the rocprof run). Engine GPU-pause lines: 108 across all
  runs. Per run: count, total, largest:
  - baseline1: 22, 15.8 s, 1.7 s;
  - **baseline2: 15, 30.4 s, 18.8 s**;
  - baseline3: 4, 2.2 s, 1.2 s;
  - timers: 20, 16.2 s, 2.0 s;
  - cpu: 3, 1.8 s, 0.9 s;
  - amd: 20, 18.3 s, 5.5 s;
  - nvidia: 19, 14.3 s, 3.1 s;
  - split33: 5, 4.3 s, 1.6 s.
- **Swap:** 540 MiB before, 891 MiB after the first part (16,383 total), 728 MiB at the end. Baseline2 grew it
  from 503 to 847 MiB.
- **After perf was enabled:** check15 runs 3 and 1 pauses (largest 0.9 and 1.7 s); `perf` 15 (10.8 s, largest
  1.7 s); `perf-inherit` 26 (12.3 s, largest 2.1 s). No new record or stop lines after 20:33 (`after-perf.txt`),
  same boot, no core dumps.
- **GPU time:** 102.6 min in the first part, plus 35.9 min after perf was enabled (2 short checks and 2 full perf
  replays): **138.5 min** in total. That is within the approved 2.5 hours; the Human User then allowed more. Runs
  were sequential. The full replays took 16 to 18 min each.

`py-spy` was installed into the checkout's `.venv` (pip). The sampler's `.so` is not committed: build it with the
`gcc` line in its source header. The Human User set `kernel.perf_event_paranoid=1`, `kptr_restrict=0` and
`yama.ptrace_scope=0` during the turn. perf and py-spy attaching were checked on test processes before use.
