# Drafter placement and the idle NVIDIA card (handoff 0095)

Status: design study. Nothing here is built. Numbers: `.ai/reports/report-0095.md`; model: `.ai/runs/0095-safety/plmodel.py`.

## 1. The model of the pipelined loop (`--pipeline-windows 2`)

Vanilla keeps at most two windows in flight. Window n+2 is made by the drafter's chain that is launched at the verdict
of window n. So, with S0 the stage-0 path (front stage + HIP0, launch to done, host view), s1 the last stage (HIP1 with
the head), c1 the chain's time to the next window's decision and c2a its time to a fresh window's decision:

    on the path:  period ~ (c1 + S0 + h + s1) / 2      (if that exceeds s1 + delta)
    fresh:        period ~ c2a + S0 + h + s1  (+ the wait for a doomed window to leave stage 0)

On the path, HIP1 waits for stage 0 (measured 2.9 ms per on-path window), so **HIP1 does not pace speculative windows;
the serial sum c1 + S0 + s1 does**, and every millisecond of c1, S0 or s1 is worth half a millisecond per on-path
window and a full one per fresh window. `plmodel.py` replays the real verdict sequence of a trace with these rules.

Check against the 15-request trace (default config, "good" state): model 12.27 ms on the path (measured 12.27),
HIP1 idle on the path 2.92 ms (measured 2.92), fresh 22.7 ms (measured 23.3: −2.5%). Drafter on HIP0
(`STRATA_MTP_STAGE0=1`) with its own measured parts: 13.31 / 22.9 (measured 13.43 / 24.0). In the "bad" state (below)
the model, fed mean values, gives 13.9 against the measured 15.1 to 15.5: the long tails of the chain and of HIP1's
window coincide there, which means cannot show.

Report 0094's 14.7 ms on the path is a mix of two states: 12.5 ms before the session's first compaction
(requests 1-39, 20% of the windows) and 15.1 ms after it (80%).

## 2. The "bad" state: HIP1's hardware queues

After the first serially decoded request (a compaction: penalties force the serial loop), the chain's time to the
next window grows from ~4.8 to ~7.4-8.0 ms (p90 4.5 -> 10.7 ms) and HIP1's window gets a tail (p90 10 -> 15 ms); fresh
windows are unchanged. It stays for the rest of the session. Reproduced on six requests (seq 36-41) of the recording.
On the HIP-only build the state also flips, the other way round (18.5 ms before, 13.3 after).

With `GPU_MAX_HW_QUEUES=2` (ROCm runtime, default 4; the limit applies per device, per process and per priority pool;
`docs` of ROCm's environment variables) the flip is gone: c1 4.3-5.0 ms, HIP1's window 8.4-8.9 ms and stage 0
10.1-10.3 ms in every request, before and after the compaction, in two runs. `=16` also removes the flip but keeps
the slow chain (7.9 ms). Reading: which streams share a hardware queue on HIP1 decides whether the chain and the last
stage's window contend; it is not yet known which streams. This is a configuration of the ROCm runtime, not code.

## 3. Candidates

| | A: drafter on the NVIDIA card | B: role swap HIP0 -> HIP1 -> CUDA | C: configuration |
|---|---|---|---|
| On the path (good state 12.27 ms) | 11.7-12.1 ms | 14.9-15.9 ms | 11.5-12.3 ms (`GPU_MAX_HW_QUEUES=2`) |
| Fresh (23.3 ms) | 23.3-25.0 ms | 24.1-25.4 ms | 21.4-23.4 ms |
| Thinking / tool calls tok/s, against good 122.6 / 147.6 and bad 113.4 / 132.8 | 117-124 / 142-150 | 107-113 / 126-133 | 122-133 / 147-159 |
| Prompt (time to first token) | +2 to +6% (drafter prompt pass on the front card, which paces prompts) | −10 to +5% (unclear whether the AMD front pipelines its two cards per chunk) | unchanged (measured: 47.4 against 47.7 s) |
| VRAM | front card +1.05 GiB (drafter 815 MiB, draft head 186 MiB, buffers): one front layer less, or a short expert cache | CUDA host 16 GB: head 0.43 GiB, drafter 1.05, prompt buffers 2.86, second verifier, KV: ~12-13 layers fit | none |
| Code | ~1.5-2.5K lines; link protocol 5 | ~3-5K lines; front stage on two cards; check_launch and PROJECT.md rule 7 change | none |
| Risk | the drafter's KV in snapshots/parks/restores across processes; the front's expert cache; serial loop's drafts | everything host-side moves to CUDA; pinned-readback patch is HIP-only; display card | a queue limit with spin-wait kernels: needs a long run |

Why B loses: the CUDA card holds fewer layers than today as the host (13 against 15), so the serial sum grows; the
drafter stays on the last card beside its window. Why A barely wins in the good state: the chain is no faster on
the 4070 Ti SUPER (0.69 ms per draft step, measured) than on the R9700; A only removes contention, and adds a hop and
contention on the front card for fresh windows (the fresh chain runs ~3.6 ms past the fresh window's launch).

## 4. Recommendation: C

1. **`GPU_MAX_HW_QUEUES` sweep** (1, 2, 3, 4), the Human User's config `env` only. Per arm: three full replays of the
   recording (content varies run to run, report 0094), untraced; and one traced six-request compaction replay
   (`session-seq36-41`) to see c1 and HIP1's window. Pass: no flip, on-path windows ≤ 12.5 ms after a compaction,
   no prompt loss, no hang (a spin-wait kernel sharing a queue with the work it waits for would hang until the
   30 s bounded wait fires; watch for `the window never finished`-type errors and kernel lines).
2. Then a long run of the chosen setting (RC soak, `.ai/RC.md`), because fewer queues change concurrency everywhere
   (prompt streams, conversation copies, the vision helper).
3. Then `--spec 2/3/4` x `--spec-min-p 0.3/0.5/0.7` on the chosen queue setting (three replays per arm): the chain's
   length sets c1 (one draft step ~0.7 ms), which counts half on every on-path window.
4. Upstream: the flip also happens on the HIP-only build (vanilla + the carried patch): a vanilla/ROCm report.

## 5. A, kept as the fallback (if no queue setting holds the good state)

- **Components:** in the front process, vanilla's `MtpDrafter` loaded on the CUDA card (as `mtp_first` puts it on
  stage 0's card in one process). On the host, a remote drafter inside `MtpDrafter` (`mtp.cpp`, behind
  `STRATA_MIXED_SPLIT`) that forwards chain_launch / chain_poll / chain_outputs_ready / set_source_R, draft and
  draft_first (serial loop), prefill (prompt), clear / kv operations.
- **Per window:** at the verdict, HIP1 writes A's final rows (≤ 8 x 40 KiB) into a new shared slot (pinned, D2H);
  PlChain carries T, tokens, p, a, the forced drafts and n_out; the front copies the rows H2D and launches vanilla's
  chain; its outputs land in shared memory with a sequence word per output, which chain_outputs_ready polls.
- **State:** the drafter's KV lives in the front; `FrontSessionBook` keeps it with the front's layers in checkpoints,
  parks and restores; the host's snapshots pass no drafter state.
- **Prompt:** each chunk's final rows go back to the front for the drafter's prompt pass (same size as today's
  front -> host chunk rows; ~1% of chunk time in copies, plus the pass itself on the pacing card).
- **Failure:** link errors end the request as today's front-link errors do; a restore failure drops the conversation.
- **Build order:** (1) the remote drafter in the serial loop (exact tokens against today); (2) the pipelined chain
  (exact tokens, forced misses); (3) prompt pass and snapshots/parks (restore tests); (4) priorities on the front
  card (chain against the fresh window). Measure c1 and the fresh window's front time after (2).
- **Vanilla places beyond the seam:** `mtp.cpp`/`mtp.hpp` (remote mode), `generate.cpp` (drafter load and bind on
  the host, ~10 chain sites, 6 prefill sites, ~19 `kv_state` sites in snapshots), `prefill.cpp` (`draft_kv`).
