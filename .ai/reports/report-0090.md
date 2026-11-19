# Report 0090

**Status: Partial — stop rule taken. Tier escalation recommended.** The mixed session ended early on a front-stage checkpoint error. Its two captured probes were replayed. The HIP-only session then hit non-finite logits; the watcher requested stop after the current request and no further GPU work was started. Neither session reached its planned end.

## Sessions and crash-case signs

| | Mixed | HIP-only production |
| --- | --- | --- |
| Result | Engine error on request 251, first cycle | Stop on non-finite logits, request 360, second cycle |
| Driver wall time | 921.4 s | 1,409.4 s |
| Requests / errors / compactions | 251 / 1 / 0 | 360 / 0 reported by driver / 1 |
| Max request prompt | 207,739 tokens | 244,628 tokens |
| Sampled swap: first → max | 68 → 3,510 MiB (+3,442 MiB) | 626 → 626 MiB (+0 MiB sampled) |
| GPU pauses | 8, total 26,308 ms | 4, total 1,401 ms |
| userptr/SVM workqueue lines | 7 | 0 |

Full side-by-side bucket medians and the compaction are in `.ai/runs/0090-safety/table.md`. Mixed and HIP request hashes matched through request 251; the mixed sequence then ended. These incomplete, differently sized runs are not a session performance comparison. The 192K–224K bucket medians were 98.0 tok/s mixed (4 requests) and 84.7 tok/s HIP-only (9 requests), not a matched-workload benchmark.

- **Mixed error:** at request 251, `restoring a conversation checkpoint failed: mixed split: the front stage: no checkpoint at 9956 tokens is kept here`. The driver ended with one error; server exit 0. This is an engine/front-stage failure requiring investigation beyond this Tier 2 scope.
- **HIP-only stop:** at 20:41:31 +01:00 the server logged `non-finite logits (#879) in a prompt window at position 74438 (row 0)` during request 360. It dropped the live session, six checkpoints and three parked conversations, and automatically retried the current prompt from token 0. That request finished; the SIGUSR1 stop then ended the driver and the server drained with exit 0. The driver's zero-error count does not negate the non-finite-logits failure. No probe replays or other GPU jobs followed.
- Mixed workqueue warnings were recorded from 20:06:00 to 20:11:13. Its GPU pauses were recorded from 20:06:13 to 20:12:05, largest 15,140 ms; the sampler first crossed 1 GiB swap at 20:06:00 (2,375 MiB) and peaked at 20:12:32 (3,510 MiB). HIP pause lines ran from 20:23:35 to 20:36:23, largest 551 ms. Exact lines and times: the two `0090-*-server/watcher.log` files; sampled swap series: `.ai/runs/0090-safety/{mixed,hip}-swap.csv`. Server-log extracts are in that safety directory; full server logs remain local and are not committed.

## Probes and RC criteria

| Probe | Mixed replay | HIP-only replay |
| --- | --- | --- |
| pruned | near-tie, 0.014 nat | not run (stop) |
| deep (200,183 tokens) | near-tie, 0.164 nat | not run (stop) |
| compacted | not captured (mixed ended early) | captured, not replayed (stop) |

Mixed replay JSON and logprobs-derived verdicts are under `.ai/runs/0090-mixed/probes/`. HIP-only captured all three probes, but no verdict can be assigned without replays.

1. **Stability: failed** for this batch: mixed checkpoint error and HIP non-finite logits; both servers drained cooperatively.
2. **Correctness: failed** because HIP emitted non-finite logits. The two judged mixed probes passed the near-tie criterion; remaining probes are unjudged/not captured.
3. **Resources: failed** the less-than-1-GiB swap-growth criterion for mixed (+3,442 MiB sampled). After-first-cycle growth across the remaining cycles and soak is not measured. HIP sampled swap did not grow beyond its first sample; its session was incomplete.
4. **Performance: not measured** for complete matched sessions. RC2 soak remains open.

## Safety, tools and limits

`HSA_ENABLE_SDMA=0` was set for every GPU process; `HSA_USERPTR_FOR_PAGED_MEM` was unset. The full mixed, two mixed replays and partial HIP session occupied **2,427.6 s** including startup and drain, sequentially; no individual run reached two hours. The only new kernel record lines were the seven mixed workqueue warnings; none matched the kernel stop patterns. No coredumps since boot, and boot ID stayed `ffcbbb74-121f-46fa-a930-ca08f55e881c`. `free -m` swap used was 68 MiB before, 626 MiB between sessions and 578 MiB after. Before/between/after snapshots and the stop marker are in `.ai/runs/0090-safety/`.

The driver selftest passed before the runs. Fixed `mixed_split/tools/agent_table.py` to display the actual near-tie verdict instead of treating every unequal replay as `DIFFERENT`, and to report an unavailable post-cycle resource baseline when the first cycle did not complete or no later sample exists. Regenerated the table with the fix. No engine, server, driver or mirror arguments were changed.
