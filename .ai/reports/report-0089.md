# Report 0089

**Status: Partial — safety stop taken.** The mixed full session stopped after its current request at the first new kernel safety-pattern match. The production session and all probe replays were not run, as required by the stop rule. No conclusion about the full-session or soak qualification is possible.

## Safety stop and crash-case signs

- At **19:52:33 +01:00**, the kernel logged `workqueue: svm_range_restore_work [amdgpu] hogged CPU for >10000us 4 times, consider switching to WQ_UNBOUND`. This is a userptr/SVM workqueue warning, not a reported GPU fault; it also matches the specified `amdgpu` safety pattern, so the watcher stopped further work. One such line occurred in the run.
- The watcher requested that the driver stop after the current request. At **19:52:43**, while it drained, the engine reported the GPU was paused **418 ms** while Linux moved GPU-used memory. One pause line, total 418 ms. The server then stopped cooperatively with exit 0; no process of the run remained. Driver exit 1 records the requested stop.
- Swap stayed on. `free -m` before: 0 MiB used; after: 69 MiB used (`swapon --show`: 256 KiB before, 69.1 MiB after). The 30-second sampler first showed 28 MiB at 19:51:18 and last showed **30 MiB** at 19:52:18; the later 69 MiB snapshot is not a sampled in-session maximum. See `.ai/runs/0089-safety/swap.csv`.
- No `non-finite logits`, `ERR` or refused-operation line was found in the captured server log. No other new kernel safety-pattern line and no coredump since boot; boot ID remained `ffcbbb74-121f-46fa-a930-ca08f55e881c`. Evidence: `.ai/runs/0089-mixed-server/watcher.log` and `.ai/runs/0089-safety/`. The full server and console logs remain local, not committed; a small relevant extract is committed.
- `HSA_ENABLE_SDMA=0` was set on every GPU process; `HSA_USERPTR_FOR_PAGED_MEM` was unset. One mixed GPU job ran for **216.2 seconds** including startup and drain, within the approved budget. No GPU job was started after the stop.

## Measurements and RC criteria

The mixed run made 58 successful requests, seven prunes, zero compactions, and one pruned probe before the stop. Maximum observed request context was 116,430 tokens; no probe was replayed. Session driver wall time was 205.5 s, with 3,662,975 prompt tokens, 3,106,450 reused and 3,546 generated. The full per-request and sample JSON is in `.ai/runs/0089-mixed/`.

| Config | Observed session | Sampled swap max | GPU pause lines / total | Userptr/SVM workqueue lines |
| --- | --- | --- | --- | --- |
| Mixed | Interrupted, 58 requests | 30 MiB | 1 / 418 ms | 1 |
| HIP-only | Not run | Not measured | Not measured | Not measured |

Partial mixed medians (tool and answer requests): 0–32K, 135.8 decode tok/s, 2,109.5 prefill tok/s (8 requests); 32–64K, 103.8 / 2,352.9 (26); 64–96K, 92.9 / 2,256.3 (15); 96–128K, 109.9 / 1,959.7 (8). `.ai/runs/0089-safety/table.md` has the full partial table. Its resource section is **not** an after-first-cycle measurement: the first cycle never completed.

1. **Stability: not measured to completion.** The stop condition was met; the completed requests had no driver-reported errors and shutdown was cooperative.
2. **Correctness: not measured.** Only the pruned probe was captured; no fresh replay, deep or compacted probe was run. No non-finite-logits line observed.
3. **Resources: failed the no-swap criterion** in the partial mixed run. Growth across completed cycles and soak remains unmeasured.
4. **Performance: not measured.** No production session or 200K-token comparison; the partial medians above are not a comparison.

The long sessions, their probe verdicts and RC2 soak remain open; a decision on the observed warning/stop rule and swap is needed before proceeding.

## Driver change and validation

Added a small `SIGUSR1` handler to `mixed_split/tools/agent_session.py`: the current request finishes, then the session writes its summary with `stopped_after_request` and exits nonzero. The watcher used this path and then `stop_cooperatively` for the server. The driver selftest passed all checks before the run. No engine, server, mirror arguments or table script were changed.
