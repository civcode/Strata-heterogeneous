# Follow-up to 0095: ROCm hardware queue sweep stopped on non-finite logits

The first full replay completed at each of `GPU_MAX_HW_QUEUES=1,2,3,4`. The second `=2` replay completed; the second `=3` replay hit the project's mandatory stop condition. The sweep was not completed to three full replays per arm. No candidate was qualified, and neither the long RC run nor the `--spec` / `--spec-min-p` sweep was started.

## Results

Same IQ3_XXS maxperf config copy and `build-0093-*`; untraced, greedy replays of the first 106 requests with recorded pauses. Medians below are across completed runs only. Varying output lengths and the small sample prevent a reliable ranking.

| Queue limit | Completed full replays | Engine decode tok/s (all samples) | Median TTFT | Median draft acceptance |
|---|---:|---:|---:|---:|
| 1 | 1 | 139.91 | 3.562 s | 76.49% |
| 2 | 2 | 144.11, 140.61 | 3.518 s | 76.60% |
| 3 | 1; second stopped | 141.64 | 3.530 s | 76.99% |
| 4 | 1 | 144.47 | 3.512 s | 77.13% |

The short six-request traced compaction replay measured on-path windows at 11.9–12.4 ms after compaction for `=1`, and 11.3–11.7 ms for `=3`. Previously measured `=2` was about 11.5 ms and the default about 15.1–15.5 ms on this slice. The full untraced results do **not** establish a speedup over the default queue limit (4). Keep the existing default; do not infer that it has been newly qualified.

## Stop condition

In `sweep-full-q3-r2`, the engine restored a 106,783-token checkpoint, then detected non-finite logits at position 107,233 in the prompt. No tokens from that window were emitted. The engine invalidated its live session and caches, began a full reread, and the runner stopped the replay and shut down the server cooperatively (exit 0, no leftovers). The three log mentions are a detection and two recovery messages, not three independent faults. This resembles the restore-associated symptom in `.ai/PROJECT.md`; the queue limit is not established as its cause.

`HSA_ENABLE_SDMA=0` was set for the GPU runs. Measured server lifetime was 5,893 s (98.2 min), including the two short probes and the stopped replay. Boot ID unchanged; no core dump or new GPU fault line. A kernel warning at 00:34:03 reported `amdgpu_amdkfd_restore_userptr_worker` hogging CPU. The stopped run logged 18 engine GPU pauses. Swap used was 792 MiB before the sweep and 758 MiB at the follow-up check. Evidence: `../runs/0095-safety/sweep-stop-evidence.txt`, `sweep-table.md`, and the run summaries under `../runs/0095-sweep-*`. Raw engine logs and traces stay in the local archive.

The stop marker was left in place. Under `.ai/PROJECT.md` the non-finite-logits event ends the hardware turn; no further GPU jobs were started.
