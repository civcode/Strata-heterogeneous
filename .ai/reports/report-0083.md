# Report 0083

**Status: Partial.** The 0.2.16 merge, targeted builds, mirrors, exactness, server checks and performance samples ran. The complete option/semantic audit and complete build targets remain open; not all historical features or all vanilla backend paths have been revalidated. `.ai/PROJECT.md` still names the old base and build layout: the workflow reserves that file to Chat AI unless the Human User directs a change to it. The handoff asks the agent to edit it, conflicting with that standing rule. Chat AI should update it.

## Merge and source

- Archived pre-merge head `7d573bee6` at `origin/archive/mixed-native-0.2.15`; merged upstream tag `v0.2.16` (`64524fb9fc82222ddce554c0643b64f056ec8366`) as `504506479`, without force push.
- One conflict: `engine/include/strata/core/conversation_cache.hpp`, the `<functional>` include. Kept the new vanilla unconditional include, removing the redundant guarded include. Git carried all other guarded insertions into `engine/` automatically. No old vanilla code restored; no hook intentionally reshaped. The new late-fork verifier path remains vanilla; exactness checks below are not a proof of every path.
- `engine/CMakeLists.txt` now adds top-level `hetero/` with an explicit binary directory. `hetero/CMakeLists.txt` uses `engine/include`; launch checker, server driver, prompt tokenizer drivers and four mirror paths follow the move. `.ai/DIVERGENCE.md` updated. `check_off_is_vanilla.py`: **6/6** changed source/header files match 0.2.16 with the option off; CMake's 12 added lines are checked separately by inspection.
- Both production mirror configs passed `check_launch.py`. The runs accepted the listed options. This is not a complete semantic audit of every option and environment variable. Notable 0.2.16 changes: `STRATA_PIPELINE_FORCE_MISS` and other pipeline tuning/test variables require `STRATA_PIPELINE_DEBUG=1`; the verifier has `STRATA_SH_FORK_LATE`; route-tail skipping is conditionally enabled by default, with the front's startup reporting on and the host reporting off for its resident fraction. The production mirror's explicit settings were not changed except executable paths. No claim that every default is unchanged.

## Builds and checks

- New Release Ninja builds under `build-0083-{host,hip,cuda,vision-hip,vision-cpu}`, 28 jobs. Host default targets built; HIP and CUDA built the `strata`, seam test, cache-drop and mock-worker targets (not every target in those trees). HIP: ROCm 10.1, gfx1201, both MMQ flags, `CMAKE_PREFIX_PATH=/opt/rocm/core-10.1` (the first configure without this prefix failed). CUDA: 13.0, sm_89. Both vision helpers built. All are stamped `234158c38`; no engine source changed since that commit. Compiler warnings appear in captured build output.
- `mixed_split_tests` and `mixed_split_cache_drop_test`: **2/2 in each host, HIP and CUDA tree**. The initial HIP test invocation failed because the explicitly selected build targets had omitted the mock-worker executable; building that target and rerunning passed. No GPU was used by these CPU tests.
- The mixed and HIP mirrors each served requests through the launch check, rc 0. Startup lines and conversation-cache evidence: `.ai/runs/0083-serve/engine-evidence.txt`.
- Exact mixed engine token IDs, `STRATA_IQ_MT_MIN=1`: serial vs two-window pipeline, code **300/300** and long **200/200 identical**; serial vs pipeline with `STRATA_PIPELINE_DEBUG=1`, `STRATA_PIPELINE_FORCE_MISS=3`, code **300/300**, long **200/200 identical**; 16K read-ahead off vs on **128/128 identical**. The first `miss` run lacked the new debug enable, so it did not force misses; `miss-debug` is the actual forced-miss check. Debug output recorded 213 pipeline verdicts; no separate undo count was collected.
- Server `checks` suite: rc 0, no reported errors, including streaming and cancellation. `features`: two-turn 16,093-token prompt parked and restored 16,088 tokens after interference; cvec=0 and vision requests returned; clean shutdown. Traced cancellation suite: request cancelled during a 32K prompt; continuation and unrelated request both returned, clean shutdown. The traced run is not used in performance comparisons.

## Performance

Untraced, fresh server per sample, alternating mixed / HIP-only, canonical pack, three samples each. Greedy code/long/16K, respectively 300/200/128 output tokens. Table gives median prefill / decode tok/s; individual request timings and output are in `.ai/runs/0083-serve/*-measure-*-result.json`. Prefixes vary per sample, so decode content may differ.

| Prompt | 0.2.16 mixed | 0.2.16 HIP-only | 0.2.15 orientation |
| --- | --- | --- | --- |
| code | 197.1 / 124.0 | 213.6 / 115.5 | report 0082 warm-request canonical mixed 364.3 / 120.6, HIP 473.6 / 111.5 |
| long | 2041.3 / 136.3 | 1975.0 / 111.0 | report 0078 mixed 2315.8 / 120.7, HIP 2371.6 / 112.5 |
| 16K | 3503.0 / 140.0 | 3019.3 / 123.7 | report 0078 mixed 3900.7 / 146.5; see report 0078 for HIP |

Report 0082 used warm requests in blocks, unlike these first requests. Reports 0078/0082 used 0.2.15 and different inputs or conditions; these columns are orientation, not a controlled version regression. The 0.2.16 prefill results merit a controlled follow-up before attributing any difference to the re-pin.

## Safety and remaining work

- Boot ID `5242f948-6a93-452b-b79e-a43f2d1afa57` before and after. Kernel safety-filter snapshot unchanged; no new GPU fault lines; no core dumps since boot. No pause or reboot. Servers and engines shut down cooperatively.
- Accounted GPU wall time: server 497.7 s + engine exactness 100.8 s = **598.5 s**, under the 30-minute budget. Each run was below two minutes.
- Evidence: `.ai/runs/0083-{build,exact,serve,safety}/`; ignored logs are not generally committed, with selected startup and safety evidence retained. Exactness result JSON contains generated token IDs.
- Open: update `.ai/PROJECT.md` under the workflow; complete the option/default/semantics audit, all-target backend builds if required, and wider compatibility validation of new vanilla paths. No 0.2.16 performance regression has been isolated. No Actions run.
