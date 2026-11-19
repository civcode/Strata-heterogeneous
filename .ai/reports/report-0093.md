# Report 0093 — Partial: collapse fixed; default-output exactness remains open

The mixed pipeline collapse is fixed on Oberon by separating the front worker's host core from the host pipeline's core. With the IQ3_XXS coding prompt the traced decode median rose from **26.5 to 108.7 tok/s**; the long prompt rose from **26.4 to 102.7 tok/s**. The three-card load is balanced again. This is not a release-candidate qualification: the default serial and pipelined IQ3 runs differed at token 29, and the existing trace does not report the individual HIP stage GPU durations for pipelined windows. Controlled equal-cache serial/pipeline exactness did pass.

**Human User's context (not measured by this turn):** vanilla, 2× R9700, IQ3_XXS: `--pipeline-windows 0` HIP0 45%, HIP1 98%; `--pipeline-windows 2` HIP0 70%, HIP1 85%, 120 tok/s. Mixed serial: CUDA 23%, HIP0 51%, HIP1 96%. These were the reference expectations for GPU busy.

## Cause and change

Both processes initially chose `--host-core first`: their host threads were pinned to logical CPU 0, and the front window runner inherited that pin. When pipelining reduced the front cache from 7680 to 7510 of its 7680 experts, its graph became a doorbell graph that needs prompt CPU service. The host pipeline and its front runner then competed on CPU 0. The front window took about 43 ms rather than 4 ms and HIP0 waited roughly 44 ms for the front ready word. Q2 retained all 7680 front experts, used the zero-doorbell graph, and did not show the collapse. GPU busy was high on CUDA and HIP0 but HIP1 was mostly idle: high utilization here includes waiting/spinning, not necessarily useful work.

This is supported by two controlled changes on the original build: setting only `--front-arg --host-core --front-arg last` recovered IQ decode to 108.9/103.6 tok/s (short/long); alternatively lowering only the front reserve from 700 to 450 MiB restored all 7680 expert slots and produced 115.6/104.3 tok/s. The latter was an experiment, **not** the fix: it consumes part of the user's requested VRAM safety reserve. A preliminary no-yield-only build reached 74.1/76.9 tok/s, not parity, and was reverted. The original yielding runner with separate cores was already fast.

The final seam change chooses front `last` when the host chooses first or sibling, and front `first` when the host chooses last, for requested pipelined mixed mode. An explicit front `--host-core` wins; serial mode is unchanged. This uses the existing CPU affinity mechanism; it adds no pipeline scheduling or speculation logic. It cannot promise distinct physical cores for a one-core CPU set, and pools or affinity overrides can still conflict. The final built sources are commit `3fb31013b`, identical to the reapplied source at `39bb4debe`; both engine builds and vision helpers are `build-0093-*`. User run files were not edited.

## Decode and card load

Three fresh server sessions per row; same greedy coding request (83 prompt tokens) and long document request (11,155 tokens), 300 generated tokens each, `STRATA_SPLIT_TIMING=1` and `STRATA_DECODE_TIMING=1`. IQ uses the user's maxperf copy; Q2 uses the 0077 production mirror including its control vector. `pipe` drafts on HIP1; `first` drafts on HIP0. Thus these are traced within-configuration comparisons, not an IQ-vs-Q2 model benchmark.

Decode is the median and range of three request speeds, not an untraced agent-work number. Busy is the median of each run's per-card median from approximately one-second `nvidia-smi` and `amd-smi` samples between the first and last streamed token; fast runs had only 2–4 samples each, so those busy figures are coarse. Draft counts and on-path/rollback counts are medians across the three sessions; a dash means serial. The engine's on-path/rollback figures are speculative-window counts, not draft-token acceptance.


### before

| Model / mode | Prompt | Decode tok/s median (range) | Draft accepted / proposed, median | Guess hit / miss, median | CUDA / HIP0 / HIP1 busy % |
| --- | --- | ---: | ---: | ---: | ---: |
| IQ serial | short | 103.0 (97.2–104.9) | 145/209 | — | 22/48.5/86.5 |
| IQ serial | long | 100.8 (100.7–104.2) | 152/231 | — | 22/53/96 |
| IQ pipe | short | 26.5 (25.1–27.0) | 144/218 | 45/63 | 86/95/20 |
| IQ pipe | long | 26.4 (26.3–26.6) | 150/235 | 39/74 | 86/98/20 |
| IQ first | short | 26.9 (25.8–27.0) | 144/218 | 45/63 | 88/96/13 |
| IQ first | long | 26.4 (26.3–26.8) | 150/235 | 39/74 | 89/98/13 |
| Q2 serial | short | 105.5 (101.7–107.6) | 132/180 | — | 22/53/93 |
| Q2 serial | long | 106.0 (105.9–107.1) | 147/235 | — | 23/52/95 |
| Q2 pipe | short | 112.7 (106.8–118.5) | 130/174 | 36/77 | 32/78.5/74 |
| Q2 pipe | long | 110.9 (102.7–111.3) | 142/235 | 42/70 | 31/79/74 |
| Q2 first | short | 103.7 (102.0–110.6) | 130/174 | 36/77 | 29/78/41 |
| Q2 first | long | 105.7 (105.2–106.9) | 142/237 | 42/70 | 35/89/41 |

### after

| Model / mode | Prompt | Decode tok/s median (range) | Draft accepted / proposed, median | Guess hit / miss, median | CUDA / HIP0 / HIP1 busy % |
| --- | --- | ---: | ---: | ---: | ---: |
| IQ serial | short | 102.1 (101.4–104.0) | 145/209 | — | 23/54/96 |
| IQ serial | long | 92.8 (85.8–100.5) | 152/230 | — | 22/54/97 |
| IQ pipe | short | 108.7 (106.9–109.2) | 144/214 | 46/62 | 28/73/72 |
| IQ pipe | long | 102.7 (102.1–102.8) | 149/230 | 41/74 | 32/72/70 |
| IQ first | short | 103.7 (92.7–105.0) | 144/214 | 46/62 | 34.5/86.5/43 |
| IQ first | long | 97.6 (97.1–102.7) | 149/230 | 41/74 | 32/86/40.5 |
| Q2 serial | short | 110.3 (110.1–110.7) | 132/180 | — | 20/52/95 |
| Q2 serial | long | 106.2 (105.6–106.4) | 147/234 | — | 24/52/96 |
| Q2 pipe | short | 118.6 (101.8–118.8) | 130/174 | 36/77 | 30.5/57/76 |
| Q2 pipe | long | 113.1 (113.0–113.4) | 142/235 | 42/70 | 27/77/75 |
| Q2 first | short | 110.7 (101.5–111.2) | 130/174 | 36/77 | 34/93/45 |
| Q2 first | long | 107.7 (107.5–107.8) | 142/235 | 42/70 | 30/89/43 |


For a same-day comparison, the HIP-only build of this branch (vanilla plus the pinned-readback patch, no mixed front), IQ3, pipelined with HIP1 drafting, produced 97.7/108.5 tok/s short/long in one traced session, GPU medians 0/64.5/73 and 0/70/80. It is not the Human User's untraced 120 tok/s median. Our fixed mixed IQ results do not establish a win over that 120 tok/s reference.

## Window and link timings

Representative first session in each table arm; ms per measured call/window, not additive across overlapping stages. `new` is the final placement fix. Front GPU spans include any doorbell waits; HIP0 wait is the GPU's wait on the front ready word. In the baseline IQ short run, 63/63 wrong speculative windows were still running at verdict and continued for a mean 38.7 ms; with the fix 61/62 had already finished, and one continued 4.7 ms. PlUndo's CPU-side round trip was only 0.024 ms before the fix; it was not itself the 40 ms cost.

| Config (representative run) | Prompt | Front kept/undone GPU ms | HIP0 wait ms | PlRun / PlCommit / PlUndo ms |
| --- | --- | ---: | ---: | ---: |
| iq-pipe | short | 43.402/43.542 | 44.176 | 0.041/0.041/0.024 |
| iq-pipe | long | 43.248/43.696 | 43.335 | 0.041/0.040/0.025 |
| iq-first | short | 43.225/43.406 | 43.236 | 0.042/0.041/0.025 |
| iq-first | long | 43.195/43.686 | 43.317 | 0.047/0.044/0.036 |
| q2-pipe | short | 3.472/3.278 | 3.341 | 0.026/0.027/0.016 |
| q2-pipe | long | 4.100/4.180 | 4.855 | 0.027/0.031/0.016 |
| q2-first | short | 3.481/3.242 | 3.351 | 0.025/0.027/0.013 |
| q2-first | long | 4.080/4.071 | 4.020 | 0.024/0.030/0.013 |
| iq-pipe-final-new | short | 4.239/4.092 | 3.961 | 0.160/0.127/0.257 |
| iq-pipe-final-new | long | 4.486/4.454 | 4.295 | 0.105/0.171/0.213 |
| iq-first-final-new | short | 4.246/4.056 | 4.036 | 0.126/0.146/0.077 |
| iq-first-final-new | long | 4.734/4.466 | 4.513 | 0.099/0.139/0.255 |
| q2-pipe-new | short | 3.347/3.131 | 3.179 | 0.051/0.038/0.042 |
| q2-pipe-new | long | 3.789/3.771 | 3.652 | 0.052/0.049/0.020 |
| q2-first-new | short | 3.337/3.135 | 4.209 | 0.056/0.061/0.028 |
| q2-first-new | long | 3.806/3.787 | 3.728 | 0.058/0.043/0.026 |


The per-stage window-span hooks in this build did not produce individual HIP stage GPU spans for pipelined windows; the serial summary also printed stage 1 with zero samples. Those zeros are **not measurements of zero work**. For the baseline IQ serial short/long, the available HIP0 window span was 5.51/5.90 ms, ready-word wait 3.51/4.06 ms and front span 4.20/4.52 ms. For HIP-only IQ pipeline, the long request's fresh/on-path classes were 22.13/12.77 ms, with 43 hits and 66 rollbacks; baseline mixed IQ was 85.33/49.07 ms, with 38 hits and 74 rollbacks. These classes are not an isolated per-miss cost; we did not measure a standalone vanilla rollback duration. All per-run counts and traces are in `.ai/runs/0093-measure/*/server.log`.

## Tail skip and exactness

The front defaulted to `STRATA_ROUTE_TAIL_SKIP=7`, while the host's default turned it off based on its own cache coverage. These are independent stage-local decisions; the front has much less VRAM. Forcing 0 in both processes in the *unfixed* pipeline gave 22.5/26.4 tok/s, with draft acceptance 117/181 and 148/237 versus 144/218 and 152/238 under the defaults. Outputs differ as well. One pair is not enough to recommend a universal policy, and passing a host-derived setting to the front would change vanilla's output/performance policy; no such change was made.

With `STRATA_IQ_MT_MIN=1`, fixed build, greedy code prompt and 300 generated token IDs:
- Default serial versus pipeline: **first difference at token 29** (reported by `cmp_tokens.py`); serial had 7680 front slots, pipeline 7510.
- With both front caches matched at 7510 and default tail behavior: **300/300 identical**.
- With both caches matched and tail skipping forced off in both processes: **300/300 identical**.
- With tail skipping off but the default unequal cache sizes, serial versus pipeline still first differs at token 29. Matched-cache serial tail-on versus tail-off first differs at token 35. Tail skipping can change answers, but does not explain the token-29 difference on its own. The graph/cache-path difference remains a plausible, unproven cause; default-configuration exactness remains unresolved.

## Validation and limits

- CUDA 13.0 sm_89 and HIP gfx1201, both MMQ options, mixed split and tests; 28 jobs. Both vision helpers also built.
- `mixed_split_tests` passed in both trees, including CPU tests for default placement, serial mode, first/last/sibling and explicit front override. `check_off_is_vanilla.py origin/base/vanilla-0.2.15-pinned-readback`: **6/6**. An initial invocation using a nonexistent local base ref failed; the remote-tracking ref passed.
- Evidence, configs, sampled utilization, full traces, token IDs, safety logs and build logs: `.ai/runs/0093-*/`. One preliminary server was ready on an unintended port, so its 241.9 s run was stopped without requests; the retry used an explicit port. One early sampling attempt included the integrated GPU's `N/A`; it was corrected before the table runs. Neither attempt is in the tables.
- One sample per request may encounter driver pauses; the three-sample ranges show the observed spread. No forced-miss, sampled-temperature, extended-context or long-running regression was run in this handoff. CPU-placement effects on other topologies and explicit affinity masks need validation.

## Safety

`HSA_ENABLE_SDMA=0` was set for all GPU processes and query tools. Every job ran sequentially and stopped cooperatively; no run process remained. Conservative sum of server/engine wall times, including the failed-port wait: **2,389 s (39.8 min)**, plus about 4 s for the two CTest pairs, under the approved 90-minute budget. All individual jobs were under 10 minutes.

Boot remained `ffcbbb74-121f-46fa-a930-ca08f55e881c`; swap was 740 MiB before and 746 MiB after, 16,383 MiB total. No new core dump, stop-pattern kernel line, or non-finite logits was found. New record warnings: 23:12:09 `svm_range_restore_work` 131 times, 23:22:14 `amdgpu_amdkfd_restore_userptr_worker` 259 times, 23:32:19 `svm_range_restore_work` 259 times, 23:48:53 `svm_range_restore_work` 515 times, 23:53:09 `amdgpu_amdkfd_restore_userptr_worker` 515 times. Also 23:15:39 `mmput_async_fn` hogged CPU (not an AMD stop line). Across the run logs 47 GPU-pause messages ranged from 143 to 3,804 ms; `.ai/runs/0093-summary/pauses.txt` lists them by server start time. Individual engine pause lines have no timestamp, so the start time is not asserted as the pause time. These warnings still warrant care in a real long session.
