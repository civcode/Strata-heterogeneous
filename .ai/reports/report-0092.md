# Report 0092

**Complete, with warnings.** Handoff 0091 was not run. Builds are at `4715bea3f`, which contains `8819f09b`. The sample is a short start, not release-candidate qualification.

## Worktrees and local files

Pulled `origin/feature/mixed-native` with fast-forward only. The kept worktree had no tracked edits before this turn; its pre-existing untracked run evidence remains. Per the resumed handoff, moved the untracked `tools/heterogeneous_pool_probe.cpp` to `/home/chris/workspace/strata-local-archive/Strata-heterogeneous/tools/heterogeneous_pool_probe.cpp`.

| Removed worktree | HEAD at removal | Archived |
| --- | --- | --- |
| Strata-0084-archive | `7d573bee6` | No untracked non-build files. |
| Strata-hardened-0072 | `b099f46fd` | No untracked non-build files; ignored build products omitted. |
| Strata-mixed-native | `4715bea3f` (shared branch) | 520 non-build files, 83 MiB, including local run logs, prompts and front-stage output, under `/home/chris/workspace/strata-local-archive/Strata-mixed-native/`. |

The mixed-native worktree showed tracked changes caused by the shared branch advancing; it was removed under the explicit resumed-handoff exception, without reset or checkout. The other two had no modified tracked files. All copies were SHA-256 checked before removal. Full per-worktree status and copy manifests are in `.ai/runs/0092-resume/`. Build products were omitted. No branch was deleted. After prune, `git worktree list` shows only `/home/chris/workspace/Strata-heterogeneous`. The checkout-local venv existed.

## Builds and tests

Four fresh Release/Ninja trees, 28 jobs, stamped `4715bea3f`: CUDA 13.0 sm_89, HIP ROCm 10.1 gfx1201 with both MMQ options, CPU vision and HIP vision. Both engine trees have mixed split and tests enabled. The initial HIP configure failed to find `hipConfig.cmake`; setting `CMAKE_PREFIX_PATH=/opt/rocm/core-10.1` fixed it; all four builds passed. The example mixed and HIP-only configs now point to these trees.

`mixed_split_tests` passed through `ctest_sdma_off.sh` in CUDA and HIP (1/1 each, 0.9 s each); the test executable calls `test_session_root_kept`. No other test suite was run. Logs are in `.ai/runs/0092-build/`.

## Input and generated files

Input: vanilla `/home/chris/workspace/Strata/strata-iq3_xxs.json` (read only). Its args are the generated args below, minus the last five entries (`--mmap-experts`, `--front-exe` and its value, `--front-layers` and its value). It uses MTP spec 4, context 131072 and KV resident 32768. No Q2_0-derived copy was needed. Both GGUF shards and their .done marks, and the native pack, exist.

The generator's launch check passed for all four builds and noted that it added `--mmap-experts` to avoid duplicate expert loads. It wrote:

- `/home/chris/workspace/Strata-heterogeneous/strata-iq3_xxs-mixed.json`
- `/home/chris/workspace/Strata-heterogeneous/run-iq3_xxs-mixed.sh`

These files are local, git-ignored vanilla-form run files. The config in full:

```json
{
 "exe": "/home/chris/workspace/Strata-heterogeneous/build-0092-hip/strata",
 "args": [
  "--pack",
  "/home/chris/workspace/Strata-data/packs/iq3_xxs",
  "--native",
  "/home/chris/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
  "--ple-gguf",
  "/home/chris/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf",
  "--expert-profile",
  "/home/chris/workspace/Strata/data/expert-profile.bin",
  "--expert-cache",
  "auto",
  "--prefill",
  "auto",
  "--spec",
  "4",
  "--spec-min-p",
  "0.5",
  "--mtp",
  "/home/chris/workspace/Strata-data/mtp/rt",
  "--max-context",
  "131072",
  "--kv",
  "int8",
  "--kv-resident",
  "32768",
  "--vision",
  "--vram-reserve-mib",
  "700",
  "--mmap-experts",
  "--front-exe",
  "/home/chris/workspace/Strata-heterogeneous/build-0092-cuda/strata",
  "--front-layers",
  "15"
 ],
 "cwd": "/home/chris/workspace/Strata-heterogeneous",
 "tokenizer": "/home/chris/workspace/Strata-data/packs/iq3_xxs/tokenizer",
 "model_name": "qwen3.8-flash-next-iq3_xxs",
 "log": "/home/chris/workspace/Strata-heterogeneous/strata-iq3_xxs-mixed.log",
 "lib_dirs": [
  "/opt/rocm/core-10.1/lib"
 ],
 "port": 8080,
 "backend": "hip",
 "gpu": [
  0,
  1
 ],
 "gpus_asked": true,
 "layer_split": "auto",
 "vision": {
  "exe": "/home/chris/workspace/Strata-heterogeneous/build-0092-vision-hip/bin/strata-vision-hip",
  "mmproj": "/home/chris/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/mmproj-Qwen3.8-Flash-Next-BF16.gguf",
  "model": "/home/chris/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
  "gpu": true,
  "max_tokens": 1024,
  "cpu_exe": "/home/chris/workspace/Strata-heterogeneous/build-0092-vision-cpu/bin/strata-vision"
 },
 "env": {
  "HSA_ENABLE_SDMA": "0"
 },
 "parallel": 1
}
```

Run script in full:

```sh
#!/bin/sh
cd "/home/chris/workspace/Strata-heterogeneous"
exec "/home/chris/workspace/Strata-heterogeneous/.venv/bin/python" "/home/chris/workspace/Strata-heterogeneous/serve/server.py" "--engine" "strata" "--config" "/home/chris/workspace/Strata-heterogeneous/strata-iq3_xxs-mixed.json" "--port" "8080" "--open"

```

## Short start

Started the generated script from a Python parent using `reset_ignored_signals()`; `HSA_ENABLE_SDMA=0` was set in the parent and config. Ready in 45.0 s; stopped with `stop_cooperatively`, exit 0, front link and front process ended cleanly. No run process remained. Elapsed GPU job: 52.0 s (plus 1.8 s for both mixed split tests, if counted); below the 10-minute budget. No GPU jobs followed.

The server first tried a locally discovered DFlash2 drafter, which mixed split refused, and automatically restarted with the configured MTP. The successful start put front layers 0–14 on CUDA, HIP layers 15–30 on its GPU 0 and 31–47 on GPU 1. Auto expert caches: primary 7680 slots / 10.71 GiB; second GPU 8704 slots / 16.17 GiB. Both stages reached ready.

| Request | Prompt / output tokens | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: | ---: |
| Short | 58 / 28 | 127.8 | 114.2 |
| Repeated-text long prompt | 9063 / 64 | 2692.3 | 100.9 |

Server timings, one sample each. Evidence including the full engine log, request timings and launch console: `.ai/runs/0092-resume/`.

## Safety

Boot ID unchanged: `ffcbbb74-121f-46fa-a930-ca08f55e881c`. Immediately before the run: 58,012 MiB available RAM, 359 MiB swap used. After: 58,198 MiB available, 367 MiB swap used (of 16,383 MiB). No core dumps since boot, no new stop-pattern line, and no non-finite logits observed.

Record-pattern warnings during the run: 21:30:31 `svm_range_restore_work [amdgpu] hogged CPU ... 19 times`; 21:30:39 `amdgpu_amdkfd_restore_userptr_worker [amdgpu] hogged CPU ... 19 times`. The server also reported GPU pauses of 251 ms between requests and 785 ms during the long request, within the 21:29:50–21:30:40 run; the engine log does not timestamp those individual lines. Historical boot warnings are preserved in the before/after kernel logs. These warnings matter for the planned real session; this short success does not prove the long-session crash is resolved.
