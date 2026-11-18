# Report 0070

**Partial. Tier escalation recommended:** HIP's vanilla `conversation_copy_streams_test` failed its first-stream VRAM bound. No engine fix was attempted. The committed HIP-only config also names vision executables from the vanilla checkout, contrary to rule 7; it is correctly refused and needs a compliant way to build or supply those executables. The standing 5-minute GPU limit in PROJECT.md takes precedence over the handoff's proposed 15 minutes (STATUS lists the increase as an open decision).

## Reset and builds

Reset the specified `Strata-mixed-native` worktree to the force-pushed branch after checking for tracked changes. Existing untracked `front-0066-1.err` and `.out`, and ignored `runs68/` and `runs69/`, were preserved. The other checkout and archive branch were not changed.

Clean Release/Ninja builds with tests and hetero enabled, 28 jobs: `build-0070-host`, `build-0070-hip` (ROCm core-10.1, gfx1201, both MMQ flags, `/opt/rocm/lib` RPATH), and `build-0070-cuda` (CUDA 13.0, sm_89). All compiled. Builds were refreshed as the implementation commits were made; the measurements below used `ea8f66f5a83a1ab26382d5eac2d8cebf9ff6688a`.

- Host CTest at that commit: 138 passed, 6 skipped, 0 failed, 144 total. The skips are io_priority (Windows), three unavailable trellis types, and two missing expert packs. Report 0067 had 225 passed and the same six skips; that was the hardened line with a different test inventory.
- Both GPU trees: `hetero_tests` and `hetero_cache_drop_test` pass. Device selftest passes on HIP and CUDA. CUDA: 10 additional selected GPU tests pass (including DMA, sampler, KV, verify and conversation-copy streams). HIP: 6 selected tests pass; a further batch had 6 passes then **conversation_copy_streams_test failed**: first copy stream/event consumed 151552 KiB, against its 16 MiB assertion (`src/core/conversation_copy_streams_test.cpp:65`). The batch stopped on failure; `hip_prefill_mmq_parity` and the remaining GPU tests were not run. CUDA's corresponding test passed. No further GPU jobs were run after the failure. These are subsets, not comparable to 0067's 95 HIP passes and 17 CUDA passes/3 skips.

## Launch check

`hetero/tools/check_launch.py` checks executable existence, enclosing CMake checkout, HIP/CUDA build roles, `BUILD_COMMIT` against current HEAD, and `ldd` resolution under the build, system or toolkit paths; `serve_mixed.py` calls it before launch. An override of server `--config` or `--engine` is refused. The mixed example passed at the measured build commit; replacing its front executable with `build-0067-cuda/strata` was refused for a missing/stale stamp, without a GPU run. A temporary HIP-only config without vision passed. The committed HIP-only example was refused because `vision.exe` is `/home/chris/workspace/Strata/engine/strata-vision-hip`, outside a checkout build; `cpu_exe` is also outside. It was not bypassed.

The stamps must be regenerated only after a successful build from the current HEAD; a later commit, even a report-only commit, makes the previous stamps stale. This is an operational constraint of the requested exact-HEAD rule. The check uses `ldd`, not a runtime trace of libraries subsequently loaded with `dlopen`; it is not a proof about dynamically loaded plugins or code modified after building.

## Mixed serving and eviction

`run_server_checks.sh` with the committed mixed config: health and model list 200, stream, nonstream, sampled, cancel and subsequent request 200. Turn 2 reused 59 tokens. Fast/slow-reader runs returned 200, with about 103.5–103.8 tok/s on the long decode segment. Server, host and front ended cooperatively; no process remained. Idle MemAvailable was 19.17 GiB (0069: 18.5 GiB server sample).

With two conversation-cache slots, the eviction driver sent A–D two turns each, then B, A and D turn 3. Cached tokens per turn:

| Conversation | Turn 1 | Turn 2 | Turn 3 |
| --- | ---: | ---: | ---: |
| A | 0 | 58 | 0 |
| B | 0 | 59 | 103 |
| C | 0 | 60 | — |
| D | 0 | 59 | 102 |

A was reread without an error and D reused its parked checkpoint. Some 80-token responses were truncated or empty; this run checks cache behaviour, not answer quality. Front stderr lines, alongside host logs showing two slots and evictions:

```text
strata serve: front Drop key=3598307123546025585 tokens=192 removed=1 parked=1
strata serve: front Drop key=2177193409813864814 tokens=191 removed=1 parked=1
```

The host logged `parked=2` after both evictions. The front's post-Drop count was 1 in each case; this is not continuous instrumentation and does **not** prove the front count never exceeded the host count at every intermediate operation. The engine and front shut down cleanly.

## Token comparison and throughput

Identical report-0069 token prompts, MTP off: long2 restored 4096 tokens from the checkpoint, and A2ck5 restored 16 via park/unpark. Each matched its uncached fresh read **50/50 token IDs**. The fresh reference used mmap/no shared arena, as in the earlier comparison.

Single samples, 4924-token long prompt, 200 generated, MTP on, caches on, tok/s prefill / decode:

| | This run | Report 0069 |
| --- | --- | --- |
| Mixed | 1787 / 115.3 | 2249 / 113.0 |
| HIP-only engine, no vision | 2260 / 111.3 | 2236 / 111.4 |

Idle MemAvailable: mixed engine 19.16 GiB, HIP-only engine 20.89 GiB (0069: 18.7 and 21.0 GiB respectively). The mixed prefill sample is lower; no cause is established from one sample. The HIP-only measurement was a direct engine run, not the refused committed server config. Its first attempted run correctly refused the unselected third integrated GPU; retry with `HIP_VISIBLE_DEVICES=0,1` succeeded.

## Safety and remaining work

Before GPU use: current-boot kernel log and core list inspected; historic 17 November monitor and crash cores predated the run. Afterwards no new core, no new line containing GPU fault/Xid/gfxhub/GCVM, and no reboot. New kernel lines included `Cannot map memory with base addr 0x717232eae000 and size of 0x7e9152 pages` during mixed arena startup, four AMD userptr/SVM workqueue hog warnings and one delayed_fput warning. The server logged whole-arena HIP host registration failing with invalid argument and its 48-slice fallback succeeding, as described in 0069; the mapping line is reported rather than treated as proof of its cause. Subsequent runs added two more AMD workqueue hog lines. No GPU process was forcibly terminated.

Approximately 242 seconds GPU process time in total, within the standing 300 seconds; each run was under two minutes. Remaining: investigate the HIP test failure and the new kernel mapping line; make HIP-only vision rule-7-compliant and run its server; run the full required GPU tests within an authorized budget; instrument enough to prove the front/host count invariant; independently review and negative-test the launcher, including runtime-loaded libraries. The performance regression needs a controlled follow-up. Run logs and JSON are in `runs70-mixed/` and `runs70-eviction/`; pre-existing untracked files were not included.
