# Report 0087

**Status: completed with differences; not an RC qualification.** All requested runs finished; all three live-versus-fresh probes differed. The scaled session itself did not restore a parked conversation, so a separate 32K mixed park/restore check was run. No engine code was changed and handoff 0086 was not resumed.

## Builds and tests

- Built at `931b65703`, Release/Ninja, 28 jobs, with `STRATA_MIXED_SPLIT=ON`, `STRATA_BUILD_TESTS=ON` and `STRATA_BUILD_CONVERSATION_TESTS=ON`: CUDA 13.0 sm_89 and HIP ROCm 10.1 gfx1201 with both MMQ options on. CPU and HIP vision helpers also built. All four new `build-0087-*` trees were stamped with that commit; the two production mirrors now name them. Build logs: `.ai/runs/0087-build/`.
- Sequential CTest via `ctest_sdma_off.sh`: HIP 14 passed, 0 failed, 0 skipped; CUDA 11 passed, 0 failed, 3 skipped. Each tree covered readback, all eight preparation modes, copy streams, cache, restore orchestration and both mixed-split tests. CUDA preparation `wrong-device`, `second` and `device` skipped because only one CUDA device is available. Individual results: `.ai/runs/0087-build/test-summary.txt` and test logs.
- `HSA_ENABLE_SDMA=0` was set for every GPU process, including CTest and both stages of mixed serving. The launch checks ran through the serving tools. Processes were started with SIGINT reset and awaited on cooperative shutdown.

## Serving

- HIP-only `features`: all requests succeeded, including control-vector-off and vision. The 16,156-token conversation was parked, restored at 16,088 tokens, then parked at 16,168. Mixed `features`: the same park/restore token counts; all requests succeeded. Mixed `checks`, including cancellation and the following request, passed with no reported check errors. Evidence: `.ai/runs/0087-serve/`.
- Scaled mixed session with `STRATA_IQ_MT_MIN=1`: 37 requests, 0 errors, three prunes, one compaction, three probes; maximum 32K context. Nine parks (14,078 to 30,884 tokens) but **no restore** appeared in this session's log. For actual medium-context restore, a separate mixed features run with the same environment and a 32K prompt parked 32,156 tokens, restored 32,088, then parked 32,168; all requests succeeded (`.ai/runs/0087-medium/`).
- Fresh-server replay for each scaled probe: **deep different**, **pruned different**, **compacted different**. The requests succeeded with no replay error and each produced 64 tokens both live and fresh; the comparison is output text/tool arguments and completion token count, not token IDs. These differences do not by themselves identify the cause. Probe and replay JSON: `.ai/runs/0087-scaled/probes/`; server logs: `.ai/runs/0087-replay-*/`.
- No `skip restore` lines in these server logs. The driver used the pre-existing uncommitted `mixed_split/tools/agent_session.py` changes (cycle limit and AMD sampling fallback); they were left untouched and are not in this commit.

## Safety and limits

- GPU runtime conservatively under 5 minutes, counting whole sequential test and server sessions, including startup and drain. No two GPU jobs overlapped except the two processes of one mixed run. No run exceeded 10 minutes; no request exceeded 64K context.
- Boot ID stayed `ffcbbb74-121f-46fa-a930-ca08f55e881c`. No new kernel lines matching the specified patterns and no coredumps since boot. Boot-time matching lines and before/after snapshots are in `.ai/runs/0087-safety/`; the initial `--since boot` form was rejected by this coredumpctl, so the separate before snapshot and after snapshot use the boot timestamp instead.
- `free -m` swap used: 0 MiB before and after; `swapon --show`: 0 B before, 256 KiB after. No >1 GiB growth or safety stop. The first attempt at the scaled server exited before starting a GPU process because the wrapper lacked path defaults; it was corrected and rerun.
- The three probe differences and their cause remain open. The 32K targeted restore is not evidence for long-context reliability or for the scaled session restoring; long runs remain on hold.
