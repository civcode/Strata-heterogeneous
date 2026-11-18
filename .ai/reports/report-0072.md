# Report 0072

**Partial.** HIP-only serving and the six alternating mixed samples completed. Both `ple_parity` invocations found the specified model and pack but stopped before GPU checks because the required captured block fixtures are absent. The remaining CUDA tests ran; the static spill guard failed because no sm_89 baseline exists. No engine or front-stage failure was established. Restoring or generating the independent PLE fixtures is outside this Tier 2 scope; **Tier escalation recommended** if it requires new development.

## HIP-only serving

The committed `hiponly.example.json` launched through `serve_mixed.py` and its launch check, with the current HIP and vision builds. Two-turn chat succeeded: turn 2 reused **140 tokens**. The PNG image request succeeded and described the shapes; its answer was truncated by the 80-token limit. The server exited 0 after SIGINT.

The server long request used the report 0070 document body but the server's chat template yielded **4,963** prompt tokens, rather than the engine driver's 4,924; it measured **2,294.0 prefill / 105.1 decode tok/s**, 200 generated tokens. Report 0070's direct HIP-only engine, without vision, measured 2,260 / 111.3 tok/s; the different interface and prompt length limit direct comparison. No `Cannot map memory` line appeared during this HIP-only run.

## Alternating mixed prefill A/B

Same saved 4,924-token prompt, 200 generated tokens, MTP and shared expert arena, runs in A/B/A/B/A/B order, no tracing. A used `build-0071-{hip,cuda}`; B used newly built HIP and CUDA trees in `/home/chris/workspace/Strata-hardened-0072`, both stamped at archive HEAD `b099f46fd`. The archive worktree was not committed or modified in tracked files.

| Sample | A current prefill / decode tok/s | B hardened prefill / decode tok/s |
| --- | ---: | ---: |
| 1 | 2260.8 / 113.8 | 2232.3 / 113.7 |
| 2 | 2260.2 / 113.0 | 2254.4 / 113.8 |
| 3 | 2261.1 / 113.7 | 2251.5 / 113.8 |
| **Median** | **2260.8 / 113.7** | **2251.5 / 113.8** |

The median prefill difference is about 0.4%, within the observed sample spread; the earlier 1,787 tok/s sample does not follow hardening removal in this experiment. Three samples per line cannot establish its original cause.

## Tests

- `ple_parity --selftest` ran in both `build-0071-hip` and `build-0071-cuda` with the actual shard and Q2_0 pack. Both returned 2 during preflight: `bench/micro/ple_in.bin` and `ple_out.bin` are missing. Neither GPU numerical parity check ran. The prior report's missing-model failure had concealed this second missing input. No captured fixtures were found in the relevant worktrees or model-data directories; they must be supplied or independently generated, not fabricated.
- All **171** remaining CUDA CTest entries were attempted individually: **163 passed, 7 skipped, 1 failed**. Skips: `prefill_fused_iq_test_sm75`, `prefill_fused_iq_test_ws`, `native_hadamard_test`, `tq_cpu_parity`, `tq_cpu_parity_avx2`, `expert_parity`, `pool_test`. The last five correspond to unavailable trellis types or expert pack as in report 0071. Detailed individual CTest outputs are in `cuda-results.json`.
- `cuda_spill_guard` failed its static binary inspection: sm_89 has no baseline, and two `native_gu_swiglu_kernel` specializations each use 8 bytes of stack and are not allowed. It cannot distinguish existing from new violations without an sm_89 baseline; no change was made to the guard or engine. `check_cuda_spills_unit` passed. This is not a runtime crash or proof of a new regression.
- Remaining CUDA tests not run: **none**, apart from the numerical portion of `ple_parity` blocked above. Prior report 0071 covered the other 172 registrations.

## Safety and evidence

Before/after current-boot kernel logs and core lists were recorded. No new core, reboot or GPU fault/Xid/gfxhub/GCVM line was observed. During the final current and hardened mixed samples respectively, two `Cannot map memory` lines appeared, each **0x7e9152 pages** (about 31.6 GiB, the shared arena size noted in report 0070); additional AMD workqueue and `mmput_async_fn`/`delayed_fput` hog warnings are in the kernel delta. No GPU process was forcibly terminated.

Conservative GPU-time accounting: **833.6 s** (13.9 min) total, counting the entire HIP server lifetime, each mixed driver lifetime, and all 171 CUDA CTest wall times including CPU-only tests. This is below the 1,500 s allowance. PLE preflight returned before creating a GPU context. The longest individual CUDA test took 55.9 s; the server and each mixed run were under 38 s. Runs were sequential.

Evidence: `.ai/runs/0072-checks/` (server logs/results, per-sample JSON and stderr, build logs, CTest outputs and safety snapshots). Pre-existing untracked `front-0066-1.err` and `.out` were left untouched. Open: independent PLE fixtures and sm_89 spill-guard baseline/triage; the original slow prefill sample remains unexplained.
