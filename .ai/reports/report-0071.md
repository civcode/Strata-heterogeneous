# Report 0071

**Partial.** The GPU test budget was exhausted before the CUDA suite, HIP-only serving and prefill A/B could be completed. No conclusion about the prefill drop is possible. The known vanilla copy-stream failure recurred; the other HIP failure was a missing test input, not an established engine defect. No new engine defect was established, so Tier 1 escalation is not recommended on this evidence.

## Builds and launch check

New Release/Ninja host, HIP and CUDA trees built successfully with tests and `STRATA_MIXED_SPLIT=ON`, using the prescribed toolchains and 28 jobs. CPU and HIP vision helpers also built from this checkout. They land under `build-vision-{cpu,hip}/bin/`; the HIP-only example now names those paths and `build-0071-hip`. After the config commit, all five builds were rebuilt incrementally and stamped at that commit. The committed HIP-only config passes `check_launch.py`, including both helpers. This is a static launch check, **not** a serving or image test.

## Tests

- HIP: 272 passed, 15 skipped, 2 failed out of 289. `hip_prefill_mmq_parity` passed (11.75 s). The isolated `conversation_copy_streams_test` again reported 151552 KiB against 16 MiB, the previously identified vanilla failure. `ple_parity` failed because its relative `../../Q2_0/...00002-of-00002.gguf` input could not be opened; it was not rerun with a model path.
- HIP skips: `io_priority_test` (Windows); `expert_cache_segmented_test` (CUDA-only facility); `pinned_alias_test` (Windows HIP only); `hip_pf_cover_test` and `hip_gr_upmix_parity` (gfx11-specific); `hip_prefill_wmma_gemm_parity` (kernel declines device); `hip_prefill_hcd_exact_parity` and `hip_prefill_hipblaslt_gemm` (require tuning); `hip_prefill_rocblas_gemm` (requires tuning or supported route); `native_hadamard_test`, `tq_cpu_parity`, `tq_cpu_parity_avx2` (trellis types unavailable); `expert_parity`, `pool_test` (expert pack absent). `hip_prefill_fused_moe` also skipped; its test reports no device or fused route unavailable, but the exact runtime reason was not captured. Skip reasons other than previously documented ones are inferred from the test sources; CTest output did not preserve individual skip messages.
- CUDA: 170 passed, 2 skipped among 172 executed: `io_priority_test` (Windows) and `pinned_alias_test` (Windows HIP only). No CUDA failures observed. `ple_parity` and the remaining 171 registered tests were not run. The executed subset includes device selftest, copy-stream test and five prefill tests; it is not the whole suite.

Evidence is in `.ai/runs/0071-tests/`. Test numbers in filtered CTest output refer to filtered lists, not necessarily the original list.

## Safety and remaining work

Before/after boot kernel and core lists were captured. No new core, reboot, or GPU fault/Xid/gfxhub/GCVM line was observed. Two new kernel `Cannot map memory` lines appeared during CUDA tests, each for `0x10` pages; their cause was not established. They are distinct from the 31.6 GiB arena line. No GPU process was forcibly terminated.

GPU-process time was estimated conservatively by summing the wall times of test executables linked to CUDA/HIP runtime libraries: about 548 s HIP and 355 s CUDA, **903 s total**, including executables that might not actually create a GPU context. This conservative accounting exceeds the 900 s budget by about 3 s; the overrun was discovered only after the last batch, and no more GPU jobs were started. Each recorded individual test was under two minutes. A future run should account for budget between individual tests, not batches.

Not reached: the remainder of CUDA tests, HIP-only server reuse and image request, server throughput, and all six alternating hardened/current prefill samples. The hardened worktree was not created or modified. A new authorized GPU budget is needed for those checks. Pre-existing untracked `front-0066-1.err` and `.out` were left untouched.
