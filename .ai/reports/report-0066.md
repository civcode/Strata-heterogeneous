# Report 0066

Status: **Partial — stopped at the Human User's direction.** Part A is unfinished and its committed implementation is **experimental and not usable as a mixed engine**. No GPU processes or builds were running when work stopped.

## Part B

The unpatched 0.2.15 HIP libraries were linked with the QSA reproducer from private commit `05134f54`, built separately in the ignored `build-hip/repro-0066` directory. With `HSA_ENABLE_SDMA=0` and `STRATA_TOPK_PAGEABLE_READBACK=1`, its complete selftest passed on each R9700 three times with default dispatch and three times with `STRATA_TOPK_REG_MIN_BLOCKS=0` (12 passes, no mismatches). **The historical bug did not reproduce under these isolated conditions; this does not prove it is gone**, particularly because the historical audit observed it under contention and this test used blocking pageable QSA reads rather than the snapshot's asynchronous copy path. No readback fix was ported.

MMQ A/B: same 4,971-token prompt after one short warm-up request per build, native two-R9700 split and otherwise the report 0065 configuration. Non-MMQ: **2,162.6 prefill tok/s**; `STRATA_MMQ=ON` and `STRATA_PREFILL_MMQ=ON`: **2,641.4 prefill tok/s** (about 22% higher in this single sample). The ordinary `build-hip` remains the non-MMQ build. Both servers exited cleanly. A preliminary non-MMQ run used an 8,526-token prompt before adjusting the prompt length.

## Part A

Commit `cdb6c881` contains unfinished seam integration: changes to `CMakeLists.txt`, `include/strata/core/verify.hpp`, `src/core/verify.cpp`, `include/strata/prefill/prefill.hpp`, `src/prefill/prefill.cpp`, `src/program/generate.cpp`, and `hetero/` link/client/tests. It adds `--front-exe`, `--front-layers`, repeatable `--front-arg`, and worker `--front-stage`; experimental mixed mode disables prompt/conversation caches and rejects several unsupported options. PLE history is intended to remain in the front session; only reset and abandonment are implemented, not snapshot/restore. These choices and the code need review, including identity robustness, lifecycle and error paths, and all requested safety gates; do not assume correctness from successful compilation.

Host, HIP and CUDA builds and `hetero_tests` passed. The first CUDA-only front-client attempt failed before serving the link, during vanilla session graph setup: `fused_gr_read: invalid arguments`. The front stage had loaded its trimmed layers and expert cache. No bring-up step has passed, no mixed request ran, and the HIP-only baseline was not rerun after the engine modifications. The failure has not been diagnosed or fixed. Raw local logs are untracked `front-0066-1.out` and `front-0066-1.err` in the mixed-native worktree; they are not committed.

Approximate GPU-process wall time used: **96 seconds** total, including the reproducer runs, three HIP server runs and the failed CUDA front attempt; no mixed two-process GPU run was started. Remaining bring-up, comparisons, memory measurements and integration validation were not attempted after the stop instruction.
