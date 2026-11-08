# HIP / ROCm validation

## Environment and isolation

ROCm installation: /opt/rocm/core-10.1. HIP compiler/runtime 7.16.26385-0000000; AMD clang 24.0.0git (d6f6cb691863e9258034ab48c1dfce21689d5606).
Runtime library: /opt/rocm/core-10.1/lib/libamdhip64.so.7.16.26385-0000000.
Kernel: 7.0.0-42-generic. rocm-smi is absent; amd-smi list and rocminfo identified:

| Physical device | PCI | HSA UUID | Architecture |
|---|---|---|---|
| R9700 #0 | 03:00.0 | GPU-e39306ee46993aad | gfx1201 |
| R9700 #1 | 07:00.0 | GPU-2f182f01261ba30b | gfx1201 |

Both observed links: PCIe x16 at 32 GT/s. The gfx1036 iGPU was excluded. ROCR_VISIBLE_DEVICES selects exactly one UUID; native logical device 0 is that selected R9700. No simultaneous GPU test processes.

## Build

/home/chris/workspace/Strata-builds/phase11b-hip:
STRATA_ENABLE_HIP=ON, CMAKE_HIP_ARCHITECTURES=gfx1201, CMAKE_PREFIX_PATH=/opt/rocm/core-10.1,
STRATA_BUILD_TESTS=ON, STRATA_BUILD_CONVERSATION_TESTS=ON, Release.
STRATA_NATIVE_EXPERTS=ON; STRATA_GGML_DIR=/home/chris/workspace/llama.cpp.

The initial missing hip CMake-package error was fixed by the explicit installation prefix, not a source/toolchain change.
The loader initially could not find ROCm libraries for some executables. Test runs explicitly set
LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib; final linkage is recorded in hip-app-linkage-runtime-path.txt.
The complete production executable and selected native tests build. Final log: hip-final-app-build.log.

## Results

Baseline on #0 with HSA_ENABLE_SDMA=0: remote-ok, stager-ok and ring exact-byte tests passed. The first ring attempt failed before execution due to the loader path; its corrected run passed.

Corrected source, SDMA off, each device independently: **19/19 passed**.
Logs/results: evidence/hip-reviewed-device{0,1}-sdma0-*.

Selection:
- Eight production-copy lifetime/error/no-unwind modes.
- Pinned ring: sizes through 4 MiB + 17 bytes, slot reuse and 100 exact 64-KiB round trips.
- Synthetic native snapshots: 4195 checks per device.
- Native readback: successful exact data, recoverable capture refusals, actual snapshot capture/restore terminal subprocesses.
- Eight existing remote readback/Stager/query tests.

Short SDMA-on comparison: success, reported issue-error drain and ring tests passed **3/3 per card**.
These runs preceded the final Prefill-only error-unwind refinement; they are not claimed as final-source long-context validation.

No new MES timeout, INVALIDATE_TLBS failure, page fault, KFD queue failure or reset matched the recorded kernel windows. Ordinary runtime power-resume messages were present. Earlier MES incidents remain unexplained; HSA_ENABLE_SDMA=0 is only a transfer-path setting.

## Actual model smoke

Used existing Qwen3.8-Flash-Next Q2_0 shards/pack and local MTP, not a download or a small dense model.
Small runtime configuration: one R9700 (#0), context 128, prefill cap 32, 128 expert slots (168 MiB),
CPU prefill share off, HSA_ENABLE_SDMA=0, 512-MiB conversation cache. The existing expert arena is about 31.64 GiB in host RAM; INFO reported approximately 27 GiB GPU memory free after setup.

Final sequence (model_smoke.py):
1. GEN 1 with IDs 1..8; token 198, successful DONE.
2. SAVE, 118296184 bytes.
3. Continue with IDs 1..8,198; reference token 1.
4. Switch to IDs 21..28.
5. RESTORE saved session.
6. Repeat the identical continuation; token 1 again, successful DONE; QUIT/exit 0.

This establishes one matched next-token continuation after disk restoration and switching. It is not full-logit equivalence, text quality, borrowed-prefix model coverage or a throughput benchmark.

Two actual Prefill final-copy failure runs on the final source also passed: after a real successful fence the interposer reports either a one-shot error or persistent failure. The one-shot case drains positively but still reaches `ERR serve prefill` and exit 1. The persistent case reaches `ERR prefill stream completion` before any unwind. Neither emits an answer or DONE. Evidence: r4-model-terminal/ and r4-model-unknown/.

## Bring-up failures retained

Wrong executable subcommand, missing minimum speculative-window setting, missing expert profile and an initially missed interposer selection are preserved as failed setup/coverage attempts, not passes. A no-MTP SAVE produced an application SERR for invalid K/V extent; the first harness failed to recognize SERR and timed out while the engine waited for input. No kernel fault or GPU hang was observed. The harness was corrected; the supported local-MTP sequence passed.

No further broad/stress workload or AMD recovery work was attempted.
