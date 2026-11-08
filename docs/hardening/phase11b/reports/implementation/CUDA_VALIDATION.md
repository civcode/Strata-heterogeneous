# CUDA validation

## Environment

- NVIDIA AD103 / RTX 4070 Ti SUPER at 0000:04:00.0, PCIe x4, observed 2.5 GT/s.
- NVIDIA driver 580.178.04 from /proc/driver/nvidia/version.
- NVML: unable to determine GPU handle, unknown error; no devices found.
- Native runtime probe: no CUDA-capable device detected.
- Toolkit nvcc 12.0.140; host compiler g++-12 12.4.0, with -allow-unsupported-compiler.
- CMake 3.28.3, Ninja 1.11.1, Ubuntu kernel 7.0.0-42-generic.
- Compute Sanitizer 2022.4.1 is installed but memcheck cannot load libsanitizer-collection.so.

The GPU is PCI-visible but was not usable for CUDA execution. No cause or OCuLink recovery claim is made. No installation/reset was attempted.

## Build

Directory: /home/chris/workspace/Strata-builds/phase11b-cuda.
Release, STRATA_ENABLE_CUDA=ON, CMAKE_CUDA_ARCHITECTURES=89, STRATA_BUILD_TESTS=ON,
STRATA_BUILD_CONVERSATION_TESTS=ON, CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12,
CMAKE_CUDA_FLAGS=-allow-unsupported-compiler.

Initial native-experts-off component targets built. The production executable in that configuration failed to link existing CPU expert symbols. Enabling the normal native expert configuration moved past that issue. The first standalone llama.cpp dependency was newer than Strata's MMQ interface; another local snapshot was older. These dependency mismatches were not patched into Strata.

Final dependency: an isolated copy of the existing matching ggml subtree at
/home/chris/workspace/Strata-builds/phase11b-ggml-source/ggml.
STRATA_NATIVE_EXPERTS=ON and STRATA_GGML_DIR names its parent. Input hashes are in evidence/ggml-dependency-hashes.txt. Nothing was written into the original checkout or installed toolkit.

### Parameter overflow

Explicit fused-IQ object build reproduced 4216 required versus 4096 permitted bytes in NT=256 native_kernel instantiations (evidence/cuda-fused-iq-build.log). The shared host/device 128-expert batch cap for CUDA <12.1 corrected it. Full production executable `strata`, Prefill and selected native test targets subsequently compiled and linked.

Final build log: evidence/cuda-final-app-build.log. Other native targets: cuda-completed-build.log and cuda-final-reviewed-build-corrected.log. This is a full application build, not a build of every unrelated test/benchmark target.

NVIDIA's documented larger-parameter support starts with CUDA 12.1 on Volta+:
https://developer.nvidia.com/blog/cuda-12-1-supports-large-kernel-parameters/

## Runtime results

- Baseline pinned_ring_gpu_test and conversation_copy_streams_test: both skipped (77).
- Corrected conversation_lifetime_gpu_test success and issue-drain, plus pinned_ring_gpu_test: 3 skipped, 0 passed, 0 failed. See cuda-reviewed-runtime-results.json.
- Compute Sanitizer command attempted:
  `compute-sanitizer --tool memcheck /home/chris/workspace/Strata-builds/phase11b-cuda/pinned_ring_gpu_test`
  exited 13 before memcheck due to its missing injection library. Not a memory-safety pass.
- NVIDIA inference, restore, exact-byte transfer execution and CUDA async-worker execution: not run because runtime initialization fails.

Compilation does not establish native CUDA correctness or qualification. The same tests are ready to run once the existing device/runtime issue is resolved outside this task.
