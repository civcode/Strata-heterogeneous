# Report 0063

Status: **Partial**. The host and HIP trees and all three probe pairings succeeded. The CUDA engine build and the specified baseline remain blocked as described below; no vanilla engine changes were made. Worktree: `/home/chris/workspace/Strata-mixed-native`. The original heterogeneous checkout and its untracked file were preserved. Hardware on this machine reports NVIDIA **RTX 4070 Ti SUPER**, not the RTX 5080 described in `.ai/PROJECT.md`.

## Old vanilla pin

Did not push `8db7ddc4`: the local vanilla checkout names its branch `private/debug/isa-guard-4349` and has two commits after its public `origin/main` which cannot safely be judged public:

- `8db7ddc4 Fix PLE reader timing assertion and refine transfer audit`
- `05134f54 WIP: harden HIP snapshot transfers and conversation restore`

The local vanilla checkout was not modified.

## Builds and CPU tests

Separate checkout-local `.venv` created with `python3 -m venv` as in vanilla `setup.sh`; `requirements.txt` installed there. All builds used `cmake --build ... -j28`, GCC 13.3.0, release mode, and `STRATA_HETERO=ON`. HIP uses ROCm clang 24.0.0git / HIP 7.16.26385, `/opt/rocm/core-10.1`; CUDA uses nvcc 12.0.140. Explicit options (other options are branch defaults):

- Host: `-DCMAKE_BUILD_TYPE=Release -DSTRATA_HETERO=ON -DSTRATA_BUILD_TESTS=ON` — all targets built.
- HIP: `-DCMAKE_BUILD_TYPE=Release -DSTRATA_HETERO=ON -DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DCMAKE_PREFIX_PATH=/opt/rocm/core-10.1 -DSTRATA_PREFILL_MMQ=ON -DSTRATA_BUILD_TESTS=ON` — all targets built. Current branch also defaults `STRATA_MMQ=OFF`, so the configure log reports no llama.cpp MMQ prompt kernels.
- CUDA: `-DCMAKE_BUILD_TYPE=Release -DSTRATA_HETERO=ON -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89` — probe and hetero tests built, **full build failed**. The installed nvcc 12.0 reports `Error: Formal parameter space overflowed (4216 bytes required, max 4096 bytes allowed)` in vanilla `src/prefill/moe_fused_iq.cu` and `moe_fused_iq_q36.cu`. Initial `120` configuration also failed because nvcc 12.0 does not support `compute_120`; the actual NVIDIA GPU is sm_89. No vanilla source was changed. A newer CUDA toolchain is needed to retry without altering vanilla code.

`ctest -R '^hetero_tests$'` passed in all three trees. Host full CTest: 143 registered, 137 passed, 6 skipped, 0 failed (skips: io_priority_test, native_hadamard_test, tq_cpu_parity, tq_cpu_parity_avx2, expert_parity, pool_test). Both GPU probes compiled without changes to the probe or CMake wiring.

## Shared mapping probe

All three pairings used a fresh 8 MiB `/dev/shm` file; process exit codes were zero. Times are the probe's one-thread measurements, **not** throughput figures. The GPU jobs were run sequentially.

```text
CUDA writer -> HIP reader device 0:
probe writer: register 417792 bytes of the shared mapping: ok
probe writer: host 0x74786e4b1000 -> device 0x74786e4b1000
probe writer: ok, 200 windows, 549.6 us per publish (one-thread kernel, not a speed figure)
probe reader: register 417792 bytes of the shared mapping: ok
probe reader: host 0x7d3a6128c000 -> device 0x7d3a6128c000
probe reader: ok, 200 windows, 4598.6 us per window from launch to rows read (includes the writer's pause)

CUDA writer -> HIP reader device 1:
probe writer: register 417792 bytes of the shared mapping: ok
probe writer: host 0x7f854fcb1000 -> device 0x7f854fcb1000
probe writer: ok, 200 windows, 549.0 us per publish (one-thread kernel, not a speed figure)
probe reader: register 417792 bytes of the shared mapping: ok
probe reader: host 0x7e11a46ac000 -> device 0x7e11a46ac000
probe reader: ok, 200 windows, 4713.7 us per window from launch to rows read (includes the writer's pause)

HIP writer device 0 -> CUDA reader:
probe writer: register 417792 bytes of the shared mapping: ok
probe writer: host 0x7d5d94e79000 -> device 0x7d5d94e79000
probe writer: ok, 200 windows, 3054.9 us per publish (one-thread kernel, not a speed figure)
probe reader: register 417792 bytes of the shared mapping: ok
probe reader: host 0x743a2a703000 -> device 0x743a2a703000
probe reader: ok, 200 windows, 8187.3 us per window from launch to rows read (includes the writer's pause)
```

## HIP-only vanilla baseline

Not run: the described existing **vanilla** Q2_0 configuration with `--max-context 262144` was not found. `/home/chris/workspace/Strata/strata-q2_0.json`, started by `run-q2_0.sh`, currently specifies `196608`, not 262144. The only matching 262144 configs found in the relevant checkout roots are the old heterogeneous adapter configs, including `strata-q2_0-hip-only.json`, which uses `--hetero-devices` and is not a vanilla server run config. Changing the vanilla config's context or converting the adapter config would violate the instruction to change **only** `exe`; neither was done. Thus no server was started, no request metrics or loaded free-memory measurements were taken, and no 0.2.15 startup comparison is claimed. Please identify the intended 262144 vanilla configuration or authorize a context change to the current vanilla config.
