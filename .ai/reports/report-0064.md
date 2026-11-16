# Report 0064

Status: **Partial**. CUDA 13.0 build and probe passed; the HIP baseline was not started because the prescribed unchanged configuration writes a log inside the read-only vanilla checkout.

## CUDA build

Fresh `build-cuda13`, 28 jobs, GCC 13.3.0, nvcc CUDA 13.0.88, driver 580.178.04:

```sh
cmake -S . -B build-cuda13 -DCMAKE_BUILD_TYPE=Release -DSTRATA_HETERO=ON \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-cuda13 -j28
ctest --test-dir build-cuda13 -R '^hetero_tests$' --output-on-failure
```

All targets built, and `hetero_tests` passed. `STRATA_BUILD_TESTS` defaulted ON and `STRATA_MMQ` defaulted OFF. No source fixes were needed. This resolves the CUDA 12.0 build failure recorded in report 0063.

## CUDA 13.0 writer, HIP device 0 reader

Fresh `/dev/shm` file, both exits 0:

```text
probe writer: register 417792 bytes of the shared mapping: ok
probe writer: host 0x7dae2c79a000 -> device 0x7dae2c79a000
probe writer: ok, 200 windows, 550.0 us per publish (one-thread kernel, not a speed figure)
probe reader: register 417792 bytes of the shared mapping: ok
probe reader: host 0x71f291ccb000 -> device 0x71f291ccb000
probe reader: ok, 200 windows, 4595.4 us per window from launch to rows read (includes the writer's pause)
```

## HIP baseline: blocked by conflicting file constraints

Copied `/home/chris/workspace/Strata/strata-q2_0.json` to the worktree's ignored `strata-q2_0.json`; verified its sole difference is `exe`, set to `/home/chris/workspace/Strata-mixed-native/build-hip/strata`. The copied config still names `log: /home/chris/workspace/Strata/strata-q2_0.log`. The branch's `serve/server.py` opens that path in append mode (`StrataEngine.__init__`, line 1132; vision startup also opens it), so starting the server exactly as instructed would modify the local vanilla checkout, contrary to `.ai/PROJECT.md`'s explicit read-only rule. Changing the copied config's log is outside this handoff's authorization to change only `exe`. An unprivileged mount namespace to redirect that path was unavailable (`unshare: write failed /proc/self/uid_map: Operation not permitted`). No server was started, no requests sent, and no baseline metrics are claimed. Please authorize redirecting the copied config's `log` to the worktree, or provide another way to preserve the read-only checkout.

For reproducibility, the copied config's engine arguments are:

```text
--pack /home/chris/workspace/Strata-data/packs/q2_0
--native /home/chris/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
--ple-gguf /home/chris/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf
--expert-profile /home/chris/workspace/Strata/data/expert-profile.bin
--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5
--mtp /home/chris/workspace/Strata-data/mtp/rt --max-context 196608 --kv int8
--vision --vram-reserve-mib 700
--control-vector-scaled /home/chris/workspace/Strata/data/experimental-speed-projection/Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0
--control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer
```

Config also selects `backend: hip`, `gpu: [0,1]`, `layer_split: auto`, `lib_dirs: [ /opt/rocm/core-10.1/lib ]`, cwd `/home/chris/workspace/Strata`, and has no `env` key. The vanilla launcher exports `HSA_ENABLE_SDMA=0` and overrides port to 8081. The actual server-expanded arguments, startup defaults, drafting mode, GPU memory and performance remain unobserved.
