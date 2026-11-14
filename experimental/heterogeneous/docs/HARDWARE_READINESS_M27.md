# M2.7 — Passive heterogeneous hardware readiness (no GPU runtime)

**Baseline:** M2.6 model-plan gate `69d0513f` passed on Oberon against the
user's real canonical Q2_0 pack and three stage config files. M2.7 reads
only Linux PCI sysfs/procfs, the already-authorized Q2_0 `index.txt`, and
the stage configuration files. It **does not** initialize CUDA/HIP, open
`/dev/nvidia*` or `/dev/kfd`, load GPU models, call `nvidia-smi`, `rocminfo`,
`rocm-smi`, probe VRAM availability, reset a GPU, or touch firmware.

## First Oberon assessment (no runtime, no explicit PCI mapping)

```bash
cd /home/chris/workspace/Strata-heterogeneous
python3 tools/heterogeneous_hardware_readiness.py \
  --plan cuda:0:0-12,hip:0:12-30,hip:1:30-48 \
  --config "$PWD/build-q2-stage-configs/cuda-stage0.cfg" \
  --config "$PWD/build-q2-stage-configs/hip-stage1.cfg" \
  --config "$PWD/build-q2-stage-configs/hip-stage2.cfg"
```

With a consistent passive inventory, the first run prints
`READINESS: UNVERIFIED` (exit 0), **even without optional PCI bindings**.
It lists the passively observed NVIDIA/AMD functions, kernel-bound drivers,
DRM card directories, reserves and per-stage compact weight/cache planning
estimates. The runtime device ordinals remain backend-local and are used
directly by the workers. Their mapping to physical PCI functions is optional
diagnostic information, not a prerequisite for a staged test. Contradictory
supplied bindings or missing planned GPU functions remain blockers.

The repository's expected inventory is one RTX 4070 Ti SUPER (nominal
16 GiB) and two AI PRO R9700 (nominal 32 GiB each). These are user/repo
expectations, **not** automatically detected VRAM capacities. A PCI
function with NVIDIA/AMD vendor and appropriate bound driver is not
proof of working runtime access or device ordinal assignment. BDF order,
DRM card numbering, and ROCm/CUDA ordinals are not interchangeable.

For optional diagnostic correlation, repeat the same command with
**declarative** bindings only after independently verifying their meaning:

```text
--bind cuda:0=<verified-NVIDIA-BDF>
--bind hip:0=<verified-first-AMD-BDF>
--bind hip:1=<verified-second-AMD-BDF>
```

These declarations check only that the named PCI function exists, is the
right vendor, and has the expected `nvidia` / `amdgpu` driver symlink.
Even a clean result is `UNVERIFIED`: it does not claim runtime
enumeration, device health, available VRAM, topology or numerical correctness.
This tool is only an inventory and cannot decide whether an integration test
is permitted. Runtime device selection should be checked at worker startup.

## Optional historical NVIDIA/AMD fault context

The tool does **not** invoke `dmesg`, `journalctl`, or a driver utility.
If the owner already has a **pre-exported**, appropriately redacted
kernel log limited to 8 MiB, supply it with
`--kernel-log-file /absolute/path/to/exported-kernel-log.txt`.
It reports only matching NVRM Xid, GPU-fallen-off-bus, AMD ring/reset/fault,
and PCIe AER signals. No matches does *not* establish a healthy GPU.
Do not copy full kernel logs or personally sensitive paths into GitHub.
Optionally, the owner can manually review the current boot's journal
with `journalctl -k -b 0 --no-pager` (read-only; may require journal
permissions). This step is not performed automatically.

## Memory estimate scope

The selected compact pack-weight arena is computed from the exact
19-column pack index `dst_bytes`, with its declared alignment and
`blk.<layer>.` ownership, matching the `StageWeightPlan` selection
formula. This is **planning arithmetic**, not a GPU allocation or an
observed VRAM measurement. For the canonical Q2_0 layout, the resident
cache estimate is `configured_slots*1,382,400` bytes plus the
48*512*4-byte device residency table. As of M4.7 the v2 native worker
and this estimator require explicit positive slots no larger than the
stage's layer count times 512; zero and oversized values are rejected. The config's safety reserve is shown
separately (not an allocation). The index digest is checked for changes.

**Not included:** `core::session_bytes` (layer-local state/sequence
length), native GGUF dense/PLE device allocations, verifier buffers,
pinned transfers, runtime allocations, and allocator fragmentation.
Therefore the reported `known_allocations_plus_reserve_bytes` is a
**partial requirement**, **not a VRAM-fit conclusion or a safe budget**.
The script never infers device free memory from PCI BAR sizes.

## Tests and boundaries

```bash
python3 -m unittest discover -s tools \
  -p 'test_heterogeneous_hardware_readiness.py' -v
```

19 CPU-only fixture tests, using synthetic PCI sysfs/procfs, pack index,
and staged config files; no real device nodes. Registered with the
standalone CTest project as the **29th** test if Python3 is available.
The ChatGPT implementation was locally tested with a minimal substitute
for `heterogeneous_model_gate`'s parser; run the actual-repository suite
on Oberon to verify the real import and environment.

No GitHub Actions, GPUs, model binaries, CUDA/HIP worker execution,
resets, reboots, or unrelated workspaces are needed. This milestone
ends with a **passive evidence report** and any explicit blockers;
separate permission is needed before real driver/runtime access.
