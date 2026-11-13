# M2.8 — Offline CUDA/HIP ordinal evidence reconciliation

**Scope:** CPU-only source and tests. No CUDA/HIP runtime was initialized,
no GPU utilities were called, and no model was loaded for this increment.
This follows the M2.7 passive PCI inventory and cannot authorize live execution.

## Why the passive inventory does not establish ordinals

The provisional plan is **cuda:0:0-12,hip:0:12-30,hip:1:30-48**.

| BDF | Vendor | Device | Bound driver |
| --- | --- | --- | --- |
| 0000:04:00.0 | NVIDIA | 0x2705 | nvidia |
| 0000:03:00.0 | AMD | 0x7551 | amdgpu |
| 0000:07:00.0 | AMD | 0x7551 | amdgpu |
| 0000:7e:00.0 | AMD | 0x13c0 | amdgpu |

This is **not an ordinal map**. In particular, do not equate DRM card
number, sysfs/BDF order or PCI ID to CUDA/HIP ordinal zero or one. An
additional AMD PCI display function is present. Effective runtime ordinals
may be affected by backend visibility and ordering variables.

## CPU-only consumer and fail-closed state

Run only on pre-exported JSON files:

    python3 tools/heterogeneous_ordinal_reconciliation.py --inventory /path/to/m27.json --json

M2.7 must have produced the input with its existing --json mode. With no
runtime identity evidence, M2.8 reports **BLOCKED** (exit code **1**).
Malformed, missing, duplicate, incomplete and contradictory data are
rejected. The consumer does not query hardware or call runtime APIs,
including in fallback paths. Each JSON input is capped at 1 MiB.

**Only after separate explicit GPU runtime authorization**, a narrowly
scoped backend-isolated probe may enumerate the runtime-visible ordinals
and obtain PCI bus IDs using CUDA's cudaGetDeviceCount and
cudaDeviceGetPCIBusId, or HIP's hipGetDeviceCount and
hipDeviceGetPCIBusId. Even enumeration can initialize a GPU runtime.
**These calls have not been authorized.** Do not run nvidia-smi,
rocminfo, native workers or other live GPU tools as substitutes for
authorization, and do not initialize both runtimes in one process.

After authorization, obtain two separately executed and reviewed
**complete** JSON reports. An illustrative CUDA example follows;
**the example is not an observed runtime result**.

    {
      "schema": "strata-heterogeneous-runtime-identity-v1",
      "backend": "cuda",
      "method": "runtime_device_get_pci_bus_id",
      "complete": true,
      "devices": [{"ordinal": 0, "pci_bdf": "0000:04:00.0"}]
    }

The HIP report uses backend=hip and lists **every visible** HIP ordinal
in its separately initialized process. Do not manufacture identities from
the passive inventory. Do not claim complete=true for a partial or failed
probe. This milestone does not add or run a runtime probe executable.

To reconcile supplied snapshots (CPU-only):

    python3 tools/heterogeneous_ordinal_reconciliation.py --inventory /path/to/m27.json --probe /path/to/cuda-identity.json --probe /path/to/hip-identity.json --json

Successful cross-checking yields only **IDENTITY_CORRELATED_ONLY**.
It states that supplied runtime-identity claims are consistent with
supplied PCI observations; it is not an authenticated evidence chain,
GPU health check, or inference authorization. Operators must also check
host, boot, collection time, process visibility masks, and driver
environment. A stale snapshot can be misleading.

## Remaining gates

1. Assess historical NVIDIA faults from appropriately bounded and
   redacted evidence. A short journal query with no device fault does
   not demonstrate device health.
2. Build a source-backed per-stage allocation ledger: exact
   core::session_bytes at the compiled geometry, NativeDense::weight_bytes_for
   on the authorized GGUF shards, first-stage PLE device scratch,
   cache/residency, verifier buffers, pinned transfers, and runtime overhead.
   The native allocator rounds many projections to allocation granules,
   so GGUF payload bytes alone may understate the requirement.
3. Verify free VRAM through separately authorized narrow mechanisms.
   PCI BAR sizes or nominal card capacity are not free VRAM.
4. Obtain separate authorization for any live GPU step. Begin with
   one backend/stage before cross-vendor handoff and parity experiments.
   Do not run GitHub Actions.

## CPU-only tests

    python3 -m unittest discover -s tools -p 'test_heterogeneous_ordinal_reconciliation.py' -v

All 21 synthetic tests passed locally without a GPU runtime. The
standalone CTest registers this as an additional Python-only test when
Python3 is present. **Oberon runtime ordinal mapping remains unresolved.**
