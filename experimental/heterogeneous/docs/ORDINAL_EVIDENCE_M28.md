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
runtime identity evidence, this optional M2.8 diagnostic reports
**BLOCKED** (exit code **1**) *for identity correlation only*. This
does not block an otherwise operator-initiated GPU integration test.
Malformed, missing, duplicate, incomplete and contradictory data are
rejected. The consumer does not query hardware or call runtime APIs,
including in fallback paths. Each JSON input is capped at 1 MiB.

During a user-initiated GPU test, an isolated backend worker can enumerate
runtime-visible ordinals
and obtain PCI bus IDs using CUDA's cudaGetDeviceCount and
cudaDeviceGetPCIBusId, or HIP's hipGetDeviceCount and
hipDeviceGetPCIBusId. Even enumeration can initialize a GPU runtime.
These calls initialize the runtime and should only happen when the user
initiates the corresponding GPU test or grants an equivalent scoped
instruction. That instruction is sufficient authorization for the test;
a second authorization ritual is unnecessary. Keep CUDA and HIP runtimes
in separate worker processes.

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
3. Verify free VRAM from the live, explicitly launched native stage worker.
   PCI BAR sizes or nominal card capacity are not free VRAM.
4. Explicitly run the manual native CUDA→HIP smoke launcher from
   LIVE_GPU_HANDOFF_M29.md. The user-initiated test is sufficient authorization;
   no additional environment approval flags are required. Do not run Actions.

## Partial source-backed session-state floor

Reviewing src/core/session.cpp and the compiled ModelGeometry defaults also
gives a conservative portion of each stage's session memory: each owned GDN
layer requires 128 * 48 * 128 floating-point recurrence elements plus
10,240 * 3 floating-point convolution history elements. At 4 bytes per
element this is exactly **3,268,608 bytes (3.1171875 MiB) per GDN layer**.

| Stage | Owned GDN layers | Per-layer GDN state alone |
| --- | ---: | ---: |
| CUDA 0, layers 0–11 | 9 | 28.05 MiB |
| HIP 0, layers 12–29 | 14 | 43.64 MiB |
| HIP 1, layers 30–47 | 13 | 40.52 MiB |

These amounts are an identified **subset** of the unpriced session arena,
not the full core::session_bytes result and not independently allocated
on top of a future complete session estimate. One shared GDN scratch area,
QSA KV/indexer states and scratch, MoE/block scratch, PLE history, native
dense projections, verifier allocations and runtime overhead are still
unaccounted for in M2.7. No free-VRAM or fit verdict follows from this floor.

## CPU-only tests

    python3 -m unittest discover -s tools -p 'test_heterogeneous_ordinal_reconciliation.py' -v

All 23 synthetic tests passed locally without a GPU runtime. The
standalone CTest registers this as an additional Python-only test when
Python3 is present. **Oberon runtime ordinal mapping remains unresolved.**
