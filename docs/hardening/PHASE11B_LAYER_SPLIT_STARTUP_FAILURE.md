# Phase 11B — real-model layer-split startup failure gate (test-only)

Additive follow-up to the validated two-card A/B/A gate. This tests one controlled **later-stage stream-creation refusal** while the production engine is loading, rather than another healthy continuation. It does not prove all startup failure paths or async terminal safety. It does not modify production inference code, drivers or model/profile files.

## Repository handoff

Start in `/home/chris/workspace/Strata-dev`, select the already-linked Phase 11B worktree and apply only the new incremental patch from `/home/chris/tmp/strata-dev/` once after `git apply --check`. Do not reapply any earlier R1/R2/R3/peer-primary/layer-split patches. Do not commit, push, run Actions or modify either separate heterogeneous project.

Test the Python runner and C++ interposer with a host compiler before any native execution. The host fake HIP header and library live only under `tests/tools/phase11b_fakehip`; never use them for production builds:

```sh
python3 -m unittest discover -s tests/tools -p 'phase11b_layer_split_startup_failure_test.py' -v
python3 -m py_compile tools/phase11b_layer_split_startup_failure.py
```

Build the **test-only shared object**, not part of the Strata executable, with the existing ROCm toolchain (no changes to installed toolchain):

```sh
/opt/rocm/core-10.1/bin/hipcc -std=c++17 -O2 -fPIC -shared tests/hip/phase11b_stream_create_fault.cpp -ldl \
  -o /home/chris/tmp/strata-dev/libphase11b_stream_create_fault.so
```

Do not install this library system-wide. It is inactive unless all three `STRATA_PHASE11B_FAULT_*` variables are supplied; only the isolated child receives them. It dynamically forwards every other HIP stream creation to the real runtime. If the prior physical-mapping audit used a separate `LD_PRELOAD`, do **not** chain it implicitly: this test refuses an inherited preload. Verify UUID/BDF mapping through a separate, read-only preflight or a deliberately reviewed interposer composition before the fault run. It is explicitly **not** a production feature.

First do a dry run (never launches or creates outputs):

```sh
python3 tools/phase11b_layer_split_startup_failure.py \
  --config /home/chris/workspace/Strata-phase10c/out/dual-final-off-config.json \
  --engine /path/to/the/already-tested-phase11b-hip/strata \
  --interposer /home/chris/tmp/strata-dev/libphase11b_stream_create_fault.so \
  --output /home/chris/workspace/Strata-phase11b/layer-split-startup-failure-a \
  --uuid-first GPU-e39306ee46993aad --uuid-second GPU-2f182f01261ba30b
```

Before `--run`: establish exclusive access, available memory, valid model/executable paths, clean device state and a read-only **kernel-journal supervisor** with immediate MES/TLB/page-fault/reset stop condition. Run the gate **once in one UUID order**, and inspect its journal/physical ordinal map before considering the reverse order. Put `--run` at the end to opt in. The test imposes a 180-second process deadline; a timeout kills only the owned child group.

Acceptance: exact one-time `PHASE11B_FAULT_INJECTED stream-create device=1 skipped=0` marker, the actual engine message `layer split, CUDA1: its streams failed`, **engine exit exactly 1 (not a crash)**, **no** `READY`, complete child exit, no lingering owned process, and no matching kernel journal faults. If the marker was triggered during an unrelated earlier stream creation or the production refusal text does not match, call the case **unqualified**, preserve logs, and do not claim startup-unwind coverage. The interposer can be made to skip 0–16 matching calls with `--skip` if needed, but do not search arbitrarily or escalate after faults.

Important limitations: the test is a GPU API **submission-time** failure, not a simulated delayed GPU hang, unknown completion or forced device reset. Failing a stream creation is safer than pretending a successfully accepted GPU job completed, but it does not itself prove that all resources were released on the correct device. The journal/physical mapping controls remain outside this standalone script, so the agent must use the same monitored wrapper established for the healthy two-GPU gate (or equivalent), not just the bare `--run`.

Return the focused run log, `results.json`, journal evidence, physical UUID-to-BDF mapping, executable/interposer hashes, and a refreshed full worktree patch. Leave the current R4/R5/R6 functionality unchanged unless a **real defect** is demonstrated.
