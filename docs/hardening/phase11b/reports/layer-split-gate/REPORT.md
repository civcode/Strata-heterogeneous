# Phase 11B bounded layer-split gate — Oberon

## Result

Passed the healthy real-model two-stage A/B/A restore gate in both confirmed physical GPU directions. This does not qualify startup-failure unwind or other error paths.

Started from `/home/chris/workspace/Strata-dev`, which remains clean. Selected the existing linked `/home/chris/workspace/Strata-worktrees/phase11b-prefill-restore`, HEAD `0ef75af1730c9374bbf1cb66a361f410d699b805`. Applied the supplied additive patch exactly once after `git apply --check`. No earlier patch was reapplied. No production changes were needed.

Input: `/home/chris/tmp/strata-dev/STRATA_PHASE11B_LAYER_SPLIT_GATE_INCREMENTAL.patch`.
SHA-256: `99f6cc4cdd0dca508faee4f04d754be2b4f8d61ff41b78458b382d8a9bcd6323`.

## Checks before execution

- CPU gate tests: 11/11 passed. Python byte-compilation passed.
- Default dry run passed before any real-model execution; created no native output directory. This dry run only describes the plan, so availability and memory checks were performed separately.
- Reused the existing validated HIP production binary from the peer-primary step; no C++ targets changed in this patch, so no rebuild was needed.
- Used existing `/home/chris/workspace/Strata-phase10c/out/dual-final-off-config.json` without editing it. Qwen3.8-Flash-Next Q2_0, int8 KV, context 24576, automatic expert cache/prefill, MTP spec 4, automatic layer split. Existing model, tokenizer, control-vector and profile paths were checked.
- Preflight: about 56 GiB host memory available; about 32567 MiB free VRAM on each R9700. No competing model engine was running. Generated A/B prompt lengths: 624/534 tokens.
- Gate overrides remained child-only: 2048 MiB conversation cache, four parked/checkpoint slots and 128-token checkpoint interval. Existing config/profile/executable hashes match before and after.

## Native results

Two fresh engines ran serially under `monitored_gate.py`, with a 300-second whole-gate deadline and an explicit 330-second outer tool timeout. `HSA_ENABLE_SDMA=0` throughout. The established journal regex checked MES failure/timeouts, INVALIDATE_TLBS, page faults, GPU resets and KFD/amdgpu errors. Journal reads succeeded; no matching faults. The supervisor polled during execution and the child checked before/after each engine, preventing a second engine after a failed boundary check. The private process group was cleaned up afterward; no model engine remained.

Both processes logged `layers 0-23 (CUDA0), 24-47 (CUDA1)` and three normal length-completed requests with nonempty generated IDs. Both A-return requests freshly logged a parked-checkpoint restore of **617 tokens**, with `reused=617`, then read the remaining seven prompt tokens.

| Visibility order | Engine PID | Logical 0 | Logical 1 | Result |
|---|---:|---|---|---|
| e39306ee46993aad, 2f182f01261ba30b | 78337 | 0000:03:00.0 / e39306ee46993aad | 0000:07:00.0 / 2f182f01261ba30b | pass |
| 2f182f01261ba30b, e39306ee46993aad | 79302 | 0000:07:00.0 / 2f182f01261ba30b | 0000:03:00.0 / e39306ee46993aad | pass |

This mapping is observed, not inferred from masks. A validation-only `LD_PRELOAD` interposer observes the real successful `hipGetDeviceCount` call and queries `hipDeviceGetPCIBusId`, `hipDeviceGetUuid` and the device name in each engine process. It does not allocate, select devices, enable/disable peer access, synchronize or inject errors. Its mapping lines are in the raw engine logs; the launcher validates them against UUID/BDF inventory and adds them to `native/results.json`. Preflight runtime probes independently showed the same reversal. The HIP runtime here returns UUID bytes containing the ASCII ROCr unique-ID string, preserved in raw hex too.

API source: [AMD HIP initialization/version reference](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/initialization_and_version.html), which defines the device-ordinal PCI bus-ID and UUID queries. Both configured web search tools were consulted. No advisor tool was available.

Generated IDs and finish/reuse values are preserved in `native/results.json`; no general parity claim is made. Prompt ID files under evidence are regenerated from the same unchanged tokenizer/template/config after execution and labeled accordingly.

## Artifacts and source identity

All evidence: `/home/chris/workspace/Strata-phase11b/layer-split-gate/`.

- `native/results.json`, both raw engine logs, `evidence/native-runner.log`.
- Journal before/live/boundary/after logs, supervisor result, dry-run/CPU logs, physical inventory and memory preflight.
- `mapping_audit.cpp`, `.so`, `preflight.py`, `monitored_gate.py`: exact validation harness, not linked into production or added to the repository.
- Input/config/profile/executable hashes in `evidence/input-hashes.json` and matching `post-input-hashes.json`.
- `pre-integration.patch` exactly matches the prior peer-primary full-patch SHA-256 `06d74352e1a8909f3ea7588de0b41c355734f65e3bcec357b2494be7f5c379e1`.
- `review.patch`: full tested worktree, SHA-256 `c396e5511f8fed299a153dd215ba8bc07ffa9e62f6b4c0fc1ec3ebafa3c3350e`.
- `LAYER_SPLIT_VALIDATED_INCREMENTAL.patch`: only the three supplied additions relative to that prior state, SHA-256 `7ab866ba45801e6eda99fb52592776b9b47c168cbfc7f608aaf1ae94e19d743e`. Different serialization from the input, no content adjustment.
- Full patch applicability checked against HEAD through a separate temporary index. Tracked/untracked whitespace checks passed; authoritative index empty. No post-test repository source edits.

## Limits

P2P-native qualification, native CUDA, longer prompts/contexts, borrowed/converted single-stage restores and real-model startup/failure injection remain open. This run exercised healthy production layer-split startup and parked-image restore, not the outstanding layer-split early-return/unwind failure. No 89-case CPU rerun was needed: only the gate, its tests and its note were added. The runtime mapping/journal enforcement is in this external launcher, not built into the supplied standalone gate; future native runs must retain it or equivalent controls.

No commits, pushes, Actions, driver/device resets, downloads, user profile writes or separate heterogeneous-project access/changes. Do not reapply the patch to this already-integrated worktree.
