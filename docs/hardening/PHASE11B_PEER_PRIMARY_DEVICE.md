# Phase 11B — PeerExperts nonzero primary-device correctness

## Why this delta exists

`PeerExperts::open` rejected helper ordinal 0, even when `STRATA_PRIMARY_DEVICE=1` made GPU 0 the valid helper. Also the `p2p_copies_data` diagnostic hardcoded ordinal 0 for its source device, and the failed-P2P fallback disabled access to ordinal 0 instead of the configured primary. This was a real control-flow mismatch found by source inspection, **not** a reproduced wrong-byte GPU failure.

## Implementation

- Accept any visible helper ordinal other than `primary_device()`.
- Probe the actual `(primary_device(), helper)` pair instead of `(0, helper)`.
- Disable failed P2P access on the actual primary/helper pair.
- Extend `peer_experts_gpu_test` with `primary-one` (logical primary 1, helper 0). The test keeps the P2P switch off initially so it isolates startup/compute/owner correctness; it is not a P2P-native qualification.

## Integration

- Agent starts in `/home/chris/workspace/Strata-dev`.
- Copy this incremental patch to `/home/chris/tmp/strata-dev/` and apply it **once** to the existing linked Phase 11B worktree. It is relative to the R3-device-scope-validated uncommitted state, not to checkpoint HEAD. Verify `git apply --check` before applying; do not reapply the old full review patch.
- Build HIP and CUDA production binaries and HIP `peer_experts_gpu_test`.
- Run targeted native `peer_native_primary-one` in both logical device UUID orders with bounded timeouts and existing kernel-journal stop policy.
- Run a small representative set of existing peer-owner and expert-cache native tests; no need to repeat full 89-case CPU group unless changed headers affect it.
- Preserve the new source regression in the repository and report the exact tested worktree patch SHA-256.

## Follow-up, not covered by this delta

- The P2P copy-probe and enable/disable branches remain untested natively in this patch's new scenario because the existing fixture sets `STRATA_PEER_NO_P2P=1`. A controlled P2P-enabled scenario is warranted only after the host-stage reverse-ordinal case passes.
- Full-model layer-split startup and CUDA runtime remain open. Do not claim them as passed.

## Oberon validation

- Applied this incremental patch once to the existing Phase 11B worktree after `git apply --check`.
- HIP gfx1201 production binary and both peer/cache native fixtures built. CUDA sm_89 production build passed after the interrupted build was resumed.
- Registered `peer_native_primary-one` passed in both GPU UUID orders with logical primary 1 and helper 0. `STRATA_PEER_NO_P2P=1` and `HSA_ENABLE_SDMA=0` were used throughout.
- Six representative existing peer-owner cases and five expert-cache cases passed: 13/13 native invocations overall. No kernel-journal fault matched the established stop policy.
- Logs, commands, source/binary identities and the full worktree patch are under `/home/chris/workspace/Strata-phase11b/peer-primary/`.
- P2P-enabled behavior, native CUDA and full-model startup were not tested. The new case uses host-staged inter-GPU transfers; its output still uses the fixture's existing mapped-host output path.

No Git commits, pushes, Actions, driver changes or changes to the separate heterogeneous project were made.
