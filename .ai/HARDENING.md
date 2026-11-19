# Hardening: one patch carried

## Carried since 2026-11-19: HIP pinned snapshot readback (`05134f54`)

**Why:** the host crash of 19 November during handoff 0086's full mixed session: memory of two processes overwritten
(a page-table page and the Python server's heap); the overwritten page-table entries decode as int8 data
(KV-cache-like); it happened at a park and restore of a 214K-token conversation, under swap and while the AMD driver
was repeatedly restoring userptr memory. Vanilla's HIP build copies parked conversations and checkpoints from the GPU
straight into pageable host memory; on AMD that memory reaches the GPU through a KFD userptr, which Linux can reclaim.
This commit stages those copies through a pinned buffer. The crash points at that path; it does not prove it. Details
of the incident stay local (Human User's evidence).

**What the commit does** (its own audit: `docs/TRANSFER_AUDIT.md`):
- HIP snapshot capture, disk-source readback and verification go through one pinned piece of 4 MiB per copying thread
  (`src/core/conversation_readback.hpp`), never straight into pageable memory. No pageable fallback; a failed or
  unknown completion never lands in the image.
- Restore preparation before any mutation (`include/strata/program/conversation_restore.hpp`): copy resources on every
  stage device are acquired first. If that fails, the image stays in the cache and the prompt is read; once mutation
  begins, a transfer failure ends the request with an error. This replaces vanilla's free-VRAM heuristic for the first
  restore (`conv_cache::restore_fits`).
- The parking race: `conversation_park_async_begin` no longer writes shared state when background parking is off (it
  is off on HIP).
- Not covered: host-to-GPU restore copies still read pageable memory (wrong data at worst, not corruption elsewhere);
  the CUDA async paths (unchanged); the mixed split's hand-off buffer.

**How it was applied:**
- On `feature/mixed-native`: cherry-pick of the earlier port `51171e10` (from `archive/hardened-0069`). One conflict in
  `generate.cpp`, at the restore skip: the commit's preparation step replaces vanilla's `restore_fits` skip, which took
  the image out of the cache and put it back. The mixed split's block on that path (drop the front stage's copy when
  the skipped image could not be returned) has no counterpart any more: the image is never taken before preparation
  succeeds. An invalid image is still discarded without telling the front stage, as before the commit.
- Reference branch `base/vanilla-0.2.15-pinned-readback`: vanilla 0.2.15 (`c775728`) plus this commit alone, with the
  same conflict resolved the same way. `check_off_is_vanilla.py` compares against it by default.

**Checked by Chat AI without a GPU (2026-11-19):**
- `check_off_is_vanilla.py`: 6 of 6 against the reference.
- CPU build with the option on: 145 of 145 CTest entries pass (6 skipped as before), including the commit's
  `conversation_restore_orchestration_test`.
- Syntax check of `generate.cpp`, `conversation_snapshot.cpp` and `conversation_state.cpp` against CUDA 12.9 headers,
  option off and on: the same 10 errors as before the commit (defines CMake sets), nothing new.
- Not checked: the HIP and CUDA builds, the commit's GPU tests (`conversation_readback_test`,
  `conversation_prepare_test`, `conversation_copy_streams_test`), a run.

## Withdrawn on 2026-11-18: the rest

The other commits of the 17 November hardening (GPU ownership through shutdown, Phase 10E, Phase 11B) stay withdrawn.
The 17 November crash was attributed to a launch config that mixed components from different builds. That line, with
its port table and reports 0067 to 0069, is on branch `archive/hardened-0069` (`b099f46f`).
