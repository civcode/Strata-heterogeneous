# Phase 11B R1 — PeerExperts borrowed-loan completion

## Where to integrate

The agent starts in `/home/chris/workspace/Strata-dev`. The patch will be copied to
`/home/chris/tmp/strata-dev/STRATA_PHASE11B_R1_INCREMENTAL.patch`.
Apply it to the existing Phase 11B implementation worktree only after identifying its
branch and the R2-validated worktree patch hash:
`8418546962721a09a0966d6135cbf8ceb972d6e7bbd25f0e3f58171432c8bcc6`.
Do not reapply the full R2 patch. The checkpoint stays `0ef75af1730c9374bbf1cb66a361f410d699b805`.

Do not inspect or modify the separate `Strata-heterogeneous` project or its tmp folder.

## Problem and correction

`PeerExperts::launch()` may enqueue DMA using a borrowed caller `x` or a borrowed
mapped `out`, then report an issue error. Prior code returned without draining.
`PeerExperts::close()` ignored stream sync failures and freed owners anyway.
`finish()` ignored the possibility that a failed stream query leaves queued work
outstanding, and `apply_pending(true)` published refill slots even after a failed
blocking event wait.

New paths use a checked device scope and the existing process-terminal policy.
Issue errors attempt a peer-stream drain; an unknown completion exits before
caller-owned storage can unwind. Teardown fences the peer device before any
buffer release, covering both helper streams and legacy-stream work. Refill
errors drain and do not publish a partial refill. An unexpected refill event
query failure is terminal; `cudaErrorNotReady` remains a normal incomplete state.
No unconditional synchronization was added to the successful `launch` / `finish`
/ `adapt` hot paths.

## ChatGPT-side checks completed

The extracted fake-runtime contract passed all nine named scenarios with GCC and
Clang, both normally and under ASan/UBSan (36 scenario executions total). The
patch passed `git apply --check` and `git diff --check` on a reconstructed
source-context fixture. This is **not** an actual production HIP/CUDA build;
that requires the complete Oberon checkout.

## Oberon results — 2026-11-07

- Started in Strata-dev and identified the existing implementation worktree. Its
  exported full patch exactly matched the R2-validated SHA-256 above.
- The supplied CMake hunk used line 1, making Git treat it as file-start context.
  Corrected that line hint to 434 in an isolated patch copy; the source delta was
  unchanged. The corrected incremental patch checked and applied once. No
  production correction was needed.
- All nine CPU completion cases plus the existing 63 regressions passed normally
  and under ASan/UBSan with leak detection: 72/72 in each configuration.
- The actual peer_experts.cpp and production executable built under HIP gfx1201
  and CUDA 12.0 sm_89.
- Added tests/core/peer_experts_gpu_test.cpp, linked to the actual production class
  through public APIs. Twenty cases passed in each physical GPU direction: 40/40.
  PeerExperts requires helper ordinal >= 1, so the harness reverses physical UUID
  visibility while keeping logical primary 0 and helper 1.
- Native cases cover direct/staged launch and finish, queued-work close, partial
  init, accepted-copy issue errors (input, plan, results), failed submission/finish
  drains, teardown sync/selection failure, refill issue/record failure, a source
  exception after an earlier refill, not-ready/query/wait event policy, and normal
  refill publication. The fixture tracks peer allocations and borrowed pinned
  x/out/blob owners. Fatal cases require child exit 1 before any owner release;
  recoverable issue failures require a successful drain before returning.
- The R2 optimization and remote-owner native regressions also passed 22/22.
  All HIP runs were serial on two R9700 cards with SDMA off. No kernel fault matched
  the MES/TLB/page-fault/reset stop condition.

Commands, logs, source identities and the full and incremental patches are in
`/home/chris/workspace/Strata-phase11b/r1-delta/`. See REPORT.md there. Native CTest
entries are peer_native_*; run them serially with two visible cards. The fixture
uses synthetic zero-weight native Q2_0 blobs, not a downloaded model. No commit,
push, driver change or separate-project inspection was performed.

## Scope and limits

The CPU test exercises the extracted policy; the new HIP fixture exercises the
production class. Native CUDA remains unavailable: no CUDA-capable device is
reported, although the RTX card is PCI-visible. Sync-error wrappers fence valid
work before reporting failure; they do not physically hang a GPU. SDMA-on R1,
real-model multi-GPU, P2P, launch-exception injection, free/get/restore-device error
injection, SYCL/Metal and performance qualification were not run. This correction
does not address expert-cache startup (R3) or long-context stability. The production
`On` helper in unrelated optional P2P-probing code is intentionally not modified.
