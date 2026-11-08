# Phase 11B R2 — remote optimization teardown completion

This file accompanies the incremental R2 patch prepared against the uncommitted
Phase 11B R4/R5/R6 + optional-MTP worktree (base `0ef75af1730c9374bbf1cb66a361f410d699b805`).
It is intentionally kept **in the repository**, as requested.

## Problem and correction

`RemoteExpertOpt::~RemoteExpertOpt` previously ignored `cudaStreamSynchronize` errors
on helper streams and did not fence primary-device `copy_rows`/`combine` mapped-host
readers before freeing `h_sum_`. Those primary readers can use streams other than a
helper's stream.

The new destructor synchronizes the primary device and all attached helper devices
**before freeing any buffers**. Device selection and completion failures fail closed
via the existing Phase 11B `transfer_terminal` policy. It checks frees and restores
its entry device on successful teardown. This is a destructor-only quiescence, not a
normal-request hot-path barrier.

## Oberon validation — 2026-11-07

- The patch applied cleanly to the actual dirty worktree. No production correction
  was needed.
- CPU: all four `remote_opt_completion_*` cases plus the existing 59 regressions
  passed, normally and under ASan/UBSan with leak detection (63/63 each).
- HIP gfx1201 and CUDA 12.0 sm_89: the actual `remote_expert_opt.cu` translation
  unit and production executable compiled and linked.
- Added `tests/core/remote_opt_gpu_test.cpp`, using the existing native fixture
  pattern and the real optimization/helper owners. On two R9700 cards, all 12
  cases passed serially with SDMA off, including reversed primary/helper roles.
  Success performs a one-token optimized helper decode and queues `copy_rows` and
  `combine` on a primary nonblocking stream before the actual destructor runs.
  Wrappers require both devices to complete before either optimization buffer is
  freed. The fixture checks output bytes, helper detachment and entry-device
  restoration. Partial init covers mapped-alias failure and helper allocation
  refusal. Primary/helper sync and helper selection errors produce child exit 1
  before either owner release. Synthetic sync errors follow real successful
  completion; these tests do not fault the GPU.
- The existing two-card `remote_experts_gpu_test` group also passed all 10 cases.
  No kernel faults matched the harness's MES/TLB/page-fault/reset stop condition.

Commands and raw evidence are in
`/home/chris/workspace/Strata-phase11b/r2-delta/REPORT.md` and its `evidence/`
folder. The native CTest cases are `remote_opt_native_*`; they need two visible
cards and run serially. No model download or driver change was made.

## Limits and dependencies

The patch targets **R2 only**. It does not fix `RemoteExperts::close`, `PeerExperts::close`
(R1), or helper-cache startup (R3). CPU fake tests exercise the extracted device
completion contract, not the full CUDA destructor. HIP native teardown is now
covered by the fixture above, not a real-model qualification. NVIDIA runtime is
still unavailable; native CUDA teardown, SDMA-on R2 coverage, device-query/free
error injection and broader backend qualification remain untested. It requires
`include/strata/core/transfer_terminal.hpp`
from the already-integrated Phase 11B patch.

No commit/push is authorized. Review the dirty worktree and create the next normal
implementation delta only after tests.
