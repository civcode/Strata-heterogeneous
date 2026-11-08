# Phase 11B — primary early raw-resource cleanup (WIP)

## Basis

The validated [control-shutdown report](phase11b/reports/control-shutdown/REPORT.md)
attributes outstanding public HIP allocations after a healthy serve exit:
20 primary hipMalloc allocations, 5 streams and 9 events. The controlled
late-stage stream-create refusal leaves 5 primary allocations and one stream.
GPU1 hipMalloc survivors were zero. Some GPU1 streams and events belong to a
deliberately process-lifetime conversation copy pool.

This change closes a bounded *early-primary* ownership gap, not the entire
primary serving or process-lifetime resource graph. The primary weight arena,
vision table, PLE embedding/scratch, expert-partials buffer and main stream are
all raw owners created before a later-stage refusal. Several also survive a
normal serve exit. The non-serving tail explicitly freed the arena and parts,
while the serving branch returns earlier. Releasing those twice would be unsafe.

## Change

A primary_raw owner is constructed at engine entry, after the control-vector
shutdown guard and before all later automatic GPU owners. The owning device
is verified before the first weight arena allocation. The owner is armed on
successful allocation and records each additional buffer or stream immediately
after its CUDA/HIP creation call, including partial-initialization failure.

The existing LayerSplitStageDelete selects the owner GPU and establishes
completion before the primary_raw destructor runs; all subsequently declared
verifiers, stages, sessions and other locals unwind first. The raw owner then
releases five buffers and the main stream with checked API return codes,
followed by the existing control-vector shutdown guard. The non-serve tail
no longer explicitly frees the two buffers now owned here. Decode and prefill
hot paths are unchanged. There is one device completion fence at engine exit.

## Local check and native acceptance

The focused fake-runtime helper compiled and passed 9 scenarios each under
GCC/Clang Release and ASan/UBSan (36 child cases). It checks owner GPU
selection, dependent destruction ordering, all six releases, empty/unarmed
cleanup, and terminal get/select/sync/free/stream/restore failure. Native
HIP/CUDA builds and actual model cleanup are **not yet validated**.

Oberon should fetch this commit into the existing clean linked worktree and
review it without reapplying older patches. Build HIP gfx1201 and CUDA sm_89
production binaries separately with **28 jobs each**. Repeat 89/89 CPU and
sanitizer groups, all Phase 11B Python harness tests, native HIP 67/67 per
physical order and CUDA lifetime 8/8. Keep native checks serial.

Using the unchanged bounded, journal-monitored real-model fixtures, run both
physical-order healthy A/B/A and later-stage stream-create refusal. Inspect
forward evidence before reverse. Repeat the pointer-identity audit on both
logical devices, including stream/event handles and graphs. Check the six
previously unowned primary handles individually against successful owner-device
releases, not only an aggregate count. A refusal should leave zero of those
primary raw resources; healthy serving should have no matching survivors.

Stop immediately for any new kernel fault, failed/wrong-device free, unknown
completion, crash, hang, duplicate free or lingering process. Preserve logs and
evidence hashes; commit/push a separate factual validation report after tests.
No GitHub Actions runs, driver changes, PR, original checkout edits or
heterogeneous-project access.

## Remaining boundaries

Even a successful audit here would leave other primary serving allocations
and explicit events/streams, process-lifetime copy pools, other allocator
families and startup branches. It does not establish all-backend leak-free
cleanup, P2P-enabled behavior, two-NVIDIA split, long-context safety, or RC1
release readiness.
