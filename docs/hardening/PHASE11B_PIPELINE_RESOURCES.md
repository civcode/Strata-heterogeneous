# Phase 11B — pipeline and refill device-handle ownership (WIP)

## Source evidence and separation

Validated baseline: `483dcf6d4477c1fe364c464ca8f95b42d6a8eed3`.
The [primary-serving report](phase11b/reports/primary-serving-resources/REPORT.md)
and pointer ledger show the remaining healthy-exit public HIP resources:
pipeline 3 allocations, 3 streams, 6 events; primary refill 1 stream, 1 event;
and a deliberately retained conversation-copy process pool of 3 streams and
2 events. Controlled early refusals have zero tracked public resources on
both logical GPUs. Do not label the retained pool as pipeline-owned.

The pipeline draft row buffer can be allocated on a later GPU or on the MTP
device; each per-stage stream, snapshot buffer, side stream, and event has
its own allocating GPU. A single assumed-device cleanup policy is unsafe.

## Change

A fixed 96-entry `Phase11bPipelineResources` registry is declared at engine
entry after primary-serving owners and before stage/verifier locals. Pipeline
and refill creation sites call `pipe_malloc`, `pipe_stream`, and `pipe_event`;
each records the actual current device immediately after creation, even if
a failed API returned a nonnull partially initialized handle. The registry
is inert when unused. On exit, after dependent graph objects unwind, it
fences every recorded owner device once before releasing any resource, then
releases handles in reverse creation order, checking every release and
device restoration. This avoids releasing an allocation while queued work on
another GPU could still be using it.
Any get/set/sync/release failure invokes the established terminal boundary
instead of continuing through uncertain ownership.

The existing serving hot path, pipeline scheduling and explicit handle users
are unchanged. Primary refill stream and completion event are covered in
both pipeline and non-pipeline serving. The previous ignored refill-event
create result is now an explicit startup refusal. No process-wide copy pool
or model/config/driver setting is modified.

## Local tests (NOT native qualification)

CPU fake-runtime cases: GCC and Clang Release plus ASan/UBSan, 14 scenarios
each (56 child scenarios, four Python methods). Exercises two-device ownership,
reverse creation order, synchronization of every owner before the first release,
failure of the second owner-device fence before any release, partial startup,
inert uninitialized registry, and terminal get/set/sync/free/stream/event/restore,
invalid-device and capacity controls. Positive cases use sanitizer leak
detection; intentional terminal cases disable it because `_Exit` is expected.

Production HIP gfx1201/CUDA sm_89, full-model and pointer attribution are
**pending Oberon validation**. Do not claim RC1 readiness on this commit.

## Oberon native handoff

1. Fetch the shared `fix/phase11b-prefill-restore-lifetime` branch into the
   existing clean linked Phase 11B worktree. Review the patch and do not
   reapply old patches. Keep original Strata-dev checkout untouched.
2. Build HIP and CUDA separately with **28 parallel jobs**. Use the existing
   ROCm core-10.1, gfx1201 Release settings and CUDA sm_89 targets. Run the
   89-case CPU/sanitizer suite, the complete Python harness including the
   new suite, HIP 67-case native group in each order, and CUDA lifetime 8.
3. Run existing monitored healthy A→B→A full-model tests and controlled
   startup-refusal gate in one physical GPU order; inspect journals, mapping
   and forward evidence before reverse. Keep the established process-group
   timeouts, live journal checks, and stop-on-failure rules.
4. Expand or reuse the pointer observer: match exactly one successful
   owner-device release for the **three pipeline allocations, three streams,
   six events, one refill stream and one refill event** on healthy exit,
   with completion after the last GPU graph use. Test partial pipeline and
   refill creation failure where feasible without changing drivers or
   production model/config assets. Ensure early refusal remains zero.
5. Preserve the deliberately process-lifetime conversation-copy pool
   (three streams and two events, split across devices) as a separate
   documented policy, NOT as a cleanup regression. Investigate any extra
   outstanding pointer or release before declaring the bounded gate passed.
6. Report separate source/executable hashes, per-owner pointer ledger,
   runtime and kernel-journal outcomes. Keep test binaries/logs external;
   commit compact results and a factual report to `docs/hardening/phase11b/`.
   Push only after verifying workflow trigger paths. Do not dispatch Actions,
   open a PR, alter drivers, or touch either heterogeneous project.

## Remaining acceptance limits

Non-serving success/GPU-only exit is not native qualified (previous optional
fixture was rejected for requiring `--serve`). No claim about all allocators,
full-runtime cleanup, native two-NVIDIA layer split, P2P-enabled operation,
long context, performance, or Metal/SYCL is made. RC1 is reserved for later
real-world validation before any pull request.
