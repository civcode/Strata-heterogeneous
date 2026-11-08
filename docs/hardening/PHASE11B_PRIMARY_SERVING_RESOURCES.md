# Phase 11B — primary serving scratch resource ownership (WIP)

## Evidence and scope

Source parent is validated report c13af36e53a05b70b67960fa158a8b8944fa648e.
The pointer ledger at phase11b/reports/primary-early-resources/evidence/
POINTER_ATTRIBUTION.json attributes 15 remaining primary hipMallocs after
healthy serving exit, plus four primary streams and nine events. Twelve of
these allocations and one event belong to the ordinary serving scratch,
session, token, hit-path, and residency resources. Three other allocations
belong to the pipeline. GPU1's conversation-copy pool is intentionally
process-lifetime and is NOT modified here.

The serving branch returns before non-serving manual cleanup. This change
assigns the 12 serving allocations and hit completion event to a single
PrimaryServingResources owner. It is constructed before later stage and
verifier locals and released after those GPU owners unwind, under the
existing LayerSplitStageDelete owner-device selection/completion boundary.
Each resource is captured before an allocation-failure return, including
partial hit-path and residency group allocations. The GPU-only and normal
non-serving branches stop freeing d_next, d_emb, d_logits, sbuf directly,
preventing double frees. The encode/decode hot path is unchanged.

## Local checks

CPU-only fake CUDA/HIP tests pass with GCC/Clang in Release and ASan/UBSan
configurations, 10 scenarios each (40 child cases; four Python methods).
Covers all twelve buffer frees plus hit event, partial allocation, no
initialization, correct device completion, success, and get/select/sync/
free/event/restore terminal controls. Child processes disable leak detection
because intentional failure modes exit without unwinding; the existing
89-case sanitizer group uses leak detection separately.

**WIP**: production HIP/CUDA compilation, native cleanup and full-model
inference are NOT verified for this commit. Do not represent it as RC1-ready.

## Oberon acceptance

1. Fetch the shared fix/phase11b-prefill-restore-lifetime branch into the
   clean linked Phase 11B worktree. Review the diff; never reapply old patches.
2. Compile production HIP gfx1201 and CUDA sm_89 targets separately,
   each using 28 parallel build jobs. Run all CPU and sanitizer tests,
   Phase11B Python harness and new primary-serving suite.
3. Run native HIP 67/67 in both physical device orders and the CUDA
   8/8 lifetime cases. Keep native groups serial and watch journal/Xid.
4. With existing bounded monitored wrappers, run the healthy real-model
   A→B→A and controlled startup stream refusal in the first physical order.
   Inspect all evidence and live kernel journal before reversing.
5. Use the existing pointer-level observer to match all twelve allocations
   and hit event individually to one successful owning-device release on
   healthy exit, with correct final graph and completion ordering. Refusal
   should retain zero tracked allocations/handles on both GPUs. Confirm
   no duplicate frees in non-serving exits where practicable.
6. Treat any build failure, wrong/free failure, GPU fault, crash, timeout,
   unexpected lingering engine or untracked resource as a stop condition.
   Record source and executable SHA-256, commands, outcomes and residual
   pointer ledger in a separate report commit; push only after workflow
   path-filter review. No GitHub Actions dispatches.

## Remaining

Pipeline buffer/stream/event lifetime remains a separate cleanup gate, as
do intentional conversation-copy process pools, all other allocation
families and CUDA two-device testing. Don't claim whole-process leak
freedom or start RC1 until release-candidate acceptance is explicitly
confirmed. Preserve original Strata-dev checkout, model/config/profile
assets, drivers, and both heterogeneous-project trees.
