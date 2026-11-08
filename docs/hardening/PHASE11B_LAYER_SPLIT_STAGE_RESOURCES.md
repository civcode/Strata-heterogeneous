# Phase 11B — later-stage raw allocation cleanup (pending Oberon validation)

## Source finding

The stage-owner validation reported 8 outstanding hipMalloc allocations per healthy model exit
and 1 per later-stage stream-create refusal, despite no observed wrong-device or
failed frees. This alone does not attribute allocations to owners or prove a leak.

Source inspection finds that the later GpuStage's weight arena is allocated
before its streams are created but had no owner for release. The session arena,
residency table, optional vision table, two streams and event likewise lacked
a stage-scoped cleanup path. The refusal's single outstanding allocation is
consistent with the weight arena, but that pointer has not been matched yet.

## Implemented

LayerSplitStageResources is the first nontrivial GpuStage member, destroyed
after the stage's Verifier, Prefill and other dependent members. The existing
LayerSplitStageDelete selects and fences the owner GPU before stage deletion.
The cleanup owns four raw allocations and three stream/event handles. Each
allocation is recorded immediately after the actual API call. On release failure
the existing transfer_terminal boundary stops further unsafe cleanup.
No normal-path synchronization is added; teardown already performed the fence.

## Local checks

GCC/Clang optimized and ASan/UBSan fake-runtime tests: 6 scenarios each, all
passed (24 child scenarios, 4 Python unittest methods). They check owner-device
selection, destruction order, all seven handles, unarmed stages, and terminal
release failure. Production HIP/CUDA builds and real-model tests are NOT YET
verified for this commit. Do not describe it as validated before Oberon runs.

## Oberon acceptance (source branch workflow)

1. Fetch the shared branch into the existing linked Phase 11B worktree; inspect
   the commit and ensure there are no concurrent or uncommitted edits. Do not
   reapply historical patches.
2. Compile HIP and CUDA production targets locally with 28 parallel jobs
   (cmake --build BUILD_DIR --parallel 28), retaining build logs.
3. Run the focused Python/fake suite and existing CPU 89-case, sanitizer, and
   harness tests, then targeted native HIP/CUDA regressions.
4. Before either real-model fault test, retain the same physical UUID/PCI
   mapping verification and kernel-journal fault stop conditions. Run bounded
   healthy A/B/A and single stream-create refusal in one order; review logs
   before reverse. No P2P is required.
5. Repeat the HIP allocation audit with pointer, allocation/free device, size,
   and callsite/caller attribution for each unmatched allocation. Compare
   healthy and refusal outcomes to historical 8 and 1; do not accept merely
   a lower count as proof of full cleanup. Track stream/event create/destroy
   separately from hipMalloc and hipFree.
6. Stop for a wrong-device release, failed GPU free, GPU fault, crash, hang,
   or unexpected new allocation. Commit a separate validation report and
   push the source/report branch only after inspecting triggered-workflow
   filters; never manually dispatch GitHub Actions.

## Open limits

Some remaining allocations may originate outside GpuStage or use other HIP
allocators. Native CUDA lifetime 8/8 does not qualify two-NVIDIA layer split.
P2P-enabled behavior, all failure paths, long contexts and performance remain
separate qualifications. Do not modify drivers, user config/profiles, the
protected Strata-dev checkout, or either heterogeneous project.
