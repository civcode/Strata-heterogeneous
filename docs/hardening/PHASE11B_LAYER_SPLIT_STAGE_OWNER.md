# Phase 11B — owner-device destruction for later GPU stages

## Patch identity and scope

Integrated after GitHub baseline `ffe0ceaf68628aa19941b8fde4aae47a19bf9d84`,
including the validated later-stage stream-create refusal. The original handoff was
incremental to the then-dirty Phase 11B worktree. Reported preceding full worktree patch SHA-256:
`e2c530de2fff71cc1e1afd57738c6fe9af7084964680a096153ee6672189366a`.
**Apply this delta only once**, never the old full patch on top of current edits.

Repository starting directory: `/home/chris/workspace/Strata-dev`.
Patch handoff: `/home/chris/tmp/strata-dev/`.
The existing linked Phase 11B worktree is the integration destination.
The two separate heterogeneous-project paths are out of scope.
Follow [the shared GitHub workflow](phase11b/README.md): commit tested corrections
and reports separately and push only after checking Actions filters. Do not run
GitHub Actions, change drivers, or edit user configs.

## Why the code changes

`GpuStage` owns device resources through several member objects and can be
partly initialized when `generate.cpp` returns early. Previously a plain
`std::unique_ptr<GpuStage>` deleted its GPU-owning members on whichever device
the calling thread selected by then. In particular, the local `OnDevice` for
later-stage initialization is destroyed before the containing `stages` vector
is destroyed. GPU stage teardown needs an explicit owner device.

A custom unique_ptr deleter now selects the later stage's `dev` and establishes
completion before deleting the stage and *all of its subobjects*, then restores
the original selection. An `OnDevice` local in a `GpuStage` destructor would be
insufficient because the local is destroyed before the member destructors.
Lookup/selection/fence/restoration errors use the already-existing terminal
transfer boundary; uncertain completion never unwinds GPU-owning members.
A failed preflight never arms this fence or allocates a stage resource.
This changes teardown only; not the prompt/decode hot path.

## Local checks completed by ChatGPT

`tests/tools/phase11b_stage_owner_test.py` compiles a fake-CUDA stage and tests
owner switch, already-owner, no-resource preflight, device lookup failure,
selection failure, fence failure, restore failure, missing owner, and the old
`unique_ptr` negative control. The negative control must still fail with
`WRONG_DEVICE_RELEASE`; the corrected policy must pass.

## Oberon verification

1. Starting from `Strata-dev`, select the existing linked implementation
   worktree and fetch the shared branch before validation. This delta is already
   integrated; do not apply it again. Review changes and `git diff --check`.
2. Run `python3 -m unittest discover -s tests/tools -p phase11b_stage_owner_test.py -v`
   under CPU, and the existing 89+ CPU regression and sanitizer subsets as
   appropriate. Test-only fake runtime must not link into production.
3. Rebuild production `strata` on gfx1201 HIP and sm_89 CUDA. Any build issue
   is a blocker; return exact compiler diagnostics. CUDA runtime remains
   separately unavailable if the NVIDIA device is still inaccessible.
4. Re-run the **already validated healthy two-card layer-split A/B/A** in both
   actual physical GPU directions, serial and bounded with the established
   kernel-journal monitor. Record stage exit status and any teardown errors.
5. Re-run the **already validated one-shot later-stage stream-create refusal**
   in the first GPU order under the same owner/journal supervisor; inspect
   before reverse. The marker and refusal are still required; exit=1 and no
   READY do not alone prove all subobject releases completed. Add read-only
   native allocation-device/release instrumentation where feasible; do not
   weaken the existing interposer acceptance checks.
6. Confirm the previous R1/R2/R3 targeted native regressions still pass if
   the new production integration touches their ownership surfaces.

Stop immediately for a new MES/TLB/page-fault/reset match or a device-release
error. Record results, source commit identities, SHA-256 evidence hashes and limits
in `phase11b/reports/`. Share the GitHub branch and commit SHAs instead of another
full patch. Preserve the original checkout and external raw evidence.

## Limitations

This is an owner-scoping correction, not a proof that every later-stage member
has implemented a complete cleanup routine. Member cleanup implementations may
have their own error/lifetime gaps. Normal exit may pay one stage device fence.
A fake-runtime success does not qualify native destructor behavior, CUDA runtime,
P2P, or long-context inference. Test one bounded fault branch at a time.
