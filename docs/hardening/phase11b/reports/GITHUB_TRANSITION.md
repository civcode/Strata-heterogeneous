# GitHub collaboration baseline — 2026-11-08

## Identity and preservation

Implementation worktree: `/home/chris/workspace/Strata-worktrees/phase11b-prefill-restore`.
Branch: `fix/phase11b-prefill-restore-lifetime`.
Parent HEAD: `0ef75af1730c9374bbf1cb66a361f410d699b805`.
Origin: `https://github.com/civcode/strata-dev-private.git`.
Origin's existing branch at inspection: `debug/isa-guard-4349`,
`032b659dd4a5a150e1665943e9f40c7953b2cff4`.
Fetched origin before checks. The target remote branch did not yet exist.

All 53 original changed/new files exactly matched the final historical per-file
SHA-256 manifest; no missing, additional or mismatching baseline source files.
That manifest is archived under `layer-split-startup-failure/evidence/`.
No history rewrite or production source adjustment was made. The baseline commit
also adds the report archive and collaboration notes. Its Git tree is the complete
published identity; historical source manifests identify their earlier subsets.

## Fresh checks during this transition

Passed:
- Rebuilt the ten relevant CPU targets; focused CTest group: **89/89**.
- Rebuilt the same targets under ASan/UBSan; with `ASAN_OPTIONS=detect_leaks=1`
  and `UBSAN_OPTIONS=halt_on_error=1`: **89/89**.
- `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests/tools -p 'phase11b*test.py'`:
  **26/26**, including the gate and startup-refusal harness controls.
- Exact 53-file source-identity comparison against the last validated state.

Build directories: `/home/chris/workspace/Strata-builds/phase11b-cpu` and
`phase11b-sanitizers`. Build targets:
`expert_cache_completion_test peer_experts_completion_test remote_opt_completion_test
transfer_failure_test conversation_lifetime_cuda_test conversation_lifetime_hip_test
prefill_completion_test pinned_ring_test conv_cache_test conversation_restore_orchestration_test`.
Commands used `cmake --build <directory> --target <targets> -j 4`, then
`ctest --test-dir <directory> --output-on-failure -R` with:

```text
^(expert_cache_completion_|peer_experts_completion_|remote_opt_completion_|transfer_|lifetime_|prefill_completion_test$|pinned_ring_test$|conv_cache_test$|conversation_restore_orchestration_test$)
```

Raw fresh logs remain in `/home/chris/workspace/Strata-phase11b/github-transition/`;
SHA-256 identities are recorded in `TRANSITION_EVIDENCE.sha256` beside this report.
No failed fresh tests. GPU tests/builds were not repeated for this unchanged-source
archive transition. No new native validation of the missing stage-owner correction
is claimed.

## Preserved native results and limits

These are verified historical report/source identities, not fresh hardware runs:
- R3 device-scope correction: HIP gfx1201 and CUDA 12.0 sm_89 builds; CPU and
  sanitizer groups 89/89 each; native owner-device cases both directions, cache
  cases 26 + 5, R1/R2 regressions 42, and a rejecting old-policy negative control.
- Peer primary-device follow-up: HIP and CUDA builds; **13/13** host-staged native
  invocations, including logical primary 1 in both physical GPU orders.
- Healthy real-model layer-split A/B/A: passed both physical GPU orders; A-return
  restored 617 parked tokens in each run.
- Later-stage stream-creation refusal: passed once per physical order; expected
  exit 1, injection/refusal markers, no READY, no lingering owned process group.
- Recorded native windows had no matches for the established kernel-fault stop
  conditions. This is not broad hardware stability qualification.

Limits remain: stage-owner correction absent; native CUDA unavailable in the
historical probes; P2P, Metal/SYCL, long-context/performance and wider failure-path
qualification open. Historical TSan and Compute Sanitizer environment failures
are retained, not passes. Startup stream refusal does not establish every later
GpuStage allocation-owner/unwind path.

## Actions safety review

All eight workflow definitions were inspected. Only
`terminal-minimal-server.yml` and `terminal-portable-monitor.yml` subscribe to push.
Their filters cover Terminal/server/packaging paths; neither matches baseline
source, tests or report paths. Other workflows require PR/issue/comment/schedule
or manual events; this task creates none of those events.

Checked both staged paths and the conservative full outgoing range from existing
origin tip `032b659dd4a5a150e1665943e9f40c7953b2cff4`, including the unpublished
Phase 10E parent. No workflow file is changed. Only two commits are outgoing for
initial publication, well below the 1,000-commit unconditional-run threshold.
Explicit branch-only push; no tags, PR, workflow dispatch or Actions-setting change.

GitHub documents a different comparison for new branches, so checking staged
paths alone is insufficient:
https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow#git-diff-comparisons

## Blocker

The requested `STRATA_PHASE11B_LAYER_SPLIT_STAGE_OWNER_INCREMENTAL.patch` was not
found in the Phase 11B evidence area, relevant temporary input directories or
Downloads. Do not infer its content or treat it as applied. No correction commit
or correction-validation report exists. Continue with the published GitHub branch
once the missing change is supplied.

No protected-checkout file edits, heterogeneous-directory access, driver/system
changes, force pushes, rebases, default-branch merges or PR creation.
