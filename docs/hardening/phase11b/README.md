# Phase 11B shared development

Shared branch: [fix/phase11b-prefill-restore-lifetime](https://github.com/civcode/strata-dev-private/tree/fix/phase11b-prefill-restore-lifetime).
Use GitHub commits for subsequent reviews and corrections, not downloadable patches.

## Preserved baseline

The baseline retains parent `0ef75af1730c9374bbf1cb66a361f410d699b805` and all 53
previously validated changed/new files. Their contents match the SHA-256 manifest in
[the final source identity](reports/layer-split-startup-failure/evidence/source-identity.json).
The commit containing this note identifies the preserved baseline in Git.
No production source was changed during the collaboration transition.

[Transition checks](reports/GITHUB_TRANSITION.md) separate fresh local checks from
historical native validation. The archived reports are snapshots: statements such
as “uncommitted” and “no push” describe their original runs, not current status.
Read them in this order: implementation, chatgpt-delta, r2-delta, r1-delta,
r3-delta, r3-device-scope, peer-primary, layer-split-gate, layer-split-startup-failure.
The failed R3 device-scope diagnostic is historical; the next report records its correction.

Reports, source/executable identities and compact native result records are in
`reports/`. Each `EXTERNAL_EVIDENCE.sha256` hashes that step's original external
artifacts relative to its directory under `/home/chris/workspace/Strata-phase11b`.
Those manifests identify evidence; they do not make raw external artifacts available
from GitHub. Large logs, binaries, patches, archives and model assets stay outside Git.
Historical report links to omitted raw artifacts refer to those external directories.

## Next iteration

1. In the existing linked Phase 11B worktree, check status and `git fetch origin`.
2. Compare HEAD with the shared remote branch. If they diverge or there are local
   edits, stop and coordinate. Do not force-push, rebase or merge the default branch.
3. Record the exact source commit before validation. Review any incoming commits
   before testing; only fast-forward a clean worktree when appropriate.
4. Run focused local and bounded native regressions with the established physical
   GPU mapping, process ownership and kernel-journal stop controls. Keep failures.
5. Commit tested source corrections and their validation reports separately.
   Review staged contents and run `git diff --cached --check` before each commit.
6. Inspect the complete outgoing range against every workflow's push filters,
   including inherited commits when creating a branch. Stop if Actions would run.
   Do not edit workflows or repository Actions settings to bypass this constraint.
7. Push explicitly to this branch, without force, and give ChatGPT its URL and SHAs.

The requested `STRATA_PHASE11B_LAYER_SPLIT_STAGE_OWNER_INCREMENTAL.patch` was not
available during this transition. It has not been applied or validated. Preserve
this baseline before integrating that correction, preferably supplied as a GitHub
commit on the shared collaboration path. Do not repeat previously applied patches.
