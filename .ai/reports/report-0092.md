# Report 0092 — Partial

Stopped before building as required by task 3: the kept worktree has an untracked source file, `tools/heterogeneous_pool_probe.cpp`. It was already present before the pull and was not changed. There are also pre-existing untracked run artifacts under `.ai/runs/0086-*`, `.ai/runs/0089-safety/` and `.ai/runs/0090-*`. No tracked changes were present before this report.

The explicit instruction to work only in `/home/chris/workspace/Strata-heterogeneous` prevents archiving or removing the other worktrees. None was changed or removed; no archive was created. Their reported HEADs after the pull were:

- `Strata-0084-archive`: `7d573bee6`, detached.
- `Strata-hardened-0072`: `b099f46fd`, `archive/hardened-0069`.
- `Strata-mixed-native`: `0976fe412`, `feature/mixed-native`.

Their full status and contents were not inspected, and no files were copied. The kept worktree is at `0976fe412`; `git pull --ff-only origin feature/mixed-native` succeeded and the front-stage fix `8819f09b` is an ancestor. `.venv/bin/python` exists.

Builds, tests, config generation and the short start were not run. No GPU process was started; GPU time was **0 minutes**, so there was no run to which `HSA_ENABLE_SDMA=0` applied. Handoff 0091 was not run.

Before-work safety evidence is in `.ai/runs/0092-safety/before.txt` (local, untracked). Boot ID `ffcbbb74-121f-46fa-a930-ca08f55e881c`; memory 61,942 MiB total, 58,192 MiB available; swap 359 MiB used of 16,383 MiB. `coredumpctl` reported no dumps since boot. The existing kernel log includes AMD restore-userptr and svm-range workqueue warnings from earlier runs, latest at 20:11:13; no new GPU run was attempted. No after-run measurements apply.

To proceed, clarify whether the pre-existing untracked source is to remain and authorize work on the other worktrees and the external archive path, or revise the handoff to stay entirely in the kept worktree.
