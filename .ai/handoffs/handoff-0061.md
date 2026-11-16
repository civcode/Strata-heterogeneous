# Handoff 0061 — mandatory native application compatibility

Start at the branch head handed back by Chat AI. Review .ai/VANILLA_FIRST.md,
.ai/APP_COMPATIBILITY.md, .ai/PERFORMANCE_ROADMAP.md, .ai/TEST_BUDGET.md,
.ai/reports/report-0060.md. The 0060 GPU performance arm remains unqualified;
do not run it before the compatibility/source gates.

1. Read exact pinned commit 8db7ddc46cd2575514408c4ab7dd878f69df2f5f
   from the authorized /home/chris/workspace/Strata object store, never its
   dirty files or Strata-dev. Finish net changed-path classification. Use
   git show pin:path and actual call graphs, no independent algorithm.
2. Validate this increment locally first: existing .venv, pip check;
   python -m unittest tools.test_strata_server_launch
   serve.test_heterogeneous_launch_compat tools.test_heterogeneous_oberon_launch
   tools.test_heterogeneous_hip_only_cli serve.test_heterogeneous_resident
   serve.test_heterogeneous_http_disconnect serve.test_heterogeneous_retirement
   serve.test_context_overflow (one command, all modules); run native setup,
   launcher, child-env, security, lifecycle, web and HTTP protocol tests
   affected by shared functions. Use shell syntax checks on run scripts.
   Check args/open/browser logic in setup and terminal start paths. Check
   actual log/trace destinations, default quiet terminal, ready and
   cooperative teardown. Fix concrete regressions before other changes.
3. Complete native option/config extraction and public launch integration:
   all 24 native options and 16 config keys in APP_COMPATIBILITY.md;
   use native parser/defaults/precedence, derive worker-private configs.
   Do not invent a separate parser or claim ignored flags are supported.
   Resolve native --spec 4 entire-window vs legacy candidate count
   while preserving and explicitly versioning legacy test contracts.
   Use native prefill, cache, sampling, control vector and MTP caller;
   reuse existing engine/lifecycle and actual worker native dispatch.
   Preserve stage capacity, all-owner/finite/identity gates, native grouped
   HIP and cooperative drain. Remove experimental alternatives only after
   native replacement is qualified. Do not silently change baseline.
4. Complete HTTP and normal terminal parity using native shared APIs,
   startup narrator, progress and engine protocol, actual feature flags.
   Do not clone HTTP service, sampler, scheduler or cache policy.
   Compare normal output and logs separately; no fake startup messages.
5. Run CPU/native builds only first (28 jobs), focused then full relevant
   tests. Only after passing source/CPU gates, and if idle, perform bounded
   same-model same-effective-settings short GPU verification; no 160 tok/s
   benchmark until parity accepted. No GitHub Actions, forced kill, reset,
   production changes or unrelated cleanup. If pin unavailable or full
   work cannot be completed, leave remaining settings explicitly rejected
   and report incomplete; do not substitute guessed compatibility.

Return .ai/reports/report-0061.md with exact source/reference revisions,
files/options classified, concrete code changes, local test output,
remaining gaps or blockers, true runtime/console/API evidence and
cooperative retirement. Commit/push safely. No unsupported parity claims.
