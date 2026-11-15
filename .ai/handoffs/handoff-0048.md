# Handoff 0048 — qualify the first request-local MTP service candidate

Read experimental/heterogeneous/docs/MTP_RESIDENT_SERVICE.md and
.ai/TEST_BUDGET.md. Report 0047 is PASS for bounded guarded mixed direct
prelaunch; no repeat. Chat AI implemented the variable native resident MTP
candidate, terminal request metadata ACK + HELLO, Q8 draft admission,
target-prefix budget checks, explicit CLI/server options and regression
fixtures. The new C++ policy test passed 85,372 assertions in isolation;
integrated compilation and Python tests still need validation.

## Local first
Use original Strata toolchain and .venv, existing build-256k-host/CUDA/HIP
caches and 28 jobs. Incrementally compile changed host and worker targets;
run focused host/final pipeline, MTP cursor/request policy, HIP adapter
tests and Python resident/CLI tests. Verify old ordinary and fixed-F16 MTP
gates remain unchanged. Fix actual build/integration defects in scope.
Verify new HELLO distinct, request metadata only once before prompt,
exact prompt slices, prefix commit/ACK, draft reset and physical dense
native INT8 state (including code/scale pointers, no host/ring/elastic).
Confirm real reserve and all expert residency. Report any actual native
Q8/asset admission blocker; do not bypass or silently switch precision.
Reuse only the approved existing runtime /home/chris/workspace/Strata-data/mtp/rt
if it matches the pinned runtime identity; no downloads or repack. Target
conversion provenance is unresolved; this is explicitly experimental.

## GPU work (aim <=5 minutes total, no long tests)
If a user's service or other inference is active, do not stop or interfere.
Complete CPU work first; report hardware blocked if necessary. Otherwise
use existing exact Q2_0 mixed 2048 stage configs, INT8, full expert
residency, >=1 GiB reserve each. No GPU clock/driver changes, spill, force
kill or reserve reduction.

1. Native buffered serial non-MTP vs MTP with equal short prompt IDs,
   same prompt_chunk and actual budget (e.g. 32 or 64 completions).
   Use --resident --shared --group-native-hip --kv int8 and source-bound
   CLI. Candidate adds --mtp /home/chris/workspace/Strata-data/mtp/rt
   --spec 4 --spec-min-p 0.5; native uses --mtp-resident-service and
   --hip-split-overlap (match HIP overlap for native reference if feasible).
   Collect actual accepted/proposed counts, generated IDs, stop/length,
   request IDs, output counts, host timings and residency. Same target
   greedy policy, but different verifier window shapes may expose known
   arithmetic divergence. Preserve and report differences, do not force
   matches or launch a numerical campaign. One short warm run per arm.
2. Candidate resident A/B/A using different prompt lengths and completion
   budgets, optionally one committed-boundary cancellation and recovery
   if time remains. Reuse process set and verify resets, committed extents,
   actual draft acceptance (do not force predictions), normal teardown.
   Use a short native stdin driver if necessary (one max_new followed by
   token IDs, continue or cancel at each progress). Verify no survivors.
3. If budget and authorized exclusive GPU access remain, use a new private
   2K HTTP server JSON from documented CLI and one short nonstream chat
   request. Do not change existing production launchers or contact user
   server without permission. Normal owner shutdown only. Otherwise omit.

Stop admitting GPU tests when budget expires, allow safe completion or
cooperative cancellation. No 256K stress, full benchmark matrix, long
provenance or numerical sweep. Never force GPU kills to meet a time budget.

Return .ai/reports/report-0048.md, commit/push to feature/heterogeneous-hip-only,
report scoped results, source/binary IDs, actual commands and blockers,
acceptance counts, cleanup and total GPU work. Preserve unrelated untracked
files and private artifacts (umask 077). Stop editing.
