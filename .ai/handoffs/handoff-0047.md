# Handoff 0047 — bounded real-model guarded prelaunch

Read .ai/TEST_BUDGET.md, report-0046.md, and
experimental/heterogeneous/docs/BOUNDED_PIPELINE.md. The model-free checked
GPU primitive and ordinary fast pipeline passed 0046; do not rerun their
entire matrix. Chat AI added the bounded live diagnostic
`--guarded-direct-prelaunch` on top of direct verified shared transport.
Its new HELLO kinds must match, GPU publication checks every FP32 element,
HIP0 is prearmed before CUDA dispatch, both HIP edges quarantine rejected
input, and host identity/finite/checksum/commit gates remain. No default
profile changed. Native compilation of these new edits is still unverified.

## Local work
- Existing vanilla Strata toolchain and .venv, incremental host/CUDA/HIP
  build with 28 jobs. No GitHub Actions or dependency installs. Run focused
  host/final_pipeline, CUDA/HIP verifier adapter, handoff guard and
  window-admission tests, plus CPU preflight admission negatives. Fix any
  actual defects (including compile/test failures); verify old direct and
  ordinary protocol paths still work.
- Inspect prearm, ready reset, timeout, shutdown and poison ordering against
  native code. Ensure the worker's prelaunch cannot consume prior-window
  data, native guard failure cannot reach commit, and legacy HELLO cannot
  enter guarded mode. In particular, confirm both HIP copy paths use the
  checked ready ABI, and no shared slot is reused before all commit ACKs.
- Keep this candidate explicitly bounded; do not enable in fast/HTTP/256K.

## GPU budget (aim <=5 minutes, no long tests)
Check availability first, do not interrupt a user server or other process.
Use the existing Q2_0 mixed 15/18/15 2048-capacity full-resident configs,
1 GiB reserve per physical owner, max_t 8, INT8 KV. No spills, reserve
reduction, force kills, driver/model/clock changes or historical numerics.
Use the existing 0046 toolchain/artifacts and exact same config across
reference and candidate (rebuild changed targets).

1. In separate fresh processes, compare legacy direct verified shared and
   guarded direct prelaunch with full STFR capture, same fixed tokens.
   Start with T=1/keep=1 (one window); if clean, T=4/keep=2, 3 windows
   with overlapping input and continuation (report 0040 rows
   248045,846,198,2523; 198,2523,513,32767; 513,32767,264,198).
   For reference use --shared --group-native-hip --hip-split-overlap
   --direct-verified-shared --kv int8; candidate adds --guarded-direct-prelaunch.
   Record exact commands, all accepted prefixes, packet bytes/hashes,
   matching identities, full residency and normal shutdown. Compare the
   entire STFR packet, not just top-1. Any difference is recorded, not
   a reason to start long numerical investigation.
2. With trace enabled only for one bounded candidate if necessary, verify
   actual native prelaunch/consumption and ordering (HIP arm acknowledgement
   before CUDA dispatch; HIP graph queued waiting for checked publication).
   Do not report host trace as GPU kernel timing. No deliberate driver hang,
   native drop fault injection, or long performance test. Stop on an
   uncertain drain or meaningful failure, report surviving PIDs without
   forced kills.
3. Optional existing server HTTP smoke only if there is an authorized
   already-running service, no interruption and enough budget; otherwise
   omit. Do not launch additional 256K servers just to smoke.
4. No full A/B/B/A, context ladder, production promotion or new numerical
   sweep. Report what remains unverified.

## Return
Report .ai/reports/report-0047.md, commit/push to feature/heterogeneous-hip-only,
then stop. Include status, source/binary IDs, results, errors, GPU admission
wall time, cleanup status and evidence limits. Keep private traces/captures
under umask 077; preserve unrelated untracked pool probe. If GPUs unavailable,
complete local build/tests and report genuine hardware blocker.
