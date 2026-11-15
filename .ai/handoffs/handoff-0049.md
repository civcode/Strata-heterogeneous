# Handoff 0049 — HIP-only MTP and committed-token streaming

Read .ai/TEST_BUDGET.md, .ai/reports/report-0048.md, and
experimental/heterogeneous/docs/MTP_RESIDENT_SERVICE.md.

0048 passed mixed buffered Q8 MTP, identical short output, 33 proposed/
18 accepted, variable A/B/A. Do not repeat the full matrix. New changes:
(1) admit one grouped HIP domain (24/24) for request-local service only;
(2) per-drafter native embedding pointer, never publish terminal HIP1's
embedding globally to HIP0's layer-zero target verifier;
(3) explicit committed MTP stream batch/terminal protocol supporting
incremental output and cancellation at an existing commit checkpoint.
The existing 256K/ordinary/nonservice diagnostic paths remain unchanged.
Chat AI ran isolated C++ stream cursor 18,248 assertions and Python three
stream contract tests against matching blobs. Native integrated compilation,
adapter/CLI and hardware tests remain outstanding.

## Local work
Use existing vanilla toolchain/.venv, incremental builds with 28 jobs.
Rebuild affected host, CUDA and HIP targets. Run focused final_pipeline,
host, window_admission, HIP verifier adapter, mtp_prefix_cursor,
mtp_request_policy, new mtp_stream_batch tests; Python resident,
HIP-only CLI, and tools.test_heterogeneous_mtp_stream_contract. Verify
legacy fixed MTP admission and ordinary one-token/buffered progress.
Fix actual integration defects directly and commit them. No Actions,
dependency installs or modifications to vanilla, unrelated probe/model
assets, drivers, clocks or existing server configurations.

## GPU scope — aim at 1–2 minutes, maximum five-minute admission budget
Use fresh process/listener and GPU checks. If idle, testing is authorized
as for resumed 0048; if any inference or user service is active, do not
interrupt/contact it, complete local work and report blocked hardware.
No 256K stress, long prompts, extended benchmarks, provenance sweep,
numerical campaign, spill, reserve reduction, forced kills.

- Existing exact Q2_0 2048 mixed and HIP-only configurations (full expert
  maps, >=1 GiB reserve, max_t8, no arena caps); no repack or downloads.
  Use approved pinned MTP runtime. One short HIP-only ordinary reference
  and one streamed MTP process, same seven token 23066 prompt as report
  0048, 32 generated tokens, chunk8. Prefer matching overlap where
  supported, do not claim speedup. Check actual Q8 target/draft state,
  terminal device ownership, no global embedding pointer, proposal/
  acceptance counts, output IDs, committed positions, both owners'
  residency/reserve and normal teardown. No CUDA worker in HIP-only.
- In the streamed candidate, reuse process for brief A/B/A, ideally cancel
  a request during a multi-token batch yield and check native cancelled
  ACK and equivalent short request after reset. Compare streamed output
  (excluding stop ID) against final record; validate real nonempty batches,
  no duplicate anchor, no unaccepted token, capped last output after final
  validation, stop handling. Do not require a particular acceptance rate.
  Short mixed streaming regression on same IDs as 0048 if remaining
  budget allows. Preserve any discrepancies without investigating old
  numerical drift. No long repeated tests.
- If resources and budget remain, optional one short streaming chat
  request against a newly launched private 2K MTP HTTP service on a free
  loopback port, after other native workers exited; use explicit small
  max_tokens, record SSE/finish and clean shutdown. Do not contact an
  existing server. Omit if no time. Existing 256K launchers unchanged.

Stop admitting GPU work at budget; allow safe drain/cancel, no forced kills,
and stop on any uncertain teardown. Private evidence umask077.
Report .ai/reports/report-0049.md with actual commands, binary/source IDs,
test results, accepted/proposed counts, streamed/record/cancel verification,
memory/cleanup observations, wall time and unattempted cases. Commit and
push only related changes to feature/heterogeneous-hip-only, then stop.
Do not claim production/256K MTP admission or robust performance.
