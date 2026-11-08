# Phase 11B implementation and validation

## Result

R4 terminal propagation and R5 restoration-copy lifetime corrections are implemented, uncommitted, in:

`/home/chris/workspace/Strata-worktrees/phase11b-prefill-restore`

Branch: `fix/phase11b-prefill-restore-lifetime`. Base: `0ef75af1730c9374bbf1cb66a361f410d699b805`.
Review patch: [review.patch](review.patch), SHA-256
`204994c9196d71b35044cf2b279477e7b6cbfb5d1756907486462e06106074b9`.
Nineteen changed/new paths; index empty. Per-file identities: evidence/source-identity.json.

### Corrections

- Prefill cleanup and explicit stream/event waits feed the existing terminal predicate. The first error stays sticky after later successful waits.
- Cancellation/early returns drain compute, copy and existing optional KV/PLE streams. Successful runs disarm this guard; no extra healthy-path wait.
- Pipeline joins check terminal state after cleanup scopes. Existing loan-release suppression and generation fail-stop boundaries remain in force.
- Generic conversation H2D/D2D and CUDA direct readback drain reported issue errors. Unknown completion exits before either operand owner can unwind.
- Failed queued ordering drains the copy stream; successful ordering remains device-side.
- Related R6 correction: async workers drain on error before destroying events; checkpoint/park joins terminate rather than merely logging a failed image.
- CUDA <12.1 uses 128-expert wide batches, consistently in host/device code. This removes the reproduced 4216/4096-byte kernel-parameter build failure without changing kernel arithmetic.

## Fresh validation

| Check | Result |
|---|---|
| Relevant CPU group | 55/55 passed |
| Same group, ASan/UBSan + leak detection | 55/55 passed |
| CUDA production executable + selected native targets | Built with CUDA 12.0, sm_89 |
| CUDA final selected runtime checks | 0 passed, 3 skipped: no CUDA-capable device detected |
| HIP production executable + selected native targets | Built for gfx1201 |
| R9700 #0, SDMA off | 19/19 selected native processes passed |
| R9700 #1, SDMA off, independently | 19/19 passed |
| Short SDMA-on comparison | 3/3 passed on each R9700, earlier implementation revision |
| HIP actual model capture/switch/disk restore/continue | Passed; reference and restored next token both `T 1` |
| HIP actual Prefill final-copy fault, later drain succeeds | Expected exit 1, ERR, no answer/DONE |
| HIP actual Prefill final-copy fault, drain reports failure | Expected exit 1 before unwind, no answer/DONE |
| TSan | Both selected processes failed before testing: unexpected memory mapping |
| Compute Sanitizer | Could not start memcheck: missing libsanitizer-collection.so |
| Diff whitespace, including new files | Passed |

The synthetic native snapshot test reports 4195 checks on each AMD card. Native fault wrappers execute and fence valid work before reporting synthetic API errors; they do not fault the GPU. CPU fakes separately model work that remains pending after a failed wait.

No new MES timeout, INVALIDATE_TLBS failure, GPU page fault or reset was observed in the recorded test windows. This is not a long-context stability result.

## Limits and discovered blockers

- NVIDIA is visible on PCI at 04:00.0 but unavailable to NVML/CUDA. Installed driver: 580.178.04. No reset, driver change or other recovery was attempted.
- TSan and Compute Sanitizer environment failures prevent sanitizer qualification in those tools.
- Real-model integration used an existing Qwen Q2_0 pack with a small context, not a newly downloaded small model: context 128, prefill 32, 128 expert slots, one device, CPU prefill share off. It is a diagnostic greedy-token check, not a quality test or a benchmark.
- Without MTP, actual disk SAVE returned `SERR io ... invalid K/V extent`. The disk path unconditionally passes mtp.kv_state(); this optional-MTP integration defect is deferred. Adding the existing local MTP made the bounded restore smoke pass.
- An initial model harness missed SERR and waited for its deadline while the engine was idle on stdin. This was a harness timeout, not observed missing GPU completion; SERR handling was corrected. Other bring-up failures and the first missed Prefill injection selection are preserved, not counted as passes.
- Borrowed/converted restore failures have real gate + production-copy CPU coverage; synthetic native snapshots cover prefix restoration. Real-model borrowed/conversion failure scenarios and native multi-GPU helper cleanup remain untested.
- CUDA async worker runtime, long-context behavior, Metal/SYCL closure and throughput comparisons remain open.

See SOURCE_CHANGES.md, CUDA_VALIDATION.md, HIP_VALIDATION.md, TEST_MATRIX.md and NEXT_STEPS.md.

No implementation commit, push, PR, Actions invocation, production binary/model overwrite, driver/toolchain change or preservation-payload write was performed. Strata-dev remains clean at the checkpoint. The original checkout was read only for existing dependency/profile inputs; those inputs were copied to isolated evidence/build locations before use.
