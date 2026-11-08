# Next work

## Risk status

| Risk | Current status / next action |
|---|---|
| R1 PeerExperts teardown | Unchanged. Coordinate caller-owned loans and helper owners before teardown on unknown completion. |
| R2 RemoteExpertOpt teardown | Unchanged. Cover primary mapped readers and remote stream completion before freeing shared allocations. |
| R3 ExpertCache startup | Still unproven as a runtime lifetime defect. Trace partial startup/error drains before proposing changes. |
| R4 Prefill terminal propagation | Corrected; CPU cleanup/cancellation coverage and actual HIP final-copy terminal requests pass. Native helper/KV cleanup fault paths and layer-split coverage remain. |
| R5 restoration operands | Copy/drain/no-unwind correction implemented; CPU whole/borrow/conversion gate coverage, native HIP copies/snapshots and real model disk restore pass. CUDA runtime and full real-model borrowed/conversion failures remain. |
| R6 CUDA async publication | Returned-error join and worker drain gaps corrected fail-closed. Native CUDA async checkpoint/park execution, C++ worker exceptions and queued resource-admission interactions still need dedicated tests. Keep opt-in defaults unchanged. |
| R7 retention | Production unknown readback completion now terminates rather than retrying. A general global retention/retry policy for other owners remains unresolved. |
| R8 integration/performance | One bounded HIP model scenario passes. CUDA inference, long contexts, layer splits, comprehensive cancellation and performance comparisons remain. |

## Recommended next batch

1. Restore NVIDIA device availability outside this source task, without treating the successful build as runtime proof.
   Repair the existing Compute Sanitizer installation separately if permitted; then run the prepared native copy/ring/snapshot tests, selected memcheck and model restore scenario on sm_89.
2. Add focused native R4 helper/KV and clean-cancel request tests; extend R5 to real-model borrowed-prefix and borrowed-to-whole scenarios with identical segmentation.
3. Address the observed no-MTP disk session SAVE defect separately: serve supports optional MTP, but disk save/read/validate/restore paths still pass mtp.kv_state() unconditionally. Use the existing pointer-capable snapshot contracts where possible; preserve session format validation. Do not silently save a dummy draft layer.
4. Finish R6 async worker failure/exception and owner-join coverage on CUDA. Avoid a new recovery/lease architecture unless evidence requires it.
5. Implement R1/R2 together with their actual caller ownership boundaries, rather than only retaining private helper objects.
6. Keep AMD workloads serialized and bounded until the historical MES issue has a separate diagnostic plan. SDMA-on component success is not long-context stability evidence.

## Performance

No global fence or extra allocation was added to successful restore/prefill. Error/cancellation drains can wait; that is intentional before releasing operands. CUDA <12.1 wide expert batches are capped at 128, so opt-in batches above 128 can take additional launches. Measure that on working NVIDIA hardware before recommending it for speed.

The ring fixture's timing output and single model timings are observations, not controlled before/after benchmarks. Run matched snapshot, prefill and expert-transfer measurements after native correctness coverage is complete.

## Contribution readiness

Keep implementation uncommitted for review. Suggested eventual contribution units:
- R4 completion propagation, cleanup and caller-boundary regression.
- R5 copy lifetime plus directly necessary R6 fail-closed publication tests.
- CUDA <12.1 batch-parameter compatibility.

Shared fixtures will need careful partitioning; do not rewrite the checkpoint or pretend these are already independent PRs.
CUDA runtime, TSan/Compute Sanitizer and touched-backend closure limits prevent a production-ready or fully backend-qualified claim.
No PR/issue/publication/push is authorized. Retain upstream coordination/dependency requirements from the earlier handoff.
