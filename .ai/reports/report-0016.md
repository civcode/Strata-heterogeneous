# Report 0016 — mapped CUDA egress and warmed fixed windows

**Status: bounded correctness PASS; performance GO/NO-GO: no material benefit demonstrated.** Starting commit `736307b1ae3a8c9125a2fe059109aafba0f8564b`. Keep `--cuda-mapped-egress` opt-in; retain the existing DMA/receiver-arm control and do not change the default. Target remains vanilla layer split CUDA0 [0,12) → HIP0 [12,30) → HIP1 [30,48), two processes, not tensor parallelism. No Actions, inference arithmetic changes, fault injection, dual runtime page registration, altered limits, hardware settings or asset substitution.

## Builds, gates and CLI

Full existing vanilla CUDA, HIP and host CMake builds passed with `-j28`. Host CTest **96/96 passed**, including CUDA mapped-egress, CUDA verifier, final pipeline and HIP adapter fixtures. Checkout `.venv`, inherited `PYTHONPATH` removed: state/parity Python discovery **18 passed**, resident Python **12 passed**, `pip check` clean. Twelve negative CLI checks passed: missing arm/overlap, native prefill, boundary capture, compact result, zero or excessive warmup, absent or >128 token IDs, capture/probe and enabled tracing. Earlier negative-driver expected the later error for two cases; the CLI correctly rejected each earlier and the corrected test records the actual rejection in `negative.log`. All logs and runnable scratch drivers: `/home/chris/tmp/strata-heterogeneous/m7-0016/`.

## Real parity, repeated calls and reset

Original authorized Q2_0 full-resident plan/configs, fingerprint 2026111301, context 2048, max_t 8, unchanged reserves. CUDA ordinal 0 PCI 04:00.0, HIP 0/1 PCI 03:00.0/07:00.0; no expert host misses reported. `run-matrix.py` records exact commands and fixed A02 token IDs. For both shared CPU memfd and ordinary sockets, fresh grouped serial, receiver-arm, and receiver-arm plus mapped CUDA egress ran three one-row/keep-1 windows and three four-row/keep-2 windows. **All eight serial-vs-arm/mapped comparisons passed**: all 3 or 12 evaluated full-logit rows, including discarded suffixes and following positions, had exact predictions/top-1, KL 0 and max logit difference 0. Independent mapped runs repeated each shape/transport; all four repeated comparisons likewise passed bit-identically. `matrix-*-compare.json`, `repeat-*-compare.json` and corresponding `.stfr`, manifests and logs contain the raw evidence. The existing comparator's same-plan `prefill=True` is a comparison mode, not bulk prefill.

`resident-mapped-reset.log`: two identical fixed-prompt requests within one session each returned token 96220, all progress/continue checkpoints completed, then `HETERO_CLOSED` and exit 0. The `final-trace-multi-shared-arm.log` run used mapped egress despite the historical filename: it shows completed producer execution, `mapped_egress_copy`, receiver publication, native prelaunch/consume and accepted-prefix commits. `enqueue_download` and `wait_download` span names remain in the generic lifecycle, but mapped mode performs no separate D2H enqueue/event wait: the observed `wait_download` includes the CPU mapped-copy span. Normal completion required a published ready flag. No productive three-GPU concurrency is inferred from a pending receiver.

## Startup-free fixed-window timing

Shared transport, full logits, identical A02 tokens and context, tracing/capture off. Each run: 32 windows, first 8 warm, 24 recorded after normal execute + accepted-prefix commit. Three independent runs per mode, order rotated across serial, HIP-local overlap, receiver arm and mapped+arm. Rows 1 keep 1 and rows 4 keep 2; 24 complete runs total. Raw `HETERO_FIXED_WINDOW_TIMING` JSON and exact commands: `warm-{one,multi}-{0,1,2}-{serial,overlap,arm,mapped}.log`. `warm-summary.json` holds all results; p95 is nearest-rank and pooled statistics use 72 windows. Values below are median / p95 milliseconds in run order, then pooled median / p95.

| Shape, mode | run 1 | run 2 | run 3 | pooled |
|---|---:|---:|---:|---:|
| 1 serial | 18.349 / 18.442 | 19.026 / 19.544 | 18.928 / 19.430 | 18.748 / 19.461 |
| 1 HIP overlap | 18.330 / 18.959 | 17.956 / 18.550 | 18.402 / 19.195 | 18.253 / 18.959 |
| 1 arm | 17.856 / 18.502 | 17.914 / 18.200 | 18.154 / 18.653 | 17.942 / 18.541 |
| 1 mapped | 17.984 / 18.321 | 17.567 / 18.319 | 18.136 / 18.669 | 17.929 / 18.633 |
| 4 serial | 37.137 / 37.604 | 37.246 / 37.706 | 37.418 / 38.156 | 37.241 / 38.105 |
| 4 HIP overlap | 36.567 / 37.398 | 36.483 / 38.192 | 36.797 / 38.025 | 36.622 / 38.025 |
| 4 arm | 36.044 / 36.425 | 35.821 / 36.659 | 35.946 / 36.568 | 35.931 / 36.601 |
| 4 mapped | 35.881 / 36.422 | 35.944 / 36.870 | 36.085 / 36.711 | 35.939 / 36.795 |

Mapped vs arm pooled medians differ by −0.013 ms (one row) and +0.008 ms (four rows), smaller than run variability; mapped p95 is slightly worse in both samples. This does not establish a material improvement. These are synthetic fixed-token latencies with full-logit transfer, not generated accepted-token throughput, a representative workload, or a claim about the reported 3840 prompt tok/s.

For bounded attribution only, the separately traced four-row mapped run's warm windows 2/3 show producer stage roundtrip 3.835/3.866 ms, execute span 3.339/3.365 ms, mapped CPU copy 4/5 µs, HIP publication 6/6 µs, HIP final execution 22.070/22.148 ms, full-logit download 574/567 µs, and all-stage commit barrier 1.449/1.454 ms. Trace overhead and asynchronous/overlapping spans preclude summing them as a critical path. The CPU transport, producer execution fence, receiver arm/ACK and ordered commit remain. Compare report 0015's separately traced DMA waits only as indicative, not an A/B performance proof.

## Remaining blockers

Native bulk prefill remains off: report 0013 found first-token CUDA committed-state divergence before the bridge. The existing adapter diagnostic compares distinct integration paths and is not an independent same-runtime vanilla Prefill-versus-Verifier test with matched session initialization/adoption; no authorized ready-made matched-state harness was identified, so no arithmetic was changed or long failing campaign repeated. MTP requires explicit approval to read `/home/chris/workspace/Strata-data/mtp/rt` and evidence that its model identity, tensor shapes/quantization and geometry match this exact Q2_0 pack, followed by native head/draft-state ownership, priming, commit/reset and rollback qualification. The referenced directory belongs to another config and was not inspected. The 160 accepted output tok/s target and 3840 prompt tok/s operator figure remain unmatched, workload-dependent and unverified. The pre-existing untracked `tools/heterogeneous_pool_probe.cpp` was left untouched.
