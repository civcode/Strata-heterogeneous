# Native HC numerical divergence: evidence and qualification policy

**Status:** the first investigated layer-0 HC discrepancy is strongly attributable to native arithmetic representations. This is not a finding of a CUDA/HIP interoperability defect, not a proof of whole-model equivalence, and not a bulk-prefill admission decision.

## Reproduction and evidence

Hardware-qualified source: [e2dbe960](https://github.com/civcode/Strata-heterogeneous/commit/e2dbe9604af0474a8d2d8e05c816ea2957ff7403). See [0027](../../../.ai/reports/report-0027.md) and [0027.1–3](../../../.ai/reports/report-0027-1-3.md) for the original observations and full numerical investigation.

The approved Q2_0 resident configuration is unchanged: CUDA0 RTX 4070 Ti SUPER (PCI 04:00), HIP0/HIP1 R9700 (03:00/07:00), layers 0–11/12–29/30–47, context 2048, max_t 8, reserve 536870912. The approved source index SHA-256 is c344c01f59f547ed8a59003857a82eb90499e5299154241dbca787acb54b5b7c. First-token capture uses the teacher-forced sequence [248045,846], cut 1 and chunk 1. The independent CUDA replay ran twice for each of the shared and socket captures, with the same real weights and observed dispatch (variant 3, QFUSE off); the replay linked CUDA/cuBLAS/cuBLASLt, not HIP or heterogeneous libraries. Inputs, seven native prompt intermediates, and serial residual/scale/injection/mixed observations matched exactly. The two transports agreed. All 173 other overlapping raw files matched the earlier instrumented run; the completion marker was the sole difference. Raw data remain restricted on Oberon under /home/chris/tmp/strata-heterogeneous/m7-0027-1-3/.

Initial embedding, reset and bridge captures were identical. The first observed HC mixed delta in 0027 was 0.0033674240112 (max absolute); finer 0027.1–3 observation found injection already differs by 0.0015344619751, RMSE 0.0009802998473, while residual and normalization scales match. These are observed checkpoints, not a claim of the first differing floating-point instruction.

## Native arithmetic and independent reference

For each of four residual streams, with width 2560 and rank 320, let x[c,d] = R[c,d] * rs[c] * gamma[c,d], where rs is reciprocal RMS with epsilon represented as float32(1e-6). Flatten x across the four streams. The ideal reference computes d = W_down x, h = SiLU(d/4), u = W_up h, injection = W_inject x and mixed[d] = sum_c x[c,d] * sigmoid(u[c,d])/4.

The native prompt contract uses BF16 round-to-nearest-even operands at normalized x (down and injection) and again after SiLU (up). Its final mixing uses the unrounded normalized value. The fused verifier uses FP32 activations in these operations. The FP64 oracle reproduces the two operand-rounding boundaries with real captured weights and otherwise evaluates high-precision reductions/nonlinearities. It is an explanatory reference, not a replacement GPU inference kernel. The effects of the two boundaries need not add linearly.

| Comparison | Max absolute | RMSE | Relative L2 |
|---|---:|---:|---:|
| Observed prompt minus verifier mixed | 0.0033674240112 | 0.0002918104504 | 0.0003535607753 |
| Prompt vs full FP64 | 0.0033675594682 | 0.0002918302652 | 0.0003535847930 |
| Verifier vs full FP64 | 3.3697067e-7 | 4.7427529e-8 | 5.7463721e-8 |
| Prompt vs both-BF16 reference | 1.0736280e-6 | 6.4655420e-8 | 7.8320206e-8 |
| Verifier vs both-BF16 reference | 0.0033665988639 | 0.0002918013536 | 0.0003534729512 |
| Ideal normalization-only delta | 0.0013681704913 | 0.0000616458616 | 0.0000746908111 |
| Ideal SiLU-only delta | 0.0034866886701 | 0.0002801502869 | 0.0003394332015 |
| Ideal both-boundary delta | 0.0033667343210 | 0.0002918211683 | 0.0003535737711 |
| Observed delta vector minus predicted delta vector | 9.7792227e-7 | 7.7634316e-8 | 0.0002660339 |

The final relative norm uses the predicted delta as denominator: 0.0266%. It is a residual **vector** comparison, not subtraction of two maxima. Captured prompt BF16 normalized and SiLU representations match the corresponding local references exactly; prompt downstream local errors remain small (mixed max 3.01552e-7). Actual fused internal down/up and SiLU are not directly observed; standalone replay materializes additional scratch for analysis. None of these measurements proves the origin or magnitude of every subsequent layer's divergence.

## Remaining numerical-quality failure and policy

The completed 0027.5 bounded stage-local reference matrix passes interoperability on both transports for all five cases, with exact stage residual/state and continuation replay, but fails the unchanged numerical-quality gate. Both transports yield the same results: cut1/chunk1 KL `1.216527297939482e-10` (PASS); cut2/chunk1 `0.001896195949946483`, cut2/chunk2 `0.0015399210952491622`, cut6/chunk3 `0.03153743887084141` and cut6/chunk6 `0.02454066607661273` (FAIL). Cut6 also changes prediction/top-1 from `2724` to `11991`. Matching uninstrumented A/B/A controls reproduce the captured outputs, excluding completion-marker changes. See [0027.5–6 report](../../../.ai/reports/report-0027-5-6.md) for per-case commands, metrics and scope.

Exact bounded native replay is evidence against a discrepancy introduced by the observed live bulk R forwarding for these cases, not proof of full vanilla arithmetic equivalence or of the cause of serial/native quality differences. The earlier HC attribution is limited to its measured layer-zero operation; it does not prove the cause of later divergence. Reset/bridge correctness and local HC attribution do not waive the quality failures. An independent full original vanilla caller comparison remains unavailable within the approved resource/settings constraints (full resident pack 38 GiB, CUDA card 16 GiB); no speculative OOM attempt, repack, precision or residency change.

Keep independent axes:

- Interoperability: compare actual native stage outputs, local committed state, and continuation results with equivalent native operation replay using the same immutable inputs, bypassing live bulk residual forwarding. Require exact equality for the bounded same-shape case and exact serial controls.
- Numerical quality: retain the existing serial-verifier comparison, exact predictions/top-1 and KL <= 0.001; no alternate tolerance or metric.
- Admission: a bounded stage-local replay is not a full vanilla caller, nor proof of correctness for arbitrary shapes or all arithmetic. Even if both bounded axes pass, separate review is required. Bulk+MTP remains disabled; exact target checkpoint conversion provenance remains unresolved.

No HC correction is justified by the observed representation difference. Preserve vanilla arithmetic, precision, hardware and model settings. If subsequent results fail, retain evidence and localize the earliest unexplained difference; do not assume all downstream differences come from HC or from a transport defect.
