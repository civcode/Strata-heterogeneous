# Vanilla Strata — operator-reported performance baseline

The operator's vanilla Strata results on Oberon are reference observations for the heterogeneous implementation, not independently reproduced measurements or a controlled A/B benchmark.

| Metric | Vanilla Strata (operator-reported) |
| --- | ---: |
| GPUs | 2 × AMD Radeon AI PRO R9700 |
| Prefill throughput (reported 2026-11-15) | reaches ~4,000 tok/s |
| Decode throughput | ~120 tok/s |
| Prefill GPU busy, AMD0 / AMD1 | ~100% / ~100% |
| Decode GPU busy, AMD0 / AMD1 | ~50% / ~100% |

Sources: prior operator observations recorded in [M5_PERFORMANCE.md](M5_PERFORMANCE.md) and [M6_GPU_BOTTLENECKS.md](M6_GPU_BOTTLENECKS.md); additional prefill rate supplied directly by the operator on 2026-11-15. The earlier reports are preserved unchanged. The 4,000 tok/s figure is a reported attained rate, not an established mean, median, sustained rate or target.

For comparison, [report 0032](../../../.ai/reports/report-0032.md) records heterogeneous native-bulk prompt rates of 641.1–697.9 tok/s across 1,378–20,748 input tokens, and decode feedback rates of 42.1–43.9 tok/s in those bulk workloads. These are host commit-progress proxies on a different execution configuration. They are not matched to the vanilla observations. The existing heterogeneous target of 160 sustained accepted single-request decode tok/s remains distinct.

## Reproducible reference needed

Record vanilla revision/executable, exact model and quantization, pack/profile, device placement, prefill chunk and context, prompt/token IDs, completion budget, sampling, MTP/draft settings, warmups/repetitions, timing denominators, GPU telemetry and raw results. Compare identical real workloads and report quality, latency and throughput separately. Do not infer a speedup or regression from unmatched rates. No new vanilla run or source-code change was performed to record these observations.
