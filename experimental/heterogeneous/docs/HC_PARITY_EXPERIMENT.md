# Experimental HC numerical parity

Candidate only, derived from hardware evidence 8f7e7a5b and the all-layer trace implementation e3ae3ce5. Default execution is unchanged. The local layer-zero BF16 attribution does not establish the whole-model cause.

| Environment mode | HC projection operands | Downstream prompt images |
|---|---|---|
| unset / off | existing BF16 | unchanged |
| bf16x2 | existing BF16 high plus BF16 residual, two existing GEMMs | existing BF16/FP16 |
| fp32 | existing native fused T=1 verifier read, FP32 activation/BF16 weights | existing BF16/FP16 |

Set STRATA_PREFILL_HC_EXPERIMENT and optionally STRATA_PREFILL_HC_EXPERIMENT_BACKEND=all/cuda/hip in a fresh process. The global STRATA_PREFILL_BF16X2 must be off. Heterogeneous coordinator requires canonical bounded stage/state A/B/A and disallows resident/MTP/first-HC probes; mode recorded in marker and actual worker logging. Layer-major and unsupported weight/activation layouts refuse rather than silently substitute. The experimental FP32 arm reuses an operator with a different accumulation order, no pending-write fold, no Q8 output, and no persistent read; it does not guarantee serial bit equality even with identical inputs. The compensated path tests precision specifically without enabling low activation images for router or other projections. Neither arm is a production solution.

The existing trace analyzer requires same-mode traced/untraced baselines. The stage gate rejects mixed precision modes within a pair or matrix. Keep serial control, native wire/reference replay, interop, numerical quality and admission distinct; KL <= 0.001 and prediction/top-1 unchanged. Bulk+MTP remains disabled. For experimental execution and evaluation see .ai/handoffs/handoff-0027-9.md. No hardware results yet.
