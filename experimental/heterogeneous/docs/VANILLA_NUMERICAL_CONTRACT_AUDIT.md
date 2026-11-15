# 0029 — independent numerical contract audit

Source inspected at 614242bb (research fork, not an independent checkout of upstream vanilla). No inference or settings changes. Static source observations, not actual GPU dispatch or measured defects.

| Operation | Prompt contract | Serial contract / risk | Evidence anchor |
|---|---|---|---|
| HC normalize/down/up/inject | FP32 RMS, low precision images, GEMM; optional BF16 low image | fused FP32 activation/BF16 stored weights; order/exp differences | src/prefill/kernels.cu:109–165,269–299; src/core/verify.cpp:1682–1750; src/kernels/cuda/fused_gr.cu |
| HC write | FP32 FMA, 2 sigmoid(injection/HC) | can fold pending write into next read; compare materialized state, not pending R | src/prefill/kernels.cu:300–309; src/core/verify.cpp:1682–1692 |
| GDN gate, conv, recurrence | prompt gate projections, convolution and token recurrence; multiple dispatch variants | verifier gdn_ab_multi, gdn_conv_l2_multi, gdn_step_norm_multi; compare same input/state before attributing temporal error | src/prefill/kernels.cu:315–345,534–585; src/core/verify.cpp:1828–1969 |
| QSA | prompt attention can use a matrix-core path explicitly described as not bitwise | verifier block score/top-k and decode attention, optional split/slot variants | src/prefill/prefill.cpp:4765–4800; src/core/verify.cpp:2240–2340 |
| Router | BF16 projection operand (possibly alternate image/low part) | native routing finite FP32 logits, softmax then top10, tie lower ID and selected normalization floor | src/prefill/prefill.cpp:4817; src/kernels/cuda/native_router.cu:30–100 |
| MoE expert/combine | grouping and combine variants; quantized expert operations | investigate identical selected IDs/weights and actual quantized operands; no verified local arithmetic attribution | src/prefill/prefill.cpp:6337; src/core/verify.cpp |
| GEMM | backend vendor dispatch; hipBLASLt FP32 compute, CUDA GemmEx FP32 compute on low precision operands | same nominal compute precision is not proof of identical order or representation | src/prefill/gemm.cu:239–246,1190–1255 |

CUDA/HIP audit: shared GPU sources have conditional HIP and CUDA dispatch, notably fused_gr.cu and gemm.cu. Actual variant, compiler flags, hardware and operand representations must be read from runtime evidence. No claim of exhaustive four-way kernel equivalence. PLE, expert quantized projections, QSA indexer and cache codecs still need matched-operand adapters; the selected-attention oracle assumes decoded valid selected K/V, not ranking/cache encoding.

## CPU tools
vanilla_numerical_oracle.py: FP64 HC read/write (independent norm/SiLU representation switches), GDN single-head recurrence before output RMS/gate, causal convolution, selected attention, routing and sensitivity, analytical HC cost. FP64 equations deliberately do not reproduce GPU accumulation or intermediate FP32 rounding; BF16x2 reconstructs operands and does not emulate two GEMM output rounding. Router ranking of logits is mathematical, not exact GPU softmax tie behavior. Margin > 2 max absolute logit change is sufficient for unchanged selected set, not necessary, not a guarantee of order, weights or output quality.

Canonical HC per read: 2*10240*320 + 4*10240 = 6,594,560 MACs; 96 reads per token excluding head; 1,266,155,520 projection FLOPs/token. BF16x2 doubles this projection work; FP32 same mathematical work, different implementation. Low image 21,120 bytes per chunk row, not whole scratch/device traffic. No runtime estimate without profiling.

vanilla_numerical_report.py: summarizes validated trace analyzer JSON by ownership; counts route changes and rows where the sufficient margin certificate fails. Reverse topology requires an explicit plan marker from a future validated reverse trace adapter. Not independent raw evidence validation, not a quality gate. Never compare unmatched source/shape/topology or replace existing stage gate.

Run: PYTHONPATH=tools OPENBLAS_NUM_THREADS=1 python -m unittest test_vanilla_numerical_oracle
Then: PYTHONPATH=tools python tools/vanilla_numerical_report.py --trace /absolute/validated-analysis.json --topology forward

Implementation candidate; no local Python execution through connector, integrated or hardware validation not performed. No changes to active Oberon branch. Do not merge while agent owns its branch.
