# 0030 — common HC representation and numerical contract

**Partial diagnostic execution; not qualified.** Tested source `45832916209248eaa3bdce9885be440425eef0f1` with only the subsequent Python test-fixture correction described below. Default inference was not changed. The principal all-backend BF16-weight activation experiments completed on the actual CUDA–HIP–HIP chain; the complete matched-operand CUDA/HIP/Q8 framework, representative statistical evaluation, owner-specific timings and peak memory remain open.

## Forward GPU measurements

`default` leaves both weight variables unset; `bf16` requests `STRATA_HETERO_HC_WEIGHT_CONTRACT=bf16 STRATA_HC_Q8=0`; `q8` requests the q8 label with `STRATA_HC_Q8=1`. The labels below indicate requested loader contracts, not by themselves proof of every effective serial projection pointer. The BF16x2 and FP32 rows use bf16-weight requests on **all owners**. Predictions are serial/native. KL is serial || native; quality threshold remains KL ≤ 0.001 with exact prediction and top-1.

| Cut/chunk | Weight request | Activation | Serial/native prediction | KL | Max abs / RMSE logits | Quality | Same-mode interop |
|---|---|---|---:|---:|---:|---|---|
| 2/1 | default | original | 96220 / 96220 | 0.001896 | 0.283430 / 0.059587 | FAIL | PASS |
| 2/1 | bf16 | original | 96220 / 96220 | 0.004323 | 0.436231 / 0.101310 | FAIL | PASS |
| 2/1 | bf16 | BF16x2/all | 96220 / 96220 | 0.002576 | 0.404895 / 0.074342 | FAIL | PASS |
| 2/1 | bf16 | FP32/all | 96220 / 96220 | 0.002707 | 0.379900 / 0.076803 | FAIL | PASS |
| 2/1 | q8 | original | 96220 / 96220 | 0.002801 | 0.423999 / 0.081229 | FAIL | PASS |
| 6/6 | default | original | 2724 / 11991 | 0.024541 | 1.187996 / 0.214580 | FAIL | PASS |
| 6/6 | bf16 | original | 11991 / 11991 | 0.046961 | 1.926822 / 0.305024 | FAIL | PASS |
| 6/6 | bf16 | BF16x2/all | 11991 / 11991 | 0.020682 | 1.301270 / 0.208959 | FAIL | PASS |
| 6/6 | bf16 | FP32/all | 11991 / 11991 | 0.027464 | 1.156836 / 0.236674 | FAIL | PASS |
| 6/6 | q8 | original | 11991 / 11991 | 0.117259 | 2.759242 / 0.510209 | FAIL | PASS |

Both shared and socket were run and yielded identical captured bytes except the `.prefill.json` transport field. Each wire has a separately executed, same-mode file-fed reference; all 20 pairs have PASS interop, exact local links/commits, same-mode state and continuation, and exact serial A/B/A. All 8 bf16 activation-vs-bf16-original analyzers exited 0; quality failures did not stop cases. BF16x2/all and FP32/all lower KL relative to the **same BF16-weight baseline** for these two shapes, but neither passes; neither changes the prediction from that baseline. Switching the weight request changes serial prediction for 6/6 relative to default; it is not evidence of better intelligence. FP32 reuses a fused T=1 HC read and changes reduction order; BF16x2 changes HC only, not full-model arithmetic. Other cut/chunk shapes and backend-local intervention scopes: NOT_RUN.

CUDA, HIP0 and HIP1 logs all record requested and effective BF16x2/FP32 for their owned ranges. In the completed activation arms, the actual per-layer experiment guard checks norm kind/size and each down/up/inject resident BF16 pointer, dimensions, byte size and absence of Q8 pointers before dispatch, providing stronger actual prompt operand evidence than the label alone. Three Q8 loader allocation messages appear on q8 serial and native runs; the default reports two HIP Q8 allocations, while bf16 reports none. The 0.62 GiB Q8 copies are generated from BF16 or loaded per `src/core/native_dense.cpp`; these aggregate messages do **not** verify each serial projection's active pointer, source bytes, Q8 scale/signed values or implicit conversion. Thus a fully verified common serial-Q8 operand contract is still NOT_ESTABLISHED. No Q8 inject override, guard removal or Q8 activation experiment was used.

## Serial topology and transport

Both ungrouped serial plans ran seven fixed token IDs `248045,846,198,2523,513,32767,264`, three fresh processes per requested weight mode and transport. Each same-plan/mode STFR is repeatable and shared/socket bytes match; 12 separate boundary processes using the corresponding explicit loader selection (without a contract marker) matched their placement logits and each passed 14/14 finite physical send/receive byte comparisons. All logged full residency and zero host misses on three owners. Cross-topology comparisons use identical requested modes, not mixed-contract comparisons:

| Weight request | Max KL forward → reverse | Max abs / RMSE | Prediction disagreements (7 rows) | Gate |
|---|---:|---:|---:|---|
| default | 0.046914 | 1.489322 / 0.277559 | 1 | FAIL |
| bf16 | 0.021771 | 1.260249 / 0.224810 | 0 | FAIL |
| q8 | 0.104898 | 2.584261 / 0.442034 | 0 | FAIL |

These are bounded same-request topology results, not proof all effective weights are identical. Placement, head owner, backend and reduction order also differ. Unchanged default wire captures for both shapes and both transports match the immutable 0027.5 inventory byte-for-byte including marker. Invalid, missing and mismatched weight contract preflights refused; matching bf16/q8 preflights accepted.

## Tracing and FP64 replay

Six shared all-layer traces cover both shapes and each BF16-requested activation mode. Every trace analyzer exited 0 against its **same-mode** trace-off capture; `vanilla_numerical_report.py` also completed. For cut2/1, earliest difference is layer 0 `attn_mixed` for prompt positions 0 and 1 under off/BF16x2, but layer 0 `attn_bo` under FP32; the continuation position 2 first differs at `attn_bo` in all modes. First changed expert set is layer 4 at position 1 for all three. For cut6/6, first prompt-position field is `attn_mixed` with off/BF16x2, `attn_bo` with FP32; continuation 6 first differs at `attn_bo` for all. First route-set changes vary by position/mode (full trace JSON retained); a route observation alone is not causal attribution to final logits. Traces for socket and other shapes are NOT_RUN.

Fresh standalone CUDA replay of the immutable 0027.1 captured real HC operands reproduced native operator A/B/A and actual prompt intermediates. The independent FP64 calculation gives prompt mixed-output max abs error 0.00336756 vs full FP64 and 1.07363e-6 vs the BF16-boundary contract; verifier vs full FP64 is 3.36971e-7. The first differing observed common field is `inject.f32`. This is one CUDA operator/captured input, not a full-model oracle or a current HIP comparison. Existing target is explicitly CUDA-only. **HIP matched-operand replay, actual Q8 scale/signed-byte versus original-BF16 audit, same-operand BF16/Q8/FP32 device comparison, and state/variant/fence validation are NOT_RUN and the requested framework is not complete.** CPU FP64 oracle unit tests pass; they do not replace these device proofs.

## Bounded statistical diagnostic

Four newly authored non-sensitive local snippets, independent fresh process/reset each: English `The river flows beneath the old bridge.`, code `def square(x): return x * x`, math `Two plus three equals five.`, French `Bonjour, le ciel est bleu.` No downloads. Token IDs and absolute placement prefixes, including the extra target token which was not fed, are in local `corpus-{bf16,q8}-manifest.json`. `tools/strata_tokenizer.py` loaded the existing original GGUF metadata, and every string round-tripped. Corpus JSON SHA-256 `c099de872ede9b9c9248cf768f48077ea8fd5c14b7082ddf153d1a565514e119`; tokenizer source SHA-256 `6db1ea62d3edbaa64dce139088146ccf370933998d5f997cd5f41a39300075a6`; source index identity below. Same 25 target tokens, forward serial, default reference:

| Candidate request | Mean NLL ref / candidate | PPL ref / candidate | Mean / max KL | Top-1 agreement |
|---|---:|---:|---:|---:|
| bf16 | 2.823300 / 2.818412 | 16.8323 / 16.7502 | 0.004608 / 0.016322 | 24/25 |
| q8 | 2.823300 / 2.819769 | 16.8323 / 16.7730 | 0.013438 / 0.047771 | 23/25 |

These are descriptive bounded teacher-forced placement diagnostics from a tiny hand-authored corpus, not a representative held-out benchmark, long-context or reasoning/task evaluation; no statistical significance or production inference follows. No statistical evaluation of native bulk activation modes was done.

## Performance and memory

Fixed-window grouped-forward timing used the seven original token IDs, positive one-row commits, two warmup windows and five measured windows per process; three fresh shared processes per weight request. `elapsed_ns` is coordinator end-to-end execute/commit latency, not HC kernel time or startup. Samples in ms, ordered by process and position:

| Request | Raw samples (ms) | Median (range) |
|---|---|---|
| default | 22.633, 22.317, 22.655, 22.574, 22.243; 23.107, 22.207, 22.821, 22.381, 23.176; 21.919, 21.936, 21.837, 21.843, 21.850 | 22.317 (21.837–23.176) |
| bf16 | 23.401, 23.549, 22.662, 22.770, 22.789; 23.211, 23.631, 22.733, 22.732, 23.670; 23.062, 23.276, 22.981, 22.873, 22.973 | 22.981 (22.662–23.670) |
| q8 | 21.815, 21.809, 22.042, 21.854, 21.791; 22.233, 22.808, 22.814, 22.775, 22.133; 22.210, 22.092, 22.436, 22.207, 22.019 | 22.133 (21.791–22.814) |

Timing-path predictions matched corresponding serial captures, but fixed-window mode forbids capture, so full-logit observer equivalence is NOT_PROVEN. This short non-randomized test cannot establish a robust Pareto winner, sustained prompt throughput or generation throughput. For the six BF16-weight native cases, three separate `STRATA_PREFILL_TIMING=1` processes each produced capture bytes exactly equal to their corresponding timing-off wire. Raw per-line HC phase samples are in `logs/timing-phase-summary.txt`; per-stage output lacks owner identity and aggregate/chunk lines can overlap, so these are not summed or presented as owner HC latency. No optimized-FP32 performance claim. Timed wall time includes startup and captures and is not kernel latency.

Peak GPU memory per intervention is NOT_RUN. At serial verifier graph capture, observed CUDA free MiB was 5085 default/bf16 and 4285 q8; HIP values were 16242/16162 default and q8, versus 17042/16962 bf16. These are pre-execution snapshots, not peak-memory or isolated allocation deltas; Q8 loader logs separately report 0.62 GiB projection copies per worker. Expert slots remain full: CUDA 8,493,465,600 bytes; each HIP 12,740,198,400 bytes. Same configured 512 MiB reserve. BF16x2 low buffers and FP32 path memory impact remain unmeasured.

## Build, tests and evidence

Used existing vanilla toolchain/venv, host/CUDA/HIP caches and `strata-hc-replay` rebuilt with `-j28`. Live smoke SHA-256 `b1744f3c7b4be4e41d687cf5a810d646259d94c2751b62b968ebd165c91e0642`; CUDA worker `5f4ddabb5fce4b99d5b2961208aebeaca8da99024b6cade839639efb4189c379`; HIP worker `04f0c0e5441f410079065da0b5eeab769d6c2bfd546b367d314d6c0238e2f7c8`. Host CTest 99/99; new oracle/qualification tests 11/11; resident Python 12/12; pip check passed. Prior Python suite initially 147/148: synthetic trace stub lacked newly required `plan`; fixed only the fixture in `tools/test_heterogeneous_numerical_trace.py`, then 148/148 passed. No GPU inference source was modified. Asset identity is certified index `c344c01f59f547ed8a59003857a82eb90499e5299154241dbca787acb54b5b7c`; prior raw audit is in report-0027-9, not repeated here. No assets downloaded or repacked.

Restricted evidence `/home/chris/tmp/strata-heterogeneous/m7-0030/`: exact commands, stdin where applicable and actual environment in `commands/`, raw captures in `captures/`, `logs/` contains all processes, pair gates, traces, reports, timing and CPU tests. `qualification-manifest.json` derives from validated pair gates; qualification output SHA-256 `7f785fc3123fc9f0802901d7ff65bf88cc78f6c0be9a793825ad0e7827b17730` is a summary, not admission. `logs/integrity.log` SHA-256 `9b349c6a1e3af228c0d7ad9799adc6547ad9d6070226bd29902e7587a978e6c3`; CUDA FP64 replay analysis `15b502b396455ddc511c8f29bd1c12ab06452924d30bad5e52cbd9f4c1ac6e04`. Raw weights and tensors remain local. No GitHub Actions, reverse grouped implementation, bulk+MTP, default change or production qualification. Untracked `tools/heterogeneous_pool_probe.cpp` was preserved.
