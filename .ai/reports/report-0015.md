# Report 0015 — bounded CPU-mediated HIP receiver arm

**Status: PASS for bounded experimental cases; not production/throughput qualified.** Starting commit `2517c1744e1d3ffaf688f187092eccf8052a271e`. The opt-in `--receiver-arm` mode uses the existing CUDA→HIP bounded transport and a HIP-owned mapped input/ready word; no page is registered with both runtimes. The regular vanilla layer split remains CUDA0 [0,12) → HIP0 [12,30) → HIP1 [30,48), in two processes. Serial and HIP-only modes remain available; receiver arm and native bulk prefill remain off by default. No tensor parallelism, inference math, precision, memory limits, residency, clocks or power settings changed. No Actions or GPU fault injection.

## Builds and tests

Existing vanilla caches/toolchains: full `cmake --build build-hetero-smoke-{cuda,hip,host} -j28` passed. Host CTest **95/95 passed**, including receiver-publish, final-pipeline, HIP final adapter and host tests. Checkout `.venv`, inherited `PYTHONPATH` removed: prefill state/parity tests **18 passed**, resident Python tests **12 passed**, `pip check` clean. Added opt-in trace evidence only: a coordinator `receiver_arm_ack` performance span and vanilla verifier `PRELAUNCH-CONSUMED` event. Rebuilt actual CUDA worker/coordinator, HIP worker and host with `-j28`; targeted CTests **4/4 passed**. Raw build/test logs and all following artifacts: `/home/chris/tmp/strata-heterogeneous/m7-0015/`.

## Fixed-token full-logit parity

Original Q2_0 full-resident pack and stage configs, fingerprint 2026111301, context/max_cells 2048, max_t 8; CUDA ordinal 0 PCI 04:00.0, HIP ordinals 0/1 PCI 03:00.0/07:00.0. Exact commands and tokens are recorded in `run-matrix.py` and each capture log. Three independent modes—grouped serial, HIP-local overlap, and receiver-arm with HIP-local overlap—used identical teacher-forced tokens, same plan and unchanged logits gate. Both ordinary sockets and shared CPU memfd transport were tested; neither passes device pointers across runtimes.

| Shape / transport | serial vs HIP overlap | serial vs receiver arm |
|---|---|---|
| 3 × 1 row, keep 1 / shared | PASS, 3/3 rows | PASS, 3/3 rows |
| 3 × 1 row, keep 1 / socket | PASS, 3/3 rows | PASS, 3/3 rows |
| 3 × 4 rows, keep 2 / shared | PASS, 12/12 rows | PASS, 12/12 rows |
| 3 × 4 rows, keep 2 / socket | PASS, 12/12 rows | PASS, 12/12 rows |

For all eight comparisons: exact predictions and top-1, KL 0, max absolute logit difference 0. All evaluated rows were compared, including rejected suffixes and later accepted-prefix positions. The existing STFR comparator used `read_run(..., allow_partial=True)` and same-plan `compare(..., prefill=True)`; this comparator option does **not** mean native bulk prefill was used. Individual captures and `*-compare.json` give full per-row evidence. A subsequent final-build traced multi-row receiver-arm run also passed all 12 logits rows (`final-trace-compare.json`).

## Arm, ordering, commit and reset

`final-trace-multi-shared-arm.log` enables vanilla verify tracing plus heterogeneous overlap, performance and window traces. For each window: native `EXTERNAL-ARMED` is emitted only after HIP's stream query reports `cudaErrorNotReady`; the `receiver_arm_ack` span completes after validating the full-identity ACK and before stage-0 spec/activation send; the validated transported payload is published afterward (`receiver_publish`). The trace shows `PRELAUNCH-CONSUMED` at HIP layers 12 and 30, downstream `WINDOW-PRE`, and HIP-local completed handoff flag 1. All-stage committed windows advance positions 0→2→4→6 after keep 2. For example window 2 ACK ends at 31677815828925 ns, stage-0 spec begins at 31677815830528 ns, and payload publication begins at 31677820044439 ns. Trace timestamps and publication order demonstrate protocol sequencing, not productive GPU concurrency. `one-arm.log` independently captures initial one-row arm and bounded completion.

`resident-arm-reset.log`, using the corrected `run-reset.py` progress/continue driver with bounded per-message waits, shows two identical fixed-prompt requests in one session, each returning token 96220, then `HETERO_CLOSED` and exit 0. This is normal reset/drain and orderly cleanup, not a fault-injection test. No stuck worker remained. The dual-registered page probe from report 0014 was not repeated.

## Bounded timing and remaining overhead

Tracing-off, one end-to-end wall sample each (seconds; includes process start, full model loading and shutdown):

| Workload | serial | HIP-only overlap | receiver arm |
|---|---:|---:|---:|
| one/shared | 8.267 | 8.210 | 8.252 |
| one/socket | 8.237 | 8.144 | 8.281 |
| multi/shared | 8.214 | 8.313 | 8.262 |
| multi/socket | 8.272 | 8.275 | 8.230 |

These startup-dominated single samples (`wall-*.json`) do not measure steady-state performance or establish a speedup. In the separately traced four-row run, windows 2–3 show arm ACK 53/58 µs, CUDA stage round trips 3.894/4.104 ms, producer download fence waits 108/108 µs, receiver publication 5/14 µs, and commit barriers 1.453/1.441 ms. These are instrumentation-affected spans, not a critical-path attribution or general benchmark. The producer still drains/fences its activation, the coordinator receives and validates it, transports it to HIP and publishes a local ready word; ACK, copying, socket/memfd transfer and ordered all-stage commits remain. A pending consumer graph alone is not proof of useful overlapping execution. Do not claim full three-GPU overlap or target throughput.

## Blockers

Native bulk prefill remains independently unqualified: report 0013 found committed CUDA state divergence before the vendor bridge; this iteration did not change that path or diagnose its root cause. Matched MTP runtime assets and their compatibility were not established in approved inputs; no unapproved directory was inspected and no assets fetched or substituted. Native MTP lifecycle/scheduler integration remains separate and unverified. The operator's 3840 prompt tok/s and prior 160 accepted output tok/s target remain unmatched and workload dependent. The pre-existing untracked `tools/heterogeneous_pool_probe.cpp` was left untouched.
