# Oberon 256K capacity runs (2026-11-15)

Both topologies completed the staged synthetic near-capacity 256K capacity and resident lifecycle qualification with full expert residency and the 1 GiB per-device reserve. Handoff 0044 additionally verified repeated near-capacity requests, cancellation at committed checkpoints, endpoint reset/commit traces and rejection by two older worker binaries; see [.ai/reports/report-0044.md](../../../.ai/reports/report-0044.md). This is bounded native resident-adapter qualification, not general server or production sign-off. Existing numerical parity failures are not resolved or waived.

## Setup and CPU checks

Source revision `2b1016bf7d0a7d781966ba28a6f6708cb8825312`, Oberon, separate Release host/CUDA/HIP builds, `--parallel 28`. CUDA sm_89; HIP gfx1201, ROCm `/opt/rocm/core-10.1`, `STRATA_PREFILL_MMQ=ON`, `HSA_ENABLE_SDMA=0`. Checkout `.venv/bin/python` 3.12.3, `pip check` passed; CMake used this interpreter. No GitHub Actions. Host CTest: 102/102 passed. Heterogeneous Python unittest discovery: 378 passed (one expected child-process assertion appeared on stderr); long-context config tests: 2 passed. `pytest` was not installed; the existing unittest runner was used instead.

Binary SHA-256: coordinator `75fea7a7b41a6cddc5604706207a452c42e0d0dcf7bb79842af06f614418c563`; HIP worker `957e51822d090c4e2a790be35a7ff6b60c617c329b72434015b55e0c61df2f24`; CUDA worker `9f6824e339de2d177ad41c77d63188174f93376c17b76d1cf27317869a226cbe`.

Q2_0 pack index SHA-256 `c344c01f59f547ed8a59003857a82eb90499e5299154241dbca787acb54b5b7c`, fingerprint `2026111301`, max_t=8. Generated private configs retained 512 experts per owned layer, uncapped arena and 1 GiB reserve. HIP plan: `hip:0:0-24,hip:1:24-48`; mixed: `cuda:0:0-15,hip:0:15-33,hip:1:33-48`. Runtime identified NVIDIA 04:00.0 and R9700s 03:00.0/07:00.0. Actual loaded residency was full, with no reported host expert misses. Q8 resident KV, grouped HIP, native prefill chunk 128, shared transport. Each tier had CPU preflight and a separate 7-prompt-token/4-output-token short run before its near-capacity run.

At 2K, ordinary Q8 and explicit F16 completed one prompt token and four output tokens on each topology. Explicit long-context completed two independent requests per topology (7 and 9 prompt tokens, four outputs each), with normal close. This demonstrates reuse, not a full reset-position trace or cancellation test.

## Near-capacity observations

Each prompt was repeated ` hello` text; counts were measured by the actual tokenizer. Four output tokens were generated (three decode feedback tokens); native prefill committed all but the final prompt token, then the verifier committed the remaining prompt and decode windows. All logged native capacity checks fit, and workers closed normally. Free values are **minimum native phase-sampled free MiB**, including all logged load/prefill/early-commit phases; independent peaks are **maximum used MiB** sampled every two seconds with `amd-smi` and `nvidia-smi`. Sampling may miss transient peaks; tools have different accounting. They do not establish a lower bound on instantaneous free memory.

| Topology | Context | Prompt / output | Min native free MiB (plan order) | Independent peak used MiB (plan order) |
|---|---:|---:|---|---|
| HIP | 32768 | 32760 / 4 | 11026, 10866 | 21706, 21870 |
| HIP | 65536 | 65528 / 4 | 10786, 10628 | 21946, 22108 |
| HIP | 131072 | 131064 / 4 | 10308, 10150 | 22424, 22586 |
| HIP | 262144 | 262136 / 4 | 9354, 9196 | 23378, 23540 |
| mixed | 32768 | 32760 / 4 | 2335, 15674, 17974 | 13609, 17061, 14761 |
| mixed | 65536 | 65528 / 4 | 2207, 15474, 17810 | 13737, 17261, 14925 |
| mixed | 131072 | 131064 / 4 | 1953, 15070, 17480 | 13991, 17665, 15255 |
| mixed | 262144 | 262136 / 4 | 1443, 14266, 16822 | 14501, 18469, 15913 |

The tightest observed native free memory was mixed CUDA at 256K: 1443 MiB, approximately 419 MiB above the configured reserve. This is not a guarantee for other workloads or transient peaks. Long-context verifier and prefill allocator-boundary admissions reported `fits=1`; no reserve was lowered. Raw configs, preflights, phase records, run outputs and independently sampled monitor logs are private under `/home/chris/tmp/strata-heterogeneous/qualification-256k/`.

CPU preflight rejected 32760 prompt tokens plus 16 requested output tokens at 32768, and rejected a 65536 request against 32768 worker capacity (both exit 2). Handoff 0044 separately rejected 262136 prompt tokens plus nine outputs at 262144 and a mismatched worker capacity. The two 256K lifecycle sessions and their limits are documented in the report above. No claims are made about general prompt quality, same-format numerical parity, arbitrary content, throughput, HTTP server lifecycle, fault recovery, or production readiness.
