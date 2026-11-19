# Report 0084

**Status: Partial.** Builds, targeted GPU checks, the rename checks and the controlled version comparison ran. The production-option audit below identifies the configured arguments and their intended meanings, but is **not a complete comparison of all relevant implicit defaults and implementation semantics across versions**; that part remains open. No engine changes were made.

## Builds and rename

- At `7d80966b1`, Release/Ninja, 28 jobs: all default targets built in new host, HIP gfx1201, CUDA sm_89, HIP vision and CPU vision trees. All five have `BUILD_COMMIT` stamps. HIP and CUDA used ROCm 10.1 and CUDA 13.0 respectively. The first vision HIP configure lacked `STRATA_VISION_HIP_ARCHS`; correcting it to `gfx1201` built successfully.
- **Important build difference:** 0.2.16's `engine/CMakeLists.txt:105` says llama.cpp MMQ prompt kernels were removed. CMake explicitly warns that `STRATA_MMQ` and `STRATA_PREFILL_MMQ` are unused, despite both being passed ON. 0.2.15 accepts both. Thus the requested flags cannot enable those kernels on 0.2.16; this is a real version difference, not a configure failure. The build uses Strata's prompt kernels and FP16 path.
- Host CTest 147/147; mixed split and cache drop tests 2/2 each on HIP and CUDA. Targeted vanilla GPU CTests: HIP `hip_prefill_gemm`, `hip_q2_zero`, `hip_handoff`, `hip_mapped_alias`, `hip_device_selftest`, `hip_prefill_fused_iq`, `hip_prefill_decode_golden` 7/7; CUDA `cuda_device_selftest` 1/1. No failures or skips in this selection; all other GPU tests not run. HIP test time 74.09 s, CUDA 0.50 s.
- New stamped mixed builds passed launch check and served the features suite: first 16,093-token conversation, interference, restored 16,088 cached tokens on second turn; vision and cvec-off requests returned; clean shutdown. Serial vs two-window pipeline with `STRATA_IQ_MT_MIN=1`: code 300/300 token IDs identical.

## Production mirror option inventory (0.2.16)

The engine parser and help are in `engine/src/program/generate.cpp`; server configuration is consumed under `terminal/serve/`; mixed options are guarded additions. The table records the stated setting and its 0.2.16 meaning, **not a proof that all underlying algorithms/defaults are unchanged**. Both mirrors were accepted in report 0083; the renamed mixed mirror was exercised here. Paths and executable roles were redirected in run-only configs; committed mirrors were not changed.

| Argument or variable | Value and meaning / qualification |
| --- | --- |
| `--pack` | Pack directory. |
| `--native` | Native GGUF model shard; enables native paths. |
| `--ple-gguf` | PLE table shard. |
| `--expert-profile` | Profile used to fill the VRAM expert tier. |
| `--expert-cache` | `auto`: size expert tier from available VRAM; actual capacity is hardware and allocation dependent. |
| `--mmap-experts` | Map expert data instead of resident arena. |
| `--prefill` | `auto`: automatically choose prompt chunking; algorithm/kernel choice can change. |
| `--spec` | Four speculative draft tokens. |
| `--spec-min-p` | 0.5 draft-confidence threshold. |
| `--mtp` | MTP draft-layer directory. |
| `--max-context` | Capacity 262144. |
| `--kv` | `int8` KV representation. |
| `--kv-resident` | `0`: no KV streaming resident-window limit. |
| `--vision` | Enable image embedding requests. |
| `--vram-reserve-mib` | Reserve 700 MiB. |
| `--control-vector-scaled` | Vector file and scale 1.0. |
| `--control-vector-layer-range` | Inclusive layers 4 through 44. |
| `--cvec-mode` | `project`: project out the chosen direction, rather than add it. |
| `--cvec-dir` | `per-layer`: use each layer's direction. |
| `--pipeline-windows` | 2, pipelined verifier windows. |
| `--prompt-cache` | Six prompt checkpoints. |
| `--prompt-cache-spread` | Spread checkpoints across the conversation. |
| `--conversation-cache-slots` | At most four parked conversations. |
| `--conversation-cache-min-tokens` | Park conversations of at least 8192 tokens. |
| `--front-exe` | Mixed only: CUDA front executable. |
| `--front-layers` | Mixed only: first 15 layers on front. |
| `layer_split` | Server passes `--layer-split auto`: automatically plan the HIP layer split. |
| `HSA_ENABLE_SDMA=0` | ROCm runtime setting disables SDMA; external runtime option, not a Strata parser option. |
| `STRATA_SPLIT_OVERLAP=1` | Requests overlapped split handoffs, **but disabled alongside `--pipeline-windows 2`**; pipeline has its own overlapping loop. Confirmed in `generate.cpp:13350` and startup log. |
| `STRATA_PIPELINE_AUTO=0` | Disable *automatic* pipeline selection; explicit `--pipeline-windows 2` still selects it. |
| `STRATA_MTP_STAGE0=0` | Keep MTP on the last stage rather than stage 0. |

The mirrors also configure server `gpu=[0,1]`, `backend=hip`, `parallel=1`, tokenizer and model name, vision GPU/CPU executables and model files, `max_tokens=1024`, and log/port/cwd/library paths. Their full cross-version server semantics and implicit defaults remain to audit. In particular, 0.2.16 removes MMQ and changes internal execution; explicit option spelling does not imply equivalent performance.

## 0.2.15 vs 0.2.16 HIP-only

Separate option-off HIP builds from archive `7d573bee6` and current `7d80966b1`; the former has `STRATA_MIXED_SPLIT=OFF`. Both used production engine arguments/environment and both MMQ flags; the latter flags are unused on 0.2.16 as noted above. Untraced direct engine protocol, not the server: three fresh-engine blocks per version in A B B A A B order. Each block warmed each prompt once then measured two distinct-prefix, uncached requests with the same token IDs across versions; long 4927 tokens / 200 generated, 16K 16027 / 128. Reported in tok/s; values are six samples in block order, median in bold. Direct-driver prefixes and omission of the server mean these are controlled engine results, not server end-to-end timings.

| Version / prompt | Prefill samples | Decode samples |
| --- | --- | --- |
| 0.2.15 long | 2783.9, 2779.7, 2747.4, 2739.7, 2740.4, 2741.2 (**2744.3**) | 106.7, 107.0, 106.4, 106.1, 106.5, 106.2 (**106.4**) |
| 0.2.16 long | 2207.7, 2204.9, 2204.7, 2199.7, 2192.3, 2189.3 (**2202.2**) | 117.4, 121.9, 108.4, 111.1, 117.2, 121.4 (**117.3**) |
| 0.2.15 16K | 3812.7, 3798.9, 3782.8, 3781.3, 3789.5, 3775.2 (**3786.2**) | 117.8, 119.8, 117.7, 119.3, 108.0, 110.1 (**117.7**) |
| 0.2.16 16K | 3219.2, 3217.8, 3211.4, 3207.8, 3194.2, 3190.8 (**3209.6**) | 116.4, 116.5, 105.5, 104.9, 116.0, 116.7 (**116.2**) |

**Yes:** 0.2.16 prefill is slower beyond the observed spread, by 19.8% long and 15.2% 16K by median. MMQ removal is a relevant documented difference, not an isolated proof of the cause; other source and default changes are not excluded. Outputs/draft counts need not be identical between versions, so decode is descriptive, not a same-output comparison.

## Safety and limits

Boot ID before/after `5242f948-6a93-452b-b79e-a43f2d1afa57`; no coredumps before or after. No new GPU fault lines. The filtered kernel log gained two `mmput_async_fn` workqueue CPU-hog notices (matched by `MES` inside the function name), not GPU faults. Accounted GPU-session wall time approximately 380.5 s (37.9 serving + 25.7 exactness + 242.3 comparison + 74.6 GPU tests), under 30 minutes; each run under two minutes. No GPU jobs overlapped. Evidence under `.ai/runs/0084-*`; no Actions run.

Remaining: complete the cross-version default/semantic audit, and investigate the prefill regression if desired. All-target builds and selected tests do not qualify every backend or full GPU suite.
