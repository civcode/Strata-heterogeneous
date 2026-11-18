# Report 0076 — production mirrors

**Completed**, with the INFO-line and vision limits below. Starting revision `f7fab0a8d`. Both `check_launch.py` checks accepted the existing 0071 HIP/CUDA and vision builds; no rebuild. Both mirrors served at context 262144 without fallback, via `serve_mixed.py` on local test port 8096. Evidence, configs, request results, scripts and logs: `.ai/runs/0076-production/`. No engine changes.

## Bring-up

| Mode | Idle VRAM: NVIDIA / R9700 0 / R9700 1 | After 16K VRAM | Available RAM idle / after 16K |
|---|---|---|---|
| HIP-only | 15 MiB / 25832 MB / 26074 MB | 15 MiB / 25848 MB / 26101 MB | 51 / 51 GiB |
| Mixed auto | 14253 MiB / 19177 MB / 20566 MB | 14347 MiB / 19186 MB / 20589 MB | 50 / 48 GiB |

NVIDIA is `nvidia-smi` MiB; AMD is `amd-smi metric -m` MB, including other device usage. RAM is `free -h` available, including reclaimable page cache. Mixed idle was after load and before requests; the other snapshots follow the 16K request. HIP-only had one short request before its 16K request.

Mixed start: CUDA front layers 0–14; HIP host auto K=31, layers 15–30 and 31–47. The captured front INFO reports `context=262144 kv=int8 kv_resident=0 expert_slots=7680 ... cvec=project:4-44 ... conversation_cache_slots=4 ... conversation_cache_min_tokens=8192 ... engine=0.1.61`. The server consumes the host's INFO protocol line rather than writing it to the engine log; a separate raw host INFO line was not captured. The host startup command and settings, plus the front INFO, are in `mixed-features/server-mixed.log`. Both processes started and shut down cleanly. The host log explicitly says its 15–30/31–47 handoff is `overlapped: STRATA_SPLIT_OVERLAP=1`; HIP-only says overlap is disabled beside `--pipeline-windows 2`, which overlaps stages itself.

Mixed cache test: first turn 16093 prompt tokens; after an intervening unrelated request the log parked 16156 tokens and restored 16088 tokens from a checkpoint. The second turn reported `cache_n=16088`. A request with `experimental_speed_projection=false` (mapped by the server to `cvec=0`) succeeded, as did a PNG vision request (318 prompt tokens). These were API/transport checks, not a semantic vision-quality test; short generation limits left the vision response in reasoning. See `mixed-features-result.json` and server log.

## Untraced production comparison

Three samples each, alternating HIP-only/mixed/fixed, one fresh server per sample. Same prompt within each paired sample; sample number prepended to avoid cache hits. `code`, `long`, `long16k` use the report 0075 token prompts (118, 5000, 16102 served prompt tokens respectively); decode limits 300, 200, 128. This prefix and the server chat template mean they are not byte-identical to earlier report 0075 requests. No prompt tokens were cached. Entries are median (min–max) tok/s; drafts are accepted/proposed in sample order. Small samples and different generated token sequences make decode comparisons noisy.

| Mode | Prompt | Prefill | Decode | Accepted/proposed |
|---|---|---:|---:|---|
| HIP-only | code | 323.7 (321.9–324.5) | 107.9 (107.4–113.7) | 150/228, 147/219, 161/229 |
| Mixed auto | code | 265.2 (265.2–265.8) | 117.8 (117.7–120.6) | 161/224, 156/213, 155/209 |
| HIP-only | long | 2350.5 (2343.6–2363.4) | 122.7 (114.4–123.4) | 111/177, 110/175, 109/143 |
| Mixed auto | long | 2287.3 (2282.8–2300.0) | 118.7 (116.0–119.7) | 112/174, 114/172, 112/150 |
| HIP-only | 16K | 3478.1 (3462.1–3483.8) | 132.2 (125.3–146.1) | 77/98, 72/100, 78/94 |
| Mixed auto | 16K | 3387.9 (3384.0–3394.3) | 131.2 (120.1–138.5) | 85/112, 78/127, 80/103 |

## Mixed auto vs fixed 33

Same alternating samples, untraced; auto rows above are reused. Fixed 33 results:

| Prompt | Prefill median (min–max) | Decode median (min–max) | Accepted/proposed |
|---|---:|---:|---|
| code | 267.1 (264.8–267.1) | 117.8 (117.8–120.1) | 161/224, 156/213, 155/209 |
| long | 2292.8 (2280.5–2294.7) | 118.6 (115.1–119.2) | 112/174, 114/172, 112/150 |

No meaningful difference established on these short samples. Fixed 16K was also run: 3308.9 (3307.2–3314.5) prefill, 133.6 (120.0–136.5) decode; accepted 85/112, 78/127, 81/100.

## Safety and accounting

Before/after kernel journal and coredump lists are captured in the run directory. No new kernel lines and no new core dumps between snapshots; no reboot or GPU fault observed. Cooperative SIGINT and drain on every run; exit status 0. Wall-clock server lifetime conservatively charged as GPU time: approximately **510 seconds**, comprising initial HIP 34.8 s, mixed 16.1 s, HIP 16.9 s, feature run 18.4 s, first nine comparison runs 203.2 s, corrected nine comparison runs 202.4 s, final feature run 18.5 s. The first comparison batch used different mode prefixes and was discarded; its output files were overwritten by the corrected batch. Every run was under two minutes and all were sequential. Long-duration qualification was not run.
