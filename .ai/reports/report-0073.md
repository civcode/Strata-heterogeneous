# Report 0073

**Partial.** Rebuilds and CPU checks passed; the HIP-only A/B, mixed serving samples and traced requests ran. The requested front/link/stage breakdown and overlap bound could not be derived from the existing traces. **Tier escalation recommended:** the front call is asynchronous and timing it alone does not measure the time until the HIP stage receives the rows; measuring that boundary and the front GPU completion separately needs additional instrumentation and validation beyond the named logging locations. No engine changes were made.

## Builds and checks

HIP and CUDA `build-0071-{hip,cuda}` rebuilt with mixed split enabled at `f8e9b6590e182dc473b2fc865943e8212e5caef4` and stamped at that revision. Both builds passed `mixed_split_tests` and `mixed_split_cache_drop_test`; `check_off_is_vanilla.py` passed 6/6. The HIP and CPU vision helpers were also rebuilt and stamped after the launch checker rejected their old stamps. The launch checker passed for the server runs.

## Untraced measurements

Decode tok/s: median (minimum–maximum), three samples. Each server process handled the short request followed by the long request. HIP-only uses the committed vision-on config with its control vector; mixed uses the committed mixed config (no vision or control vector), so the mixed column is **not** a controlled backend comparison. Both use the same text prompts; the server chat template produces 83 and 4,963 prompt tokens. The long body is the same 4,900-token decoded document as the earlier 4,924-token direct-engine prompt, but server templating changes its length. Short responses were 300 tokens, long responses 200.

| Prompt | HIP pipeline 0 | HIP pipeline 2 | Mixed baseline |
| --- | ---: | ---: | ---: |
| Coding | 92.6 (92.1–94.7) | 97.9 (92.8–98.2) | 102.1 (101.6–102.2) |
| Long | 101.2 (101.2–101.2) | 105.3 (104.5–105.5) | 108.8 (108.5–108.9) |

HIP medians improve 5.7% and 4.1% respectively, with only three samples. Accepted/tested draft counts: coding HIP 0 = 133/190 each; HIP 2 = 127/187, 127/189, 127/187; mixed = 132/181 each. Long HIP 0 = 107/169 each; HIP 2 = 102/178 each; mixed = 110/180 each. The arm order was **0, 2, 2, 0, 0, 2**, not the strictly alternating order requested; this is a limitation of the result. In a separate run without the option, the log reports `--pipeline-windows 2 is on by default for a layer split of 2 stages`.

Additionally, three direct mixed runs used the old 23-token short prompt (only 75 generated tokens) and 4,924-token long prompt: median decode 114.8 (112.1–115.1) and 114.5 (114.4–114.6), drafts 42/66 and 117/186 respectively. These are not the same short prompt as the server A/B.

## Traced mixed requests

One server run with both timing variables set handled the same coding and long text prompts. The following are traced-only decode timing values, not compared to untraced throughput:

| Prompt | Decode ms/window | Verify ms/window | Draft ms/window | Commit/emit ms/window |
| --- | ---: | ---: | ---: | ---: |
| Coding | 17.52 | 15.18 | 1.61 | 0.73 |
| Long | 20.42 | 17.57 | 2.14 | 0.71 |

The stage traces show zero GPU wait, pool/plan and staging times, and report cumulative commit figures only (stage 0 about 0.54 ms and stage 1 about 0.16 ms); they do not expose actual HIP stage elapsed times. No front-stage per-window GPU/host timing is printed. `RemoteFrontStage::run` returns when the front has taken the work, not when its residual rows arrive, so its duration is not the requested host wait. **Front-plus-link share and longest stage: unknown; no reliable overlap bound can be given.** The separate direct mixed traced run also exists in the evidence, but uses the shorter prompt.

## Safety and evidence

No new core, reboot, Xid, gfxhub, GCVM or GPU-fault line. Seven new `Cannot map memory` lines each name `0x7e9152` pages, matching the previously observed shared-arena warning; kernel also logged AMD workqueue and delayed-fput hog warnings. No GPU process was forcibly terminated. Conservative GPU-time accounting includes entire process lifetimes: direct mixed 62.1, 45.9, 43.5, 42.7 s; six HIP servers 29.7, 29.6, 29.2, 29.7, 29.2, 29.6 s; mixed servers 32.8, 38.8, 35.8 s; default HIP server 29.3 s; traced mixed server 34.8 s: approximately **543 s / 9.1 min** total. All runs were sequential and individually under two minutes.

Evidence: `.ai/runs/0073-measure/` (configs, scripts, responses, server logs, trace, builds and kernel/core snapshots). Pre-existing untracked `front-0066-1.err` and `.out` were not modified. Open: accurate stage/link instrumentation, strictly alternating A/B if required, and an apples-to-apples mixed/HIP configuration comparison.
