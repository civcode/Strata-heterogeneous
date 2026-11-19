# Report 0082

**Status: Partial.** All 12 blocks completed, with six measured requests per pack, prompt and mode. The outputs differ between packs, and the requested accepted drafts per window is not exposed in the untraced server results. These limitations prevent attributing the measured differences solely to the pack's execution speed or making a definitive production switch decision.

## Results

A = canonical production pack; B = native. Each row has the six measured requests in block order (two per block). Decode and prefill are server tok/s; `median [min–max]; samples`. First request per prompt in each block was warm-up and excluded. All requests had zero cached prompt tokens. Drafts are accepted/offered for measured requests, also in block order; the untraced API does not report the number of MTP verification windows, so accepted **per window** cannot be computed reliably from these counters. Warm-ups and full outputs are retained in the result JSON.

### Mixed

| Prompt | Pack | Decode | Prefill median [min–max] | Accepted/offered |
| --- | --- | --- | --- | --- |
| code | A | 120.6 [118.6–129.6]; 129.6, 129.2, 121.2, 120.0, 119.9, 118.6 | 364.3 [362.9–367.1] | 145/201, 148/214, 145/201, 148/214, 145/201, 148/214 |
| code | B | 110.3 [109.0–111.0]; 111.0, 109.0, 110.9, 109.6, 111.0, 109.1 | 508.7 [507.2–510.3] | 139/210, 139/215, 139/210, 139/215, 139/210, 139/215 |
| long | A | 136.5 [126.3–146.1]; 146.1, 136.4, 136.1, 126.3, 146.0, 136.5 | 2452.6 [2441.2–2470.7] | 117/148, 108/147, 117/148, 108/147, 117/148, 108/147 |
| long | B | 126.4 [115.3–140.0]; 124.3, 140.0, 115.3, 128.4, 116.0, 129.8 | 2852.5 [2841.0–2862.8] | 109/171, 114/141, 109/171, 114/141, 109/171, 114/141 |
| 16K | A | 159.4 [155.9–162.6]; 162.6, 156.2, 162.4, 156.5, 162.3, 155.9 | 3951.1 [3905.0–3979.8] | 79/97, 79/100, 79/97, 79/100, 79/97, 79/100 |
| 16K | B | 141.6 [136.7–157.1]; 157.1, 150.3, 141.7, 136.7, 141.4, 136.9 | 4069.1 [4039.9–4083.2] | 79/96, 79/103, 79/96, 79/103, 79/96, 79/103 |

**Mixed:** native median decode is −8.6%, −7.4%, −11.2% for code, long, 16K. Code ranges do not overlap; long and 16K do. Different generated text and acceptance patterns confound a pure engine-speed comparison.

### HIP-only

| Prompt | Pack | Decode | Prefill median [min–max] | Accepted/offered |
| --- | --- | --- | --- | --- |
| code | A | 111.5 [109.1–113.5]; 113.4, 109.6, 113.5, 109.3, 113.3, 109.1 | 473.6 [472.2–477.1] | 148/194, 142/198, 148/194, 142/198, 148/194, 142/198 |
| code | B | 108.6 [106.5–110.7]; 110.6, 106.6, 110.7, 106.5, 110.6, 106.5 | 523.0 [521.9–525.6] | 154/202, 149/201, 154/202, 149/201, 154/202, 149/201 |
| long | A | 117.7 [113.5–122.0]; 121.9, 114.0, 122.0, 114.0, 121.4, 113.5 | 2733.6 [2719.0–2739.5] | 114/153, 112/170, 114/153, 112/170, 114/153, 112/170 |
| long | B | 109.0 [103.3–114.6]; 114.6, 103.3, 114.6, 103.4, 114.6, 103.4 | 3272.4 [3257.6–3281.5] | 116/157, 106/173, 116/157, 106/173, 116/157, 106/173 |
| 16K | A | 144.4 [142.1–146.1]; 142.8, 146.1, 142.1, 145.9, 142.5, 145.9 | 3796.0 [3782.4–3801.2] | 75/93, 79/101, 75/93, 79/101, 75/93, 79/101 |
| 16K | B | 143.5 [128.3–150.1]; 150.1, 143.5, 132.0, 128.3, 149.9, 143.4 | 4312.9 [4304.7–4317.9] | 82/95, 80/97, 82/95, 80/97, 82/95, 80/97 |

**HIP-only:** native median decode is −2.6%, −7.4%, −0.6%; all ranges overlap. The same content/acceptance confounding applies.

## Method and interpretation

Order in each mode: A B B A A B; fresh server for every block, untraced, greedy, 300/200/128 output tokens. Each prompt ran warm-up then twice measured. The three prefixes within each block differ to prevent prompt-cache reuse, but corresponding requests use identical inputs across arms and blocks. Hence these are three independent server blocks with only **two distinct measured inputs**, not six independent prompts; tight spreads and range separation should not be treated as statistical significance or generality across workloads. Prefill here is warm request prefill and is not directly comparable to the previous reports' first-request figures.

**Greedy outputs were not identical.** First differing zero-based character positions below compare the first block of each arm, by output field; `=` means identical, including empty. There was no within-arm output variation across the three blocks. Character positions are not token positions; the API does not return token IDs. Reasoning and content are reported separately, not concatenated.

| Mode | Prompt | Warm-up reasoning / content | Measured 1 reasoning / content | Measured 2 reasoning / content |
| --- | --- | --- | --- | --- |
| mixed | code | = / 199 | = / 65 | = / 321 |
| mixed | long | 182 / 0 | 309 / 0 | 404 / = |
| mixed | 16K | 201 / = | = / = | = / = |
| HIP-only | code | = / 335 | = / 319 | 19 / 432 |
| HIP-only | long | = / = | = / = | 209 / 200 |
| HIP-only | 16K | 201 / = | 201 / 0 | 201 / 0 |

The second GGUF shard, first shard, and (for A) `experts.bin` were read in that order on every switch; 60 GB RAM cannot keep all simultaneously resident. A0 was warmed similarly before the first start. Time from launch to loaded health in seconds, including non-switch blocks: mixed A0/B1/B2/A3/A4/B5 = 9.0/11.0/10.0/9.5/8.0/11.0; HIP-only = 9.5/12.0/10.5/10.0/8.5/12.0.

## Safety and validation

- Boot ID before and after: `5242f948-6a93-452b-b79e-a43f2d1afa57`. No pause or reboot. The initial shell's `cmp` against the procfs boot ID falsely reported a change because of procfs file size; direct reads confirmed identical IDs before any further GPU run.
- Kernel safety-filter log: no new matching lines; no GPU fault lines reported by any session. Core dumps since boot: none before or after.
- GPU wall accounting: mixed 269.3 s, HIP-only 278.1 s, total **547.4 s** (9.1 min); each block 43.2–46.9 s, sequential; all servers exited cooperatively with rc 0. No forced GPU process termination.
- `check_launch.py` accepted all four configs and existing builds. `python -m py_compile hetero/tools/serve_session.py` passed. No engine sources changed. The first block succeeded and wrote evidence, but the driver's summary printer crashed afterward on the new list-shaped result; fixed before the remaining blocks. The first block's JSON reports rc 0 and no errors/faults.
- Evidence: `.ai/runs/0082-{mixed,hiponly}/` result JSON, filled configs, GPU totals; `.ai/runs/0082-safety/` before/after snapshots and computed summary. Server logs are ignored and not used as evidence for claims here.

## Open

A production switch remains a judgment call: native prefill is consistently higher in these warm measurements, while decode depends on mode and prompt and generated text can change. For a causal decode comparison, use controlled identical token trajectories or additional representative prompts and expose actual verification-window counts without comparing traced to untraced runs.
