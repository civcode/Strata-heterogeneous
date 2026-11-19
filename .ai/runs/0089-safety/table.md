### Totals

|  | 0089-mixed |
|---|---|
| requests | 58 |
| prompt_tokens | 3662975 |
| cached_tokens | 3106450 |
| completion_tokens | 3546 |
| prunes | 7 |
| pruned_outputs | 37 |
| compactions | 0 |
| cut_outputs | 0 |
| errors | 0 |
| probes | 1 |
| wall time (min) | 3.4 |
| new tokens read | 556525 |

### Tool and answer requests by context (medians)

| context | 0089-mixed n | 0089-mixed decode tok/s | 0089-mixed prefill tok/s | 0089-mixed ttft s |
|---|---|---|---|---|
| 0K-32K | 8 | 135.8 | 2109.5 | 1.7 |
| 32K-64K | 26 | 103.8 | 2352.9 | 2.0 |
| 64K-96K | 15 | 92.9 | 2256.3 | 2.6 |
| 96K-128K | 8 | 109.9 | 1959.7 | 1.8 |

### Compactions

| run | cycle | prompt | cached | ttft s | decode tok/s | generated | wall s |
|---|---|---|---|---|---|---|---|

### Probes (fresh-server replay)

| probe | 0089-mixed |
|---|---|
| pruned | not replayed |

### Resources (after cycle 0 to the end)

| run | VRAM growth % per card | RAM available after cycle 0 (MiB) | at the end | lowest | swap max (MiB) |
|---|---|---|---|---|---|
| 0089-mixed | {"cud0": 0.67, "hip0": 0.06, "hip1": 0.15, "hip2": 0.0} | 50127 | 41331 | 41331 | 30 |

### Errors

none
