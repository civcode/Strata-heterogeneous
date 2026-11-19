### Totals

|  | 0090-mixed | 0090-hip |
|---|---|---|
| requests | 251 | 360 |
| prompt_tokens | 26275401 | 39967665 |
| cached_tokens | 23910742 | 36166493 |
| completion_tokens | 17562 | 26103 |
| prunes | 31 | 43 |
| pruned_outputs | 194 | 273 |
| compactions | 0 | 1 |
| cut_outputs | 0 | 1 |
| errors | 1 | 0 |
| probes | 2 | 3 |
| wall time (min) | 15.4 | 23.5 |
| new tokens read | 2364659 | 3801172 |

Request sequences identical: **no** (first difference at request 252)

### Tool and answer requests by context (medians)

| context | 0090-mixed n | 0090-mixed decode tok/s | 0090-mixed prefill tok/s | 0090-mixed ttft s | 0090-hip n | 0090-hip decode tok/s | 0090-hip prefill tok/s | 0090-hip ttft s |
|---|---|---|---|---|---|---|---|---|
| 0K-32K | 9 | 134.5 | 2115.7 | 1.7 | 11 | 112.4 | 1707.3 | 1.3 |
| 32K-64K | 34 | 110.1 | 2330.1 | 2.0 | 50 | 101.0 | 2295.0 | 1.8 |
| 64K-96K | 62 | 109.0 | 2143.5 | 2.0 | 95 | 101.3 | 2235.3 | 1.8 |
| 96K-128K | 77 | 111.9 | 2052.5 | 1.8 | 85 | 104.5 | 2235.8 | 2.1 |
| 128K-160K | 55 | 112.6 | 1967.8 | 1.8 | 65 | 103.9 | 1886.7 | 1.5 |
| 160K-192K | 8 | 109.7 | 2020.8 | 2.5 | 34 | 95.9 | 2303.2 | 2.7 |
| 192K-224K | 4 | 98.0 | 1779.6 | 2.1 | 9 | 84.7 | 2070.3 | 2.2 |
| 224K-256K | 0 |  |  |  | 7 | 99.4 | 1343.6 | 1.0 |

### Compactions

| run | cycle | prompt | cached | ttft s | decode tok/s | generated | wall s |
|---|---|---|---|---|---|---|---|
| 0090-hip | 0 | 244628 | 244590 | 0.571 | 90.2 | 1365 | 15.691 |

### Probes (fresh-server replay)

| probe | 0090-mixed | 0090-hip |
|---|---|---|
| compacted | not replayed | not replayed |
| deep | near-tie | not replayed |
| pruned | near-tie | not replayed |

### Resources (after cycle 0 to the end)

| run | VRAM growth % per card | RAM available after cycle 0 (MiB) | at the end | lowest | swap max (MiB) |
|---|---|---|---|---|---|
| 0090-mixed | first cycle incomplete / no post-cycle sample |  | 41691 | 40582 | 3510 |
| 0090-hip | {"cud0": 0.0, "hip0": 0.0, "hip1": 0.0, "hip2": 0.0} | 47346 | 46999 | 46041 | 626 |

### Errors

- 0090-mixed #251: {'type': 'server_error', 'message': 'restoring a conversation checkpoint failed: mixed split: the front stage: no checkpoint at 9956 tokens is kept here'}
