| Config | Aggregate decode tok/s | Aggregate output tok/s | Load wall s |
| --- | --- | --- | --- |
| parallel 1 | **116.7** (116.7 / 124.3 / 115.5) | **77.2** (77.2 / 80.0 / 76.3) | **38.9** (38.9 / 37.5 / 39.3) |
| parallel 2, default groups (2 pipelined groups of 1) | **105.5** (106.2 / 105.3 / 105.5) | **84.9** (85.7 / 84.9 / 84.7) | **35.3** (35.0 / 35.3 / 35.4) |
| parallel 2, --batch-groups 1 | **69.7** (70.5 / 69.7 / 69.5) | **69.1** (69.8 / 69.1 / 69.0) | **43.4** (43.0 / 43.4 / 43.5) |
| parallel 2, --batch-groups 1 --batch-mtp | **79.8** (86.0 / 79.8 / 79.2) | **79.0** (79.6 / 79.0 / 78.5) | **38.0** (37.7 / 38.0 / 38.2) |

| Config | Prompt | Decode tok/s per request, client (min to max) | Server decode tok/s | First token ms (min to max) | Requests |
| --- | --- | --- | --- | --- | --- |
| parallel 1 | code | 116.0 (108.0 to 126.0) | 115.7 | 3186 (385 to 3890) | 18 |
| parallel 1 | long | 121.4 (114.4 to 130.7) | 120.8 | 5224 (4541 to 5754) | 18 |
| parallel 2, default groups (2 pipelined groups of 1) | code | 68.3 (65.2 to 71.3) | 69.0 | 564 (281 to 878) | 18 |
| parallel 2, default groups (2 pipelined groups of 1) | long | 54.6 (40.5 to 68.6) | 55.5 | 2995 (1868 to 4148) | 18 |
| parallel 2, --batch-groups 1 | code | 43.5 (36.8 to 49.0) | 43.8 | 299 (293 to 777) | 18 |
| parallel 2, --batch-groups 1 | long | 39.4 (33.1 to 50.2) | 39.9 | 1883 (1847 to 2171) | 18 |
| parallel 2, --batch-groups 1 --batch-mtp | code | 56.9 (40.5 to 64.4) | 57.5 | 308 (300 to 753) | 18 |
| parallel 2, --batch-groups 1 --batch-mtp | long | 46.4 (35.0 to 65.4) | 47.1 | 1896 (1875 to 4141) | 18 |
