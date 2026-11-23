| run | requests | generated | decode s | engine decode tok/s | draft acceptance % | ttft median s | overall tok/s | thinking tok/s | answering tok/s | tool calls tok/s | phases from | prompt mismatches |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| replay | 106 | 91470 | 650.139 | 140.69 | 77.12 | 3.543 | 140.19 | 119.38 | 106.49 | 154.19 | /metrics | 0 |
| replay | 106 | 85358 | 616.727 | 138.4 | 76.86 | 3.591 | 137.56 | 115.82 | 106.49 | 154.66 | /metrics | 0 |
| replay | 106 | 86618 | 642.814 | 134.75 | 75.35 | 3.514 | 133.75 | 108.42 | 106.49 | 147.78 | /metrics | 0 |
| **median** | | | | 138.4 | 76.86 | 3.54 | 137.56 | 115.82 | 106.49 | 154.19 | | |

Phase columns from /metrics: phase_tps.py (the Human User's benchmark). From chunks: stream chunks counted as tokens. reasoning = thinking, content = answering, tool_calls = writing tool calls.
