# 15/18/15 placement and fully resident 256K Q8 KV planning

CPU-only tool: from repository root, run

    .venv/bin/python tools/heterogeneous_vram.py --context 262144 --kv q8 --split 15 18 15
    .venv/bin/python tools/heterogeneous_vram.py --context 262144 --kv q8 --sweep
    .venv/bin/python tools/heterogeneous_vram.py --context 262144 --kv q8 --split 15 18 15 --json

The committed calculator reproduces the KV payload formula in include/strata/kernels/kv_q8.hpp, layer geometry in include/strata/core/geometry.hpp and stage-local state in src/core/session.cpp and src/core/layer.cpp. QSA is every fourth layer; Q8 K/V uses 1056 bytes per allocated QSA layer per cell including FP16 scales. At 262144 tokens: 264 MiB per QSA layer, 3.09375 GiB over all 12. Also accounts for pooled FP32 indexer + tail, page table, shared RoPE and fixed GDN recurrence. Payload/subset only: excludes small alignment, staging, scratch and other state allocations. Stages with no QSA still allocate one primary state.

Estimate for plan cuda:0:0-15,hip:0:15-33,hip:1:33-48:

| Owner | Layers | QSA/GDN | Full expert slots | Q8 KV MiB | Estimated used GiB | Estimated free GiB |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| CUDA0 | 15 | 3/12 | 7680 | 792 | 14.50 | 1.50 |
| HIP0 | 18 | 5/13 | 9216 | 1320 | 18.03 | 13.97 |
| HIP1 | 15 | 4/11 | 7680 | 1056 | 15.83 | 16.17 |

Total estimate anchors on report 0032's 32K sampled native-prefill maxima (11817 / 17135 / 17280 MiB, split 12/18/18). Assumes baseline FP16 KV (source default, not observed format). Replace with --baseline-kv q8 if logs contradict. Delta uses 675 MiB Q2_0 expert payload per layer, state delta, adjustable 64 MiB incremental dense allowance per added layer (no credit for removed dense weights), and 64 MiB extra workspace per owner for >32K. These allowances are assumptions, not measured bounds; they can underestimate peak. Default desired remaining VRAM is 1024 MiB. With these assumptions 16 CUDA layers estimate 15.51 GiB used, 0.49 free. Not a fit guarantee or recommendation to reduce safety reserves.

Baseline hardware experiment first: compare 12/18/18 versus 15/18/15 at 32768 configured context with identical current format/options. Preserve all precision defaults and full residency. Actual Q8 and 256K runtime support require explicit opt-in implementation and staged qualification; do not label a 32K test as a 256K Q8 run. See .ai/handoffs/handoff-0034-placement.md.
