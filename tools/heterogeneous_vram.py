#!/usr/bin/env python3
"""CPU-only VRAM estimate for the 48-layer Qwen4Exp heterogeneous model.

Exact persistent KV payload arithmetic; other VRAM is a report-0032-based estimate.
No GPU access, allocation or config modification. All capacities use binary units.
"""
import argparse
import json

MIB = 1 << 20
GIB = 1 << 30
LAYERS = 48
EXPERT_BYTES_PER_LAYER = 675 * MIB
GDN_BYTES = (128 * 128 * 48 + 10240 * 3) * 4
BASE_CONTEXT = 32768
BASE_SPLIT = (12, 18, 18)
BASE_USED_MIB = (11817, 17135, 17280)  # report 0032, max sampled bulk prefill
VRAM_MIB = (16384, 32768, 32768)


def persistent(lo, hi, context, kv="q8"):
    if not (0 <= lo < hi <= LAYERS and 1 <= context <= 1_000_000):
        raise ValueError("invalid layer range or context")
    if kv not in ("q8", "f16"):
        raise ValueError("kv must be q8 or f16")
    qsa = hi // 4 - lo // 4
    gdn = hi - lo - qsa
    # One primary QSA state even for a stage with no QSA layers (session.cpp).
    allocated = max(1, qsa)
    pages = (context + 3) // 4
    cells = pages * 4
    # 2 heads, head_dim 256, K and V; Q8 has fp16 scale per 64 codes.
    kv_bytes = allocated * cells * (1056 if kv == "q8" else 2048)
    index_bytes = allocated * ((context // 4 + 2) * 128 * 4 + 3 * 128 * 4 + 128 * 4)
    page_bytes = allocated * pages * 4
    rope_bytes = context * 32 * 4 * 2
    state_bytes = gdn * GDN_BYTES
    return dict(qsa=qsa, allocated_qsa=allocated, gdn=gdn, kv_bytes=kv_bytes,
                index_bytes=index_bytes, page_bytes=page_bytes, rope_bytes=rope_bytes,
                gdn_bytes=state_bytes, persistent_bytes=kv_bytes + index_bytes + page_bytes + rope_bytes + state_bytes)


def estimate(split, context=262144, kv="q8", baseline_kv="f16", dense_mib=64,
             workspace_mib=64, reserve_mib=1024):
    if len(split) != 3 or sum(split) != LAYERS or any(type(n) is not int or n <= 0 for n in split):
        raise ValueError("split must be three positive layer counts totaling 48")
    if any(type(n) is not int or n < 0 for n in (dense_mib, workspace_mib, reserve_mib)):
        raise ValueError("allowances must be nonnegative integers")
    rows, lo, old_lo = [], 0, 0
    for i, count in enumerate(split):
        hi, old_hi = lo + count, old_lo + BASE_SPLIT[i]
        state = persistent(lo, hi, context, kv)
        old = persistent(old_lo, old_hi, BASE_CONTEXT, baseline_kv)
        delta = count - BASE_SPLIT[i]
        # No dense-weight credit for removed layers: a conservative assumption, not a measured bound.
        dense_extra = max(delta, 0) * dense_mib * MIB
        workspace_extra = workspace_mib * MIB if context > BASE_CONTEXT else 0
        total = (BASE_USED_MIB[i] * MIB + delta * EXPERT_BYTES_PER_LAYER +
                 state["persistent_bytes"] - old["persistent_bytes"] + dense_extra + workspace_extra)
        free = VRAM_MIB[i] * MIB - total
        rows.append(dict(device=("cuda:0", "hip:0", "hip:1")[i], first_layer=lo, end_layer=hi,
                         layers=count, expert_slots=count * 512, expert_bytes=count * EXPERT_BYTES_PER_LAYER,
                         **state, baseline_used_bytes=BASE_USED_MIB[i] * MIB,
                         estimated_used_bytes=total, estimated_free_bytes=free,
                         reserve_bytes=reserve_mib * MIB, estimated_reserve_fit=free >= reserve_mib * MIB))
        lo, old_lo = hi, old_hi
    return dict(context=context, kv=kv, baseline_context=BASE_CONTEXT, baseline_kv=baseline_kv,
                dense_allowance_mib_per_added_layer=dense_mib, extra_long_workspace_mib=workspace_mib,
                reserve_mib=reserve_mib, plan=",".join(f"{r['device']}:{r['first_layer']}-{r['end_layer']}" for r in rows),
                stages=rows, total_kv_bytes=sum(r["kv_bytes"] for r in rows),
                total_persistent_bytes=sum(r["persistent_bytes"] for r in rows),
                qualification="ESTIMATE_ONLY")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", nargs=3, type=int, default=(15, 18, 15), metavar=("CUDA", "HIP0", "HIP1"))
    p.add_argument("--context", type=int, default=262144)
    p.add_argument("--kv", choices=("q8", "f16"), default="q8")
    p.add_argument("--baseline-kv", choices=("q8", "f16"), default="f16")
    p.add_argument("--dense-mib", type=int, default=64, help="assumed incremental dense MiB per added layer")
    p.add_argument("--workspace-mib", type=int, default=64, help="assumed extra MiB per device above 32K")
    p.add_argument("--reserve-mib", type=int, default=1024, help="desired remaining free MiB, not allocated")
    p.add_argument("--sweep", action="store_true", help="evaluate CUDA 12..18 with HIP1 fixed to 15")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    try:
        splits = [(n, 33 - n, 15) for n in range(12, 19)] if a.sweep else [a.split]
        results = [estimate(s, a.context, a.kv, a.baseline_kv, a.dense_mib, a.workspace_mib, a.reserve_mib) for s in splits]
    except ValueError as e:
        p.error(str(e))
    if a.json:
        print(json.dumps(results if a.sweep else results[0], indent=2))
    else:
        print("ESTIMATE ONLY; baseline report-0032 sampled maxima, baseline KV format assumed; no GPU allocation.")
        for r in results:
            print(f"\n{r['plan']} context={a.context} kv={a.kv} total KV={r['total_kv_bytes']/GIB:.3f} GiB")
            for s in r["stages"]:
                print(f"  {s['device']:6s} {s['layers']:2d} layers ({s['qsa']} QSA/{s['gdn']} GDN) "
                      f"slots={s['expert_slots']:5d} KV={s['kv_bytes']/MIB:7.1f} MiB "
                      f"used~{s['estimated_used_bytes']/GIB:5.2f} GiB free~{s['estimated_free_bytes']/GIB:5.2f} GiB "
                      f"reserve={'OK' if s['estimated_reserve_fit'] else 'LOW'}")
        print("Not a fit guarantee: exact dense weights, alignment, native scratch, other GPU allocations and peak must be measured.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
