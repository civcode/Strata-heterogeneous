#!/usr/bin/env python3
"""Same-placement F16 vs INT8 STFR drift diagnostic; never a parity gate."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


def require(ok, message):
    if not ok:
        raise ValueError(message)


def metrics(a, b):
    require(len(a) == len(b) and len(a) >= 2, "invalid logit shape")
    require(all(math.isfinite(x) for x in a) and all(math.isfinite(x) for x in b),
            "nonfinite logits")
    at = max(range(len(a)), key=a.__getitem__)
    bt = max(range(len(b)), key=b.__getitem__)
    am, bm = a[at], b[bt]
    az = math.log(math.fsum(math.exp(x - am) for x in a))
    bz = math.log(math.fsum(math.exp(x - bm) for x in b))
    kl = math.fsum(math.exp(x - am - az) * ((x - am) - (y - bm) - az + bz)
                   for x, y in zip(a, b))
    return dict(f16_top1=at, int8_top1=bt, kl_f16_to_int8=max(0.0, kl),
                max_abs=max(abs(x-y) for x, y in zip(a, b)),
                rmse=math.sqrt(math.fsum((x-y)**2 for x, y in zip(a, b))/len(a)),
                f16_margin=am-max(x for i, x in enumerate(a) if i != at),
                int8_margin=bm-max(x for i, x in enumerate(b) if i != bt))


def compare(f16, int8):
    require(len(f16) == len(int8) and bool(f16), "different or empty window counts")
    rows = []
    for (fm, fp), (qm, qp) in zip(f16, int8):
        require(fm.get("kv_mode") == "f16" and qm.get("kv_mode") == "int8",
                "explicit F16 reference and INT8 candidate required")
        for key in ("plan", "model_fingerprint", "pack_index_sha256", "gguf_shards",
                    "profile_path", "window_id", "pos0", "rows", "n_keep", "token_ids"):
            require(fm[key] == qm[key], f"capture identity differs: {key}")
        require(fp["rows"] == qp["rows"] == fm["rows"] and fp["vocab"] == qp["vocab"],
                "different packet shapes")
        n = fp["vocab"]
        for i in range(fp["rows"]):
            row = metrics(fp["logits"][i*n:(i+1)*n], qp["logits"][i*n:(i+1)*n])
            row.update(window=fm["window_id"], position=fm["pos0"]+i,
                       input=fm["token_ids"][i], accepted=i < fm["n_keep"],
                       f16_prediction=fp["predictions"][i], int8_prediction=qp["predictions"][i])
            rows.append(row)
    def summary(group):
        mismatches = [r for r in group if r["f16_top1"] != r["int8_top1"] or
                      r["f16_prediction"] != r["int8_prediction"]]
        return dict(rows=len(group), top1_disagreements=sum(r["f16_top1"] != r["int8_top1"] for r in group),
                    prediction_disagreements=sum(r["f16_prediction"] != r["int8_prediction"] for r in group),
                    max_kl=max((r["kl_f16_to_int8"] for r in group), default=None),
                    max_abs=max((r["max_abs"] for r in group), default=None),
                    first_disagreement=mismatches[0] if mismatches else None)
    return dict(status="DIAGNOSTIC_ONLY", plan=f16[0][0]["plan"],
                accepted=summary([r for r in rows if r["accepted"]]),
                discarded=summary([r for r in rows if not r["accepted"]]), row_metrics=rows)


def main():
    from heterogeneous_compare_parity import read_run
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f16-prefix", type=Path, required=True)
    parser.add_argument("--int8-prefix", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        result = compare(read_run(args.f16_prefix, allow_partial=True),
                         read_run(args.int8_prefix, allow_partial=True))
        encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.report:
            args.report.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
        return 0
    except (OSError, ValueError, KeyError, TypeError, OverflowError, MemoryError) as exc:
        print(f"Invalid KV diagnostic capture: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
