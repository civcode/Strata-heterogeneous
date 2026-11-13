#!/usr/bin/env python3
"""M2.13: CPU-only two-run layer-zero HC precision-mode comparison.

Requires committed STL0/STFR artifacts from fresh unsplit HIP and original
CUDA->HIP->HIP runs under *the same* documented environment overrides.
A match here never upgrades M2.10 final-head numerical parity.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import heterogeneous_layer0_arithmetic as l0
import heterogeneous_compare_parity as parity

EXPECTED_REF = "hip:0:0-48"
EXPECTED_HET = "cuda:0:0-12,hip:0:12-30,hip:1:30-48"


def compare(reference: list[tuple[dict, dict]],
            heterogeneous: list[tuple[dict, dict]],
            arm: str) -> dict:
    if not arm or len(arm) > 64 or any(not (c.isascii() and
            (c.isalnum() or c in "_-")) for c in arm):
        raise parity.ParityError("arm must be a short alphanumeric label")
    if not reference or len(reference) != len(heterogeneous):
        raise parity.ParityError("different or empty committed window sets")
    if (reference[0][0]["plan"] != EXPECTED_REF or
            heterogeneous[0][0]["plan"] != EXPECTED_HET):
        raise parity.ParityError("expected full HIP reference and original mixed plan")
    rows = []
    for (rmeta, r), (hmeta, h) in zip(reference, heterogeneous):
        for field in ("model_fingerprint", "pack_index_sha256", "gguf_shards",
                      "profile_path", "pos0", "window_id", "token_ids", "n_keep"):
            if rmeta[field] != hmeta[field]:
                raise parity.ParityError(f"cross-backend source/window differs: {field}")
        if (r["token"] != h["token"] or
                r["position"] != h["position"]):
            raise parity.ParityError("STL0 token/position differs")
        measurement = l0.metric(r, h)
        rows.append({"position": rmeta["pos0"], "input_token": r["token"],
                     "first_nonidentical_component":
                         measurement["first_nonidentical_component"],
                     "components": measurement["components"]})
    return {
        "status": "DIAGNOSTIC_ONLY",
        "arm": arm,
        "interpretation": "Component agreement does not prove final logits or model parity.",
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-prefix", type=Path, required=True)
    parser.add_argument("--heterogeneous-prefix", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        report = compare(l0.read_run(args.reference_prefix),
                         l0.read_run(args.heterogeneous_prefix), args.arm)
        print(f"M2.13 {args.arm}: DIAGNOSTIC ONLY")
        for row in report["rows"]:
            print(f"position={row['position']} token={row['input_token']} "
                  f"first_component={row['first_nonidentical_component']}")
            for key, stats in row["components"].items():
                if key == "q8_input":
                    print(f"  {key}: differing_bytes={stats['mismatched_bytes']}")
                else:
                    print(f"  {key}: relativeL2={stats['relative_l2']:.9g} "
                          f"rmse={stats['rmse']:.9g} maxAbs={stats['max_abs']:.9g}")
        if args.report:
            if args.report.exists():
                raise parity.ParityError("report already exists; use a fresh path")
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return 0
    except (parity.ParityError, ValueError, OSError, MemoryError, OverflowError) as e:
        print(f"M2.13 HC mode comparison failed: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
