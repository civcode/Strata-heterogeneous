#!/usr/bin/env python3
"""Compare read-only committed CUDA prefix state from the M7 A/B/A diagnostic.

Byte equality (including the native indexer's checkpoint scratch) is diagnostic,
not the final logits parity gate. K/V files contain only committed cells.
"""
from __future__ import annotations

import argparse
from array import array
import json
import math
from pathlib import Path
import sys

from heterogeneous_compare_parity import require

SCHEMA = "strata-heterogeneous-prefix-state/v1"
CAP = 256 * 1024 * 1024
MODES = ("serial-a", "native", "serial-b")


def load(prefix: Path):
    run = json.loads(Path(f"{prefix}.prefill.json").read_text())
    cut = run.get("prefix_rows")
    tokens = run.get("token_ids")
    require(run.get("schema") == "strata-heterogeneous-prefill-parity/v1" and
            run.get("state_capture") is True and type(cut) is int and 1 <= cut <= 16 and
            type(tokens) is list and cut < len(tokens) <= 8192 and
            all(type(t) is int and 0 <= t <= 2147483647 for t in tokens),
            "missing completed short-prefix state capture")
    plan = run.get("plan")
    require(isinstance(plan, str), "missing plan")
    stages = plan.split(",")
    require(len(stages) == 3 and [s.split(":")[0] for s in stages] == ["cuda", "hip", "hip"],
            "wrong physical topology")
    _, _, layers = stages[0].split(":")
    lo, hi = map(int, layers.split("-"))
    require(lo == 0 and 4 <= hi <= 48 and hi // 4 > 0, "invalid entry carve")
    # This fork's worker validates the Qwen Flash-Next four-layer QSA interval.
    qsa_count = hi // 4
    gdn_count = hi - qsa_count
    expected = {"ple.history": "f32"}
    for j in range(gdn_count):
        for name in ("recurrence", "conv"):
            expected[f"gdn{j}.{name}"] = "f32"
    for j in range(qsa_count):
        for name, kind in (("tail", "f32"), ("dead", "f32"), ("block_pos", "i32"),
                           ("k", "opaque"), ("v", "opaque"), ("k_scale", "opaque"),
                           ("v_scale", "opaque"), ("pooled", "f32")):
            expected[f"qsa{j}.{name}"] = kind
    results = {}
    identity = None
    for request, mode in enumerate(MODES, 1):
        stem = f"{prefix}.state.r{request}.p{cut}"
        meta = json.loads(Path(stem + ".json").read_text())
        require(meta.get("schema") == SCHEMA and meta.get("stage") == 0 and
                meta.get("request") == request and meta.get("position") == cut and
                (meta.get("layer_lo"), meta.get("layer_hi")) == (0, hi) and
                (meta.get("gdn_ord0"), meta.get("gdn_alloc"),
                 meta.get("qsa_ord0"), meta.get("qsa_alloc")) == (0, gdn_count, 0, qsa_count),
                f"{mode}: invalid stage, request, extent or layer identity")
        prev = meta.get("ple_prev")
        require(type(prev) is list and len(prev) == 2 and
                all(type(t) is int and -1 <= t <= 2147483647 for t in prev), "invalid PLE token history")
        layouts = meta.get("qsa")
        require(type(layouts) is list and len(layouts) == qsa_count, "missing native QSA layout")
        for j, q in enumerate(layouts):
            require(q.get("ordinal") == j and type(q.get("format")) is int and
                    all(type(q.get(k)) is int and q[k] > 0 for k in
                        ("cells", "page_size", "heads", "head_dim", "pooled_rows", "idx_dim")) and
                    q["cells"] >= cut and q["cells"] % q["page_size"] == 0,
                    "invalid native QSA shape")
        fields = {}
        require(type(meta.get("fields")) is list and len(meta["fields"]) == len(expected),
                "wrong field count")
        total = 0
        for f in meta["fields"]:
            name, kind, size = f.get("name"), f.get("type"), f.get("bytes")
            require(isinstance(name, str) and name in expected and name not in fields and
                    kind == expected[name] and type(size) is int and 0 <= size <= CAP - total,
                    "invalid or duplicate state field")
            require((kind != "f32" or size > 0 and size % 4 == 0) and
                    (kind != "i32" or size == 4), "misaligned numeric state field")
            if kind == "opaque":
                require(size % cut == 0 and (name.endswith("_scale") or size > 0),
                        "invalid committed K/V extent")
            if name.endswith(".pooled"):
                q = layouts[int(name[3:name.index(".")])]
                require(size == q["pooled_rows"] * q["idx_dim"] * 4, "invalid pooled shape")
            path = Path(f"{stem}.{name}.bin")
            require(path.is_file() and path.stat().st_size == size, f"{mode}: missing or truncated {name}")
            fields[name] = (kind, size, path)
            total += size
        shape = ({n: (k, s) for n, (k, s, _) in fields.items()}, layouts)
        if identity is None:
            identity = shape
        require(shape == identity, "snapshot shapes differ across passes")
        results[mode] = (prev, fields)
    return run, results, ([-1] + tokens[:cut])[-2:]


def metric(a: bytes, b: bytes, kind: str) -> dict:
    same = a == b
    result = {"bytes_equal": same}
    if kind == "opaque":
        if not same:
            result["first_different_byte"] = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
        return result
    x, y = array("f" if kind == "f32" else "i"), array("f" if kind == "f32" else "i")
    x.frombytes(a)
    y.frombytes(b)
    if sys.byteorder != "little":
        x.byteswap()
        y.byteswap()
    if kind == "i32":
        result.update(a=list(x), b=list(y))
        return result
    nonfinite = sum(not math.isfinite(v) for v in x) + sum(not math.isfinite(v) for v in y)
    result["nonfinite_values"] = nonfinite
    if nonfinite:
        return result
    if same:
        result.update(max_abs=0.0, rmse=0.0)
        return result
    maximum, squared, first = 0.0, 0.0, None
    for i, (v, w) in enumerate(zip(x, y)):
        d = float(v) - float(w)
        if d and first is None:
            first = i
        maximum = max(maximum, abs(d))
        squared += d * d
    result.update(max_abs=maximum, rmse=math.sqrt(squared / len(x)), first_different_value=first)
    return result


def analyze(prefix: Path) -> dict:
    run, captures, expected = load(prefix)
    history = {mode: prev for mode, (prev, _) in captures.items()}
    comparisons = {}
    identical = all(prev == expected for prev in history.values())
    for a, b in (("serial-a", "serial-b"), ("serial-a", "native"), ("serial-b", "native")):
        fields = {}
        for name, (kind, _, ap) in captures[a][1].items():
            bp = captures[b][1][name][2]
            item = metric(ap.read_bytes(), bp.read_bytes(), kind)
            fields[name] = item
            identical &= item["bytes_equal"] and not item.get("nonfinite_values", 0)
        comparisons[f"{a}_vs_{b}"] = fields
    return {"status": "IDENTICAL" if identical else "DIFFERENT",
            "scope": "CUDA committed state only; diagnostic, not logits parity",
            "position": run["prefix_rows"], "expected_ple_prev": expected,
            "observed_ple_prev": history, "comparisons": comparisons}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prefix", required=True, type=Path)
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    try:
        report = analyze(args.prefix)
        if args.report:
            args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "comparisons"}, indent=2))
        for name, fields in report["comparisons"].items():
            diff = {k: v for k, v in fields.items()
                    if not v["bytes_equal"] or v.get("nonfinite_values", 0)}
            print(name, json.dumps(diff, allow_nan=False))
        return 0 if report["status"] == "IDENTICAL" else 1
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError, MemoryError) as exc:
        print(f"INVALID prefix state capture: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
