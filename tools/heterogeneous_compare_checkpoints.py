#!/usr/bin/env python3
"""Localize same-format numerical drift in bounded, paired checkpoint captures.

JSON manifests: kv_mode, plan, token_ids, complete=true, checkpoints in actual
execution order. Each checkpoint records position, layer (-1 for non-layer),
name, dtype (f32/f16/i8/u8), shape, and a relative raw little-endian file path.
No parity or model-quality admission is implied by this diagnostic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

FORMATS = {"f32": "f", "f16": "e", "i8": "b", "u8": "B"}
MAX_BYTES = 16 * 1024 * 1024
MAX_POINTS = 2048


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read_manifest(path):
    m = json.loads(path.read_text(encoding="utf-8"))
    need(m.get("complete") is True and m.get("kv_mode") in ("f16", "int8"),
         "incomplete or untagged capture")
    need(isinstance(m.get("plan"), str) and bool(m["plan"]), "missing plan")
    tokens = m.get("token_ids")
    need(isinstance(tokens, list) and 0 < len(tokens) <= 128 and
         all(type(t) is int and t >= 0 for t in tokens), "invalid token sequence")
    points = m.get("checkpoints")
    need(isinstance(points, list) and 0 < len(points) <= MAX_POINTS, "invalid checkpoint count")
    seen = set()
    previous = -1
    for p in points:
        pos, layer, name = p["position"], p["layer"], p["name"]
        need(type(pos) is int and previous <= pos < len(tokens) and pos >= 0 and
             type(layer) is int and -1 <= layer < 48 and isinstance(name, str) and bool(name),
             "invalid checkpoint identity/order")
        key = (pos, layer, name)
        need(key not in seen, f"duplicate checkpoint {key}")
        seen.add(key)
        previous = pos
        shape = p["shape"]
        need(p["dtype"] in FORMATS and isinstance(shape, list) and 1 <= len(shape) <= 8 and
             all(type(n) is int and n > 0 for n in shape), "invalid dtype or shape")
        need(math.prod(shape) * struct.calcsize(FORMATS[p["dtype"]]) <= MAX_BYTES,
             "checkpoint exceeds bounded size")
    return m


def raw(root, point):
    relative = Path(point["file"])
    need(not relative.is_absolute() and ".." not in relative.parts, "invalid relative checkpoint path")
    file = (root / relative).resolve(strict=True)
    need(root in file.parents and file.is_file(), "checkpoint outside capture directory")
    code = FORMATS[point["dtype"]]
    n = math.prod(point["shape"]) * struct.calcsize(code)
    need(file.stat().st_size == n, "checkpoint byte size differs from shape")
    with file.open("rb") as stream:
        blob = stream.read(n + 1)
    need(len(blob) == n, "checkpoint byte size changed during read")
    return blob, code


def compare(reference, candidate, atol=0.0, rtol=0.0):
    need(all(math.isfinite(x) and x >= 0 for x in (atol, rtol)), "invalid tolerance")
    a, b = read_manifest(reference), read_manifest(candidate)
    for key in ("kv_mode", "plan", "token_ids"):
        need(a[key] == b[key], f"capture identity differs: {key}")
    need(len(a["checkpoints"]) == len(b["checkpoints"]), "different checkpoint counts")
    rows = []
    for x, y in zip(a["checkpoints"], b["checkpoints"]):
        fields = ("position", "layer", "name", "dtype", "shape")
        need(all(x[k] == y[k] for k in fields), "checkpoint identity/order/shape differs")
        aa, code = raw(reference.resolve().parent, x)
        bb, _ = raw(candidate.resolve().parent, y)
        count = math.prod(x["shape"])
        maximum = squared = norm = 0.0
        violations = 0
        first = None
        for i, ((v,), (w,)) in enumerate(zip(struct.iter_unpack("<" + code, aa), struct.iter_unpack("<" + code, bb))):
            need(math.isfinite(v) and math.isfinite(w), f"nonfinite checkpoint: {x['name']}, index {i}")
            delta = abs(v-w)
            maximum = max(maximum, delta)
            squared += delta * delta
            norm += v * v
            if delta > atol + rtol * abs(v):
                violations += 1
                if first is None:
                    first = i
        rows.append(dict(**{k: x[k] for k in fields}, byte_equal=aa == bb,
                         reference_sha256=hashlib.sha256(aa).hexdigest(),
                         candidate_sha256=hashlib.sha256(bb).hexdigest(),
                         max_abs=maximum, rmse=math.sqrt(squared/count),
                         relative_l2=math.sqrt(squared/norm) if norm else None,
                         differing_elements=violations, first_element=first))
    return dict(status="DIAGNOSTIC_ONLY", atol=atol, rtol=rtol,
                first_byte_difference=next((r for r in rows if not r["byte_equal"]), None),
                first_numeric_difference=next((r for r in rows if r["differing_elements"]), None),
                checkpoints=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--atol", type=float, default=0.)
    p.add_argument("--rtol", type=float, default=0.)
    args = p.parse_args()
    try:
        need(args.reference.resolve() != args.candidate.resolve(), "distinct manifests required")
        print(json.dumps(compare(args.reference, args.candidate, args.atol, args.rtol), indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        print(f"Invalid checkpoint capture: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
