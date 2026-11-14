#!/usr/bin/env python3
"""Analyze grouped serial/native/serial teacher-forced captures; no GPU dependency."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from heterogeneous_compare_parity import SCHEMA, MAX_PAYLOAD, compare, read_stfr, require

PREFILL_SCHEMA = "strata-heterogeneous-prefill-parity/v1"
MODES = ("serial-a", "native", "serial-b")
MAX_KL = 0.001  # Existing M2.10 threshold; do not tune it to the suspect run.


def load(prefix: Path):
    marker = json.loads(Path(f"{prefix}.prefill.json").read_text())
    require(marker.get("schema") == PREFILL_SCHEMA, "wrong prefill completion schema")
    cut, chunk, hc, embd, fingerprint = (marker.get(k) for k in
        ("prefix_rows", "chunk", "hc", "n_embd", "model_fingerprint"))
    require(all(type(x) is int and x > 0 for x in (cut, chunk, hc, embd, fingerprint)),
            "invalid prefill marker bounds")
    tokens = marker.get("token_ids")
    require(type(tokens) is list and cut < len(tokens) <= 8192 and
            len(tokens) - cut <= 128 and chunk <= 2048 and
            all(type(t) is int and 0 <= t <= 2147483647 for t in tokens),
            "invalid teacher-forced sequence")
    stride_bytes = hc * embd * 4
    require(0 < stride_bytes <= MAX_PAYLOAD, "invalid residual row size")
    plan = marker.get("plan")
    require(isinstance(plan, str), "missing plan")
    stages = plan.split(",")
    require(len(stages) == 3, "expected three physical stages")
    devices, end = [], 0
    for backend, spec in zip(("cuda", "hip", "hip"), stages):
        b, ordinal, layers = spec.split(":")
        lo, hi = map(int, layers.split("-"))
        require(b == backend and ordinal.isascii() and ordinal.isdecimal() and
                lo == end < hi, "invalid grouped plan")
        devices.append((b, int(ordinal)))
        end = hi
    require(len(set(devices)) == 3, "duplicate physical device")
    runs = {}
    source = None
    for request, mode in enumerate(MODES, 1):
        run = []
        for pos in range(cut, len(tokens)):
            window = pos - cut + 2 if mode == "native" else pos + 1
            base = f"{prefix}.{mode}.w{window}"
            meta = json.loads(Path(base + ".json").read_text())
            packet = read_stfr(Path(base + ".stfr"))
            require(meta.get("schema") == SCHEMA and meta.get("plan") == plan and
                    meta.get("model_fingerprint") == fingerprint and
                    meta.get("plan_fingerprint") == packet["fingerprint"] and
                    meta.get("window_id") == packet["window"] == window and
                    packet["request"] == request and packet["stage"] == 2 and
                    meta.get("rows") == packet["rows"] == meta.get("n_keep") == 1 and
                    meta.get("pos0") == pos and meta.get("token_ids") == [tokens[pos]],
                    f"{mode}: invalid capture identity at position {pos}")
            fields = tuple(meta.get(k) for k in ("pack_index_sha256", "gguf_shards", "profile_path"))
            require(all(isinstance(x, str) and x for x in fields), "missing model source")
            identity = (packet["fingerprint"], packet["vocab"], fields)
            if source is None:
                source = identity
                require(max(tokens) < packet["vocab"], "input outside vocabulary")
            require(identity == source, "capture model, plan fingerprint or vocabulary changed")
            run.append((meta, packet))
        runs[mode] = run
    # The command's window ID is not the residual-piece ID. Pieces use 1-based
    # absolute prompt positions, split both at chunk boundaries and wire cap.
    pieces, matched, first_mismatch, pos = 0, 0, None, 0
    cap = MAX_PAYLOAD // stride_bytes
    while pos < cut:
        rows = min(cap, chunk - pos % chunk, cut - pos)
        size = rows * stride_bytes
        a = Path(f"{prefix}.stage0.r2.w{pos+1}.out.f32")
        b = Path(f"{prefix}.stage1.r2.w{pos+1}.in.f32")
        require(a.is_file() and b.is_file() and a.stat().st_size == b.stat().st_size == size,
                f"missing or incorrect physical residual capture at position {pos}")
        pieces += 1
        if a.read_bytes() == b.read_bytes():
            matched += 1
        elif first_mismatch is None:
            first_mismatch = pos
        pos += rows
    return marker, runs, {"pieces": pieces, "matched_pieces": matched,
                          "first_mismatched_piece_position": first_mismatch,
                          "status": "PASS" if pieces == matched else "FAIL"}


def analyze(prefix: Path) -> dict:
    marker, runs, bridge = load(prefix)
    comparisons = {}
    for a, b in (("serial-a", "serial-b"), ("serial-a", "native"), ("serial-b", "native")):
        comparisons[f"{a}_vs_{b}"] = compare(runs[a], runs[b], MAX_KL, prefill=True)
    passed = bridge["status"] == "PASS" and all(x["status"] == "PASS" for x in comparisons.values())
    return {"status": "PASS" if passed else "FAIL", "prefix_rows": marker["prefix_rows"],
            "chunk": marker["chunk"], "bridge": bridge, "comparisons": comparisons}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        result = analyze(args.prefix)
        if args.report is not None:
            args.report.write_text(json.dumps(result, indent=2) + "\n")
        summary = dict(result)
        summary["comparisons"] = {name: {k: v for k, v in comparison.items() if k != "row_metrics"}
                                  for name, comparison in result["comparisons"].items()}
        print(json.dumps(summary, indent=2))
        return 0 if result["status"] == "PASS" else 1
    except (OSError, ValueError, TypeError, KeyError, OverflowError, MemoryError) as exc:
        print(f"INVALID prefill parity capture: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
