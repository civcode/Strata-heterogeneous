#!/usr/bin/env python3
"""M2.11: locate first observable post-layer native verifier divergence.

Compares STLT/v1 per-layer hc-mean post-write residual taps from a completed
unsplit HIP reference against a completed mixed CUDA/HIP pipeline.
This diagnostic is NOT an M2.10 parity PASS/FAIL or a full R|bo|inject match.
"""
from __future__ import annotations

import argparse
from array import array
import json
import math
from pathlib import Path
import re
import struct
import sys

import heterogeneous_compare_parity as parity

HEADER = struct.Struct("<8IQQqQII")
MAGIC = 0x544C5453
VERSION = 1
MAX_BYTES = 32 * 1024 * 1024 + 72 + 8 * 4
STAGE = re.compile(r"(cuda|hip):([0-9]+):([0-9]+)-([0-9]+)\Z")


def checked(condition: bool, reason: str) -> None:
    if not condition:
        raise parity.ParityError(reason)


def decode_tap(path: Path) -> dict:
    checked(path.is_file(), f"missing layer-tap record: {path}")
    size = path.stat().st_size
    checked(HEADER.size <= size <= MAX_BYTES,
            f"invalid layer-tap record size: {path}")
    blob = path.read_bytes()
    (magic, version, stage, first, count, width, rows, total,
     request, window, pos0, fingerprint, checksum, reserved) = HEADER.unpack_from(blob)
    checked(magic == MAGIC and version == VERSION and reserved == 0,
            "STLT/v1 header invalid")
    checked(0 <= stage < 128 and 0 < total <= 128 and
            0 < count <= 48 and 0 < width <= 16384 and 0 < rows <= 8 and
            first < total - 1 and first + count <= total - 1 and
            request > 0 and window > 0 and pos0 >= 0 and fingerprint > 0,
            "STLT/v1 dimensions, layer range or identity invalid")
    expected = HEADER.size + 4 * rows + 4 * rows * count * width
    checked(size == expected, "STLT/v1 payload size invalid")
    digest = 2166136261
    for i, byte in enumerate(blob):
        if 64 <= i < 68:
            continue
        digest = ((digest ^ byte) * 16777619) & 0xFFFFFFFF
    checked(digest == checksum, "STLT/v1 checksum mismatch")
    tokens = struct.unpack_from(f"<{rows}i", blob, HEADER.size)
    checked(all(t >= 0 for t in tokens), "invalid STLT input token")
    samples = array("f")
    samples.frombytes(blob[HEADER.size + 4 * rows:])
    if sys.byteorder != "little":
        samples.byteswap()
    checked(len(samples) == rows * count * width and
            all(math.isfinite(v) for v in samples),
            "STLT/v1 nonfinite or incomplete post-layer values")
    return {"stage": stage, "first": first, "count": count,
            "width": width, "rows": rows, "total": total,
            "request": request, "window": window, "pos0": pos0,
            "fingerprint": fingerprint, "tokens": tokens,
            "samples": samples}


def parse_plan(spec: str) -> list[tuple[str, int, int, int]]:
    stages = []
    for part in spec.split(","):
        m = STAGE.fullmatch(part)
        checked(m is not None, "unrecognized plan stage in layer diagnostics")
        backend, ordinal, lo, hi = m.groups()
        ordinal, lo, hi = int(ordinal), int(lo), int(hi)
        checked(0 <= ordinal <= 64 and lo >= 0 and hi > lo,
                "invalid stage ordinal or layer range in plan")
        stages.append((backend, ordinal, lo, hi))
    checked(bool(stages) and stages[0][2] == 0 and
            all(stages[i - 1][3] == stages[i][2] for i in range(1, len(stages))),
            "non-contiguous layer plan")
    return stages


def load_run(prefix: Path) -> list[tuple[dict, dict, list[dict]]]:
    windows = parity.read_run(prefix)  # requires successful native shutdown
    plan = parse_plan(windows[0][0]["plan"])
    total = plan[-1][3]
    result = []
    for meta, packet in windows:
        records = []
        for stage_index, (_, _, lo, hi) in enumerate(plan):
            expected_count = min(hi, total - 1) - lo
            checked(expected_count > 0, "stage has no post-layer tap support")
            path = Path(f"{prefix}.stage{stage_index}.w{meta['window_id']}.stlt")
            record = decode_tap(path)
            checked(record["stage"] == stage_index and
                    record["first"] == lo and record["count"] == expected_count and
                    record["total"] == total and
                    record["width"] == 2560 and
                    record["rows"] == meta["rows"] and
                    record["request"] == packet["request"] and
                    record["window"] == packet["window"] and
                    record["pos0"] == meta["pos0"] and
                    record["fingerprint"] == packet["fingerprint"] and
                    list(record["tokens"]) == meta["token_ids"],
                    f"STLT stage {stage_index} identity/window mismatch")
            records.append(record)
        result.append((meta, packet, records))
    return result


def row_layer(records: list[dict], row: int, layer: int) -> array:
    for record in records:
        if record["first"] <= layer < record["first"] + record["count"]:
            width = record["width"]
            offset = (row * record["count"] + layer - record["first"]) * width
            return record["samples"][offset:offset + width]
    raise parity.ParityError(f"missing layer {layer} tap")


def compare(reference: list[tuple[dict, dict, list[dict]]],
            heterogeneous: list[tuple[dict, dict, list[dict]]],
            threshold: float) -> dict:
    checked(bool(reference) and len(reference) == len(heterogeneous),
            "mismatched or empty committed window counts")
    checked(math.isfinite(threshold) and threshold > 0,
            "invalid diagnostic relative-L2 threshold")
    ref_plan = parse_plan(reference[0][0]["plan"])
    het_plan = parse_plan(heterogeneous[0][0]["plan"])
    checked(len(ref_plan) == 1 and ref_plan[0][0] == "hip" and
            len(het_plan) > 1 and {s[0] for s in het_plan} == {"cuda", "hip"} and
            ref_plan[-1][3] == het_plan[-1][3] == 48,
            "M2.11 expects unsplit HIP and mixed CUDA/HIP 48-layer runs")
    results = []
    first_by_position = []
    for number, ((rm, rp, rr), (hm, hp, hr)) in enumerate(
            zip(reference, heterogeneous), 1):
        for field in ("model_fingerprint", "pack_index_sha256", "gguf_shards",
                      "profile_path", "pos0", "rows", "token_ids", "n_keep", "window_id"):
            checked(rm[field] == hm[field],
                    f"window {number} reference/heterogeneous {field} differs")
        checked(rp["vocab"] == hp["vocab"] and rr[0]["width"] == hr[0]["width"],
                "reference and heterogeneous shape differ")
        for row in range(rm["rows"]):
            first = None
            previous_relative = None
            for layer in range(47):
                a = row_layer(rr, row, layer)
                b = row_layer(hr, row, layer)
                width = len(a)
                checked(len(b) == width, "post-layer width mismatch")
                diff2 = ref2 = het2 = 0.0
                max_delta = 0.0
                for va, vb in zip(a, b):
                    d = float(va) - float(vb)
                    diff2 += d * d
                    ref2 += float(va) * float(va)
                    het2 += float(vb) * float(vb)
                    max_delta = max(max_delta, abs(d))
                rel = math.sqrt(diff2) / max(math.sqrt(ref2), 1e-30)
                rmse = math.sqrt(diff2 / width)
                if first is None and rel > threshold:
                    first = layer
                at_boundary = layer in (11, 12, 29, 30)
                results.append({
                    "window": number,
                    "position": rm["pos0"] + row,
                    "layer": layer,
                    "reference_backend": "HIP",
                    "heterogeneous_backend":
                        next(stage[0].upper() for stage in het_plan
                             if stage[2] <= layer < stage[3]),
                    "relative_l2": rel,
                    "rmse": rmse,
                    "max_abs": max_delta,
                    "reference_l2": math.sqrt(ref2),
                    "heterogeneous_l2": math.sqrt(het2),
                    "ratio_vs_previous":
                        None if previous_relative is None else
                        rel / max(previous_relative, 1e-12),
                    "stage_boundary_neighbor": at_boundary,
                })
                previous_relative = rel
            first_by_position.append({
                "window": number, "position": rm["pos0"] + row,
                "first_layer_above_diagnostic_threshold": first,
                "diagnostic_relative_l2_threshold": threshold,
            })
    return {
        "status": "DIAGNOSTIC_ONLY",
        "caution": "Layer taps are mean post-write residuals, not full R|bo|inject; "
                   "numerical parity remains failed until logits pass the M2.10 criterion.",
        "rows": first_by_position,
        "layer_metrics": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-prefix", type=Path, required=True)
    parser.add_argument("--heterogeneous-prefix", type=Path, required=True)
    parser.add_argument("--relative-threshold", type=float, default=0.001)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        result = compare(load_run(args.reference_prefix),
                         load_run(args.heterogeneous_prefix),
                         args.relative_threshold)
        print("M2.11 DIAGNOSTIC ONLY — M2.10 parity not presumed")
        for row in result["rows"]:
            print(f"position={row['position']} "
                  f"first layer with relative-L2>{args.relative_threshold:g}: "
                  f"{row['first_layer_above_diagnostic_threshold']}")
            observed = [s for s in result["layer_metrics"]
                        if s["position"] == row["position"]]
            for layer in (0, 1, 10, 11, 12, 13, 28, 29, 30, 31, 45, 46):
                m = observed[layer]
                print(f"  layer={layer:02d} hetero={m['heterogeneous_backend']} "
                      f"relL2={m['relative_l2']:.8g} "
                      f"rmse={m['rmse']:.8g} maxAbs={m['max_abs']:.8g}")
        if args.report is not None:
            args.report.write_text(json.dumps(result, indent=2) + "\n",
                                   encoding="utf-8")
        return 0
    except (parity.ParityError, ValueError, OSError, MemoryError,
            OverflowError, struct.error) as exc:
        print(f"M2.11 layer-tap diagnostic failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
