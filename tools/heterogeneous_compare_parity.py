#!/usr/bin/env python3
"""Compare committed STFR/v1 logits from a native full-model reference and a heterogeneous run.

Requires matching token windows and identical configured GGUF shards/pack/profile.
No GPU APIs, no numpy, and no claim of an independent numerical implementation.
"""
from __future__ import annotations

import argparse
from array import array
import json
import math
from pathlib import Path
import struct
import sys

SCHEMA = "strata-heterogeneous-parity/v1"
HEADER = struct.Struct("<IHHIIIIQQQIIQ")
MAGIC = 0x52465453
MAX_PAYLOAD = 16 * 1024 * 1024


class ParityError(ValueError):
    pass


def require(condition: bool, description: str) -> None:
    if not condition:
        raise ParityError(description)


def read_stfr(path: Path) -> dict:
    require(path.is_file(), f"missing STFR packet: {path}")
    size = path.stat().st_size
    require(HEADER.size <= size <= HEADER.size + MAX_PAYLOAD,
            f"invalid STFR packet size: {path}")
    blob = path.read_bytes()
    (magic, version, flags, stage, rows, vocab, reserved,
     request, window, fingerprint, payload_bytes, checksum, tail) = HEADER.unpack_from(blob)
    require(magic == MAGIC and version == 1 and flags in (0, 1) and
            reserved == tail == 0, "invalid STFR header")
    require(0 < rows <= 8192 and 0 < vocab <= 2147483647 and
            request > 0 and window > 0 and fingerprint > 0,
            "invalid STFR identity/shape")
    require(size == HEADER.size + payload_bytes, "STFR truncated or extended")
    digest = 2166136261
    for index, byte in enumerate(blob):
        if 52 <= index < 56:
            continue
        digest = ((digest ^ byte) * 16777619) & 0xffffffff
    require(digest == checksum, "STFR checksum mismatch")
    tokens_end = HEADER.size + 4 * rows
    require(payload_bytes >= 4 * (rows + 1), "STFR missing prediction data")
    predictions = list(struct.unpack_from(f"<{rows}i", blob, HEADER.size))
    require(all(0 <= tok < vocab for tok in predictions), "invalid STFR predicted token")
    marker_count = struct.unpack_from("<I", blob, tokens_end)[0]
    require(marker_count <= rows, "STFR invalid nonfinite marker count")
    offset = tokens_end + 4 + marker_count * 4
    expected = 4 * (rows + 1 + marker_count) + (4 * rows * vocab if flags else 0)
    require(payload_bytes == expected, "STFR declared data size inconsistent")
    require(marker_count == 0, "STFR nonfinite row markers present")
    require(flags == 1, "full FP32 logits required; predictions alone are insufficient")
    logits = array("f")
    logits.frombytes(blob[offset:])
    if sys.byteorder != "little":
        logits.byteswap()
    require(len(logits) == rows * vocab and all(math.isfinite(v) for v in logits),
            "invalid or nonfinite STFR logits")
    return {"stage": stage, "rows": rows, "vocab": vocab, "request": request,
            "window": window, "fingerprint": fingerprint,
            "predictions": predictions, "logits": logits}


def read_run(prefix: Path, *, allow_partial: bool = False, placement: bool = False) -> list[tuple[dict, dict]]:
    """Read completed captures; legacy parity callers require full commits.

    For explicit partial-prefix callers, STFR contains all evaluated rows;
    only n_keep rows belong to the accepted prefix.
    """
    completed = Path(str(prefix) + ".complete.json")
    require(completed.is_file(), f"missing completion marker: {completed}")
    marker = json.loads(completed.read_text(encoding="utf-8"))
    require(marker.get("schema") == SCHEMA and
            type(marker.get("placement_capture", False)) is bool and
            marker.get("placement_capture", False) == placement and
            type(marker.get("windows")) is int and 1 <= marker["windows"] <= (128 if placement else 8) and
            type(marker.get("model_fingerprint")) is int and marker["model_fingerprint"] > 0 and
            isinstance(marker.get("plan"), str) and marker["plan"],
            "invalid parity completion marker")
    windows = []
    for number in range(1, marker["windows"] + 1):
        path = Path(f"{prefix}.w{number}")
        manifest_path = Path(str(path) + ".json")
        require(manifest_path.is_file(), f"missing window manifest: {manifest_path}")
        meta = json.loads(manifest_path.read_text(encoding="utf-8"))
        packet = read_stfr(Path(str(path) + ".stfr"))
        require(meta.get("schema") == SCHEMA and
                packet["stage"] == len(marker["plan"].split(",")) - 1 and
                meta.get("plan") == marker["plan"] and
                meta.get("model_fingerprint") == marker["model_fingerprint"] and
                meta.get("plan_fingerprint") == packet["fingerprint"] and
                meta.get("window_id") == number == packet["window"] and
                meta.get("rows") == packet["rows"] and
                type(meta.get("n_keep")) is int and
                1 <= meta["n_keep"] <= packet["rows"] and
                (allow_partial or meta["n_keep"] == packet["rows"]) and
                (not placement or (packet["rows"] == 1 and meta["n_keep"] == 1 and packet["vocab"] <= 248320)) and
                packet["request"] == 1 and
                type(meta.get("pos0")) is int and meta["pos0"] >= 0 and
                type(meta.get("token_ids")) is list and
                len(meta["token_ids"]) == packet["rows"] and
                all(type(t) is int and 0 <= t < packet["vocab"] for t in meta["token_ids"]),
                f"window {number} metadata does not match STFR commit")
        for field in ("pack_index_sha256", "gguf_shards", "profile_path"):
            require(isinstance(meta.get(field), str) and meta[field],
                    f"window {number} missing model source identity: {field}")
        if windows:
            previous_meta = windows[-1][0]
            require(meta["pos0"] == previous_meta["pos0"] + previous_meta["n_keep"],
                    "non-contiguous committed positions within run")
            require(all(meta[field] == previous_meta[field] for field in
                        ("model_fingerprint", "pack_index_sha256",
                         "gguf_shards", "profile_path")), "model source changed mid-run")
        else:
            require(meta["pos0"] == 0, "reference comparison must start from fresh position zero")
        windows.append((meta, packet))
    return windows


def compare(reference: list[tuple[dict, dict]],
            heterogeneous: list[tuple[dict, dict]], max_kl: float, *, placement: bool = False) -> dict:
    require(len(reference) == len(heterogeneous) and bool(reference),
            "different or empty window counts")
    ref_plan = reference[0][0]["plan"].split(",")
    het_plan = heterogeneous[0][0]["plan"].split(",")
    if placement:
        def placement_identity(plan):
            devices = []
            end = 0
            for stage in plan:
                backend, ordinal, layers = stage.split(":")
                lo, hi = map(int, layers.split("-"))
                require(backend in ("cuda", "hip") and ordinal.isdecimal() and lo == end < hi <= 48,
                        "invalid placement plan")
                devices.append((backend, int(ordinal)))
                end = hi
            require(end == 48 and len(set(devices)) == len(devices) and
                    {backend for backend, _ in devices} == {"cuda", "hip"}, "invalid mixed placement plan")
            return devices
        require(ref_plan != het_plan and placement_identity(ref_plan) == placement_identity(het_plan),
                "placement comparison requires distinct layer splits on identical ordered devices")
    else:
        require(len(ref_plan) == 1 and ref_plan[0].split(":")[0] in ("cuda", "hip") and
                ref_plan[0].endswith(":0-48") and
                len(het_plan) >= 2 and
                {stage.split(":")[0] for stage in het_plan} == {"cuda", "hip"},
                "parity requires one unsplit 48-layer reference and a mixed CUDA/HIP pipeline")
    require(math.isfinite(max_kl) and max_kl >= 0, "invalid KL threshold")
    rows = 0
    prediction_disagreements = 0
    top1_disagreements = 0
    worst_kl = 0.0
    worst_max_abs = 0.0
    worst_rmse = 0.0
    diagnostics = []
    for window, ((ref_meta, ref), (het_meta, het)) in enumerate(zip(reference, heterogeneous), 1):
        for key in ("model_fingerprint", "pack_index_sha256", "gguf_shards",
                    "profile_path", "window_id", "pos0", "rows", "n_keep", "token_ids"):
            require(ref_meta[key] == het_meta[key],
                    f"window {window} source/sequence differs: {key}")
        require(ref["rows"] == het["rows"] and ref["vocab"] == het["vocab"],
                f"window {window} head shape differs")
        count = ref["vocab"]
        for row in range(ref["rows"]):
            a = ref["logits"][row * count:(row + 1) * count]
            b = het["logits"][row * count:(row + 1) * count]
            require(len(a) == len(b) == count, "incomplete logits row")
            # Stabilized log-sum-exp; compute KL(reference || heterogeneous).
            am, bm = max(a), max(b)
            az = am + math.log(sum(math.exp(x - am) for x in a))
            bz = bm + math.log(sum(math.exp(x - bm) for x in b))
            kl = 0.0
            squared = 0.0
            max_abs = 0.0
            for x, y in zip(a, b):
                delta = float(x) - float(y)
                max_abs = max(max_abs, abs(delta))
                squared += delta * delta
                kl += math.exp(x - az) * (delta - az + bz)
            kl = max(0.0, kl)  # tiny negative accumulation noise is possible
            rmse = math.sqrt(squared / count)
            ref_top = max(range(count), key=a.__getitem__)
            het_top = max(range(count), key=b.__getitem__)
            ref_pred = ref["predictions"][row]
            het_pred = het["predictions"][row]
            prediction_disagreements += ref_pred != het_pred
            top1_disagreements += ref_top != het_top
            rows += 1
            worst_kl = max(worst_kl, kl)
            worst_max_abs = max(worst_max_abs, max_abs)
            worst_rmse = max(worst_rmse, rmse)
            diagnostics.append({"window": window, "position": ref_meta["pos0"] + row,
                                "input": ref_meta["token_ids"][row],
                                "reference_prediction": ref_pred,
                                "heterogeneous_prediction": het_pred,
                                "reference_top1": ref_top, "heterogeneous_top1": het_top,
                                "kl_ref_to_hetero": kl,
                                "max_abs_logit_difference": max_abs, "rmse_logits": rmse})
    passed = (prediction_disagreements == 0 and top1_disagreements == 0 and
              worst_kl <= max_kl)
    return {"status": "PASS" if passed else "FAIL", "rows": rows,
            "prediction_disagreements": prediction_disagreements,
            "top1_disagreements": top1_disagreements, "max_kl_ref_to_hetero": worst_kl,
            "max_abs_logit_difference": worst_max_abs, "max_rmse_logits": worst_rmse,
            "kl_threshold": max_kl, "comparison": "placement" if placement else "unsplit", "row_metrics": diagnostics}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-prefix", required=True, type=Path)
    parser.add_argument("--heterogeneous-prefix", required=True, type=Path)
    parser.add_argument("--max-kl", default=0.001, type=float,
                        help="maximum KL(reference || hetero) per row (default 0.001)")
    parser.add_argument("--report", type=Path, help="optional machine-readable JSON report")
    parser.add_argument("--placement", action="store_true", help="compare explicit one-row placement captures")
    args = parser.parse_args()
    try:
        result = compare(read_run(args.reference_prefix, placement=args.placement),
                         read_run(args.heterogeneous_prefix, placement=args.placement), args.max_kl,
                         placement=args.placement)
        print(json.dumps({k: v for k, v in result.items() if k != "row_metrics"},
                         indent=2, sort_keys=True))
        for row in result["row_metrics"]:
            print("window={window} pos={position} token={input} "
                  "pred={reference_prediction}/{heterogeneous_prediction} "
                  "top1={reference_top1}/{heterogeneous_top1} "
                  "KL={kl_ref_to_hetero:.8g} max_abs={max_abs_logit_difference:.7g}"
                  .format(**row))
        if args.report is not None:
            args.report.write_text(json.dumps(result, indent=2) + "\n",
                                   encoding="utf-8")
        return 0 if result["status"] == "PASS" else 1
    except (ParityError, OSError, ValueError, OverflowError, MemoryError) as exc:
        print(f"FAIL: M2.10 parity capture/identity: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
