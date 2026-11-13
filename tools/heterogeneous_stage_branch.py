#!/usr/bin/env python3
"""M2.15: compare exact layer-5 router/expert and layer-6 GDN intermediates.

STB0/v1 is captured by opt-in native verifiers. This CPU-only tool requires
fully committed STFR/v1 evidence and exact model/window provenance.
Numeric diagnostics are NOT a full-head KL acceptance or root-cause proof.
"""
from __future__ import annotations

import argparse
from array import array
import json
import math
from pathlib import Path
import struct
import sys

import heterogeneous_compare_parity as parity

HEADER = struct.Struct("<8IQQqQII")
MAGIC = 0x30425453
REF_PLAN = "hip:0:0-48"
HET_PLAN = "cuda:0:0-12,hip:0:12-30,hip:1:30-48"
FLOAT_FIELDS = (
    "layer5_router_logits", "layer5_route_weights", "layer5_shared",
    "layer5_expert_parts", "layer5_moe_bo", "layer6_qkv", "layer6_conv",
    "layer6_gate", "layer6_beta", "layer6_z", "layer6_recurrent_y",
    "layer6_gdn_bo",
)


def require(ok: bool, description: str) -> None:
    if not ok:
        raise parity.ParityError(description)


def lengths(p: dict) -> tuple[int, ...]:
    n, ne, k, c, hv, zv = (p[x] for x in ("n_embd", "n_expert", "k",
                                         "conv_channels", "value_heads", "value_dim"))
    return ne, k, n, k * n, n, c, c, hv, hv, zv, zv, n


def read_packet(path: Path) -> dict:
    require(path.is_file(), f"missing STB0 packet: {path}")
    size = path.stat().st_size
    require(HEADER.size + 4 <= size <= (1 << 20), "STB0 file size out of bounds")
    blob = path.read_bytes()
    (magic, version, stage, n, ne, k, c, hv,
     request, window, pos0, fingerprint, checksum, zv) = HEADER.unpack_from(blob)
    require(magic == MAGIC and version == 1 and stage == 0 and
            n == 2560 and ne == 512 and k == 10 and
            0 < c <= 16384 and 0 < hv <= 1024 and 0 < zv <= 16384 and
            request == 1 and window > 0 and pos0 >= 0 and fingerprint > 0,
            "STB0 invalid header, shape or request/window identity")
    data = {"n_embd": n, "n_expert": ne, "k": k, "conv_channels": c,
            "value_heads": hv, "value_dim": zv, "window": window,
            "request": request, "pos0": pos0, "fingerprint": fingerprint}
    field_counts = lengths(data)
    expected = HEADER.size + 4 * (1 + k + sum(field_counts))
    require(size == expected, "STB0 component payload size mismatch")
    digest = 2166136261
    for i, byte in enumerate(blob):
        if 64 <= i < 68:
            continue
        digest = ((digest ^ byte) * 16777619) & 0xffffffff
    require(digest == checksum, "STB0 checksum mismatch")
    token, = struct.unpack_from("<i", blob, HEADER.size)
    require(0 <= token < 248320, "STB0 token ID invalid")
    data["token"] = token
    ids = list(struct.unpack_from(f"<{k}i", blob, HEADER.size + 4))
    require(all(0 <= ident < ne for ident in ids), "STB0 expert IDs invalid")
    data["layer5_route_ids"] = ids
    samples = array("f")
    samples.frombytes(blob[HEADER.size + 4 * (1 + k):])
    if sys.byteorder != "little":
        samples.byteswap()
    require(len(samples) == sum(field_counts) and
            all(math.isfinite(x) for x in samples),
            "STB0 incomplete or nonfinite FP32 data")
    offset = 0
    for name, count in zip(FLOAT_FIELDS, field_counts):
        data[name] = samples[offset:offset + count]
        offset += count
    return data


def read_run(prefix: Path) -> list[tuple[dict, dict]]:
    result = []
    for meta, packet in parity.read_run(prefix):
        require(meta["rows"] == meta["n_keep"] == 1,
                "STB0 capture requires one committed row per window")
        capture = read_packet(Path(f"{prefix}.stage0.w{meta['window_id']}.stb0"))
        require(capture["request"] == packet["request"] and
                capture["window"] == packet["window"] and
                capture["pos0"] == meta["pos0"] and
                capture["fingerprint"] == packet["fingerprint"] and
                capture["token"] == meta["token_ids"][0],
                "STB0 input, position, request or plan identity mismatch")
        result.append((meta, capture))
    return result


def difference(a: array, b: array) -> dict:
    require(len(a) == len(b) and len(a) > 0, "STB0 component shape mismatch")
    ref2 = diff2 = max_abs = 0.0
    n_diff = 0
    first = None
    for i, (x, y) in enumerate(zip(a, b)):
        delta = float(x) - float(y)
        ref2 += float(x) * float(x)
        diff2 += delta * delta
        max_abs = max(max_abs, abs(delta))
        if delta:
            n_diff += 1
            if first is None:
                first = i
    return {"relative_l2": math.sqrt(diff2) / max(math.sqrt(ref2), 1e-30),
            "rmse": math.sqrt(diff2 / len(a)), "max_abs": max_abs,
            "different_values": n_diff, "first_different_index": first}


def compare(reference: list[tuple[dict, dict]],
            heterogeneous: list[tuple[dict, dict]]) -> dict:
    require(bool(reference) and len(reference) == len(heterogeneous),
            "STB0 missing or mismatched committed windows")
    require(reference[0][0]["plan"] == REF_PLAN and
            heterogeneous[0][0]["plan"] == HET_PLAN,
            "STB0 requires unsplit HIP reference and original mixed plan")
    output = []
    for (rm, r), (hm, h) in zip(reference, heterogeneous):
        for key in ("model_fingerprint", "pack_index_sha256", "gguf_shards",
                    "profile_path", "window_id", "pos0", "token_ids",
                    "rows", "n_keep"):
            require(rm[key] == hm[key], f"STB0 model/window identity differs: {key}")
        for key in ("n_embd", "n_expert", "k", "conv_channels",
                    "value_heads", "value_dim"):
            require(r[key] == h[key], f"STB0 component geometry differs: {key}")
        ids_match = r["layer5_route_ids"] == h["layer5_route_ids"]
        first_id_diff = next((i for i, (x, y) in enumerate(
            zip(r["layer5_route_ids"], h["layer5_route_ids"])) if x != y), None)
        metrics = {name: difference(r[name], h[name]) for name in FLOAT_FIELDS}
        k = r["k"]
        n = r["n_embd"]
        per_expert = [
            {"slot": j, "reference_expert": r["layer5_route_ids"][j],
             "heterogeneous_expert": h["layer5_route_ids"][j],
             **difference(r["layer5_expert_parts"][j * n:(j + 1) * n],
                          h["layer5_expert_parts"][j * n:(j + 1) * n])}
            for j in range(k)
        ]
        first_numeric = next((name for name in FLOAT_FIELDS
                              if metrics[name]["different_values"] > 0), None)
        output.append({
            "window": rm["window_id"], "position": rm["pos0"],
            "input_token": r["token"], "route_ids_match": ids_match,
            "first_route_id_mismatch_slot": first_id_diff,
            "reference_route_ids": r["layer5_route_ids"],
            "heterogeneous_route_ids": h["layer5_route_ids"],
            "first_nonidentical_float_component": first_numeric,
            "components": metrics, "expert_slots": per_expert,
        })
    return {
        "status": "DIAGNOSTIC_ONLY", "positions": output,
        "note": "Branch source localization, not first causal operation or proof of parity; "
                "routing/parts are only comparable elementwise where expert IDs align.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-prefix", type=Path, required=True)
    parser.add_argument("--heterogeneous-prefix", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        result = compare(read_run(args.reference_prefix),
                         read_run(args.heterogeneous_prefix))
        print("M2.15 BRANCH DIAGNOSTIC ONLY")
        for row in result["positions"]:
            print(f"position={row['position']} token={row['input_token']} "
                  f"route_ids_match={row['route_ids_match']} "
                  f"first_float_difference={row['first_nonidentical_float_component']}")
            for name, m in row["components"].items():
                print(f"  {name}: relL2={m['relative_l2']:.8g} "
                      f"rmse={m['rmse']:.8g} maxAbs={m['max_abs']:.8g} "
                      f"different={m['different_values']}")
        if args.report:
            require(not args.report.exists(), "report already exists; choose fresh path")
            args.report.write_text(json.dumps(result, indent=2) + "\n",
                                   encoding="utf-8")
        return 0
    except (parity.ParityError, ValueError, OSError, OverflowError,
            MemoryError, struct.error) as exc:
        print(f"M2.15 branch diagnostic failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
