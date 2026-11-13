#!/usr/bin/env python3
"""Compare exact native layer-0 arithmetic from unsplit HIP, HIP-only split,
and CUDA->HIP->HIP runs, all with identical token/model inputs.

STL0/v1 captures *actual* Verifier embeddings, HC-mixed activation, raw Q8_1
input bytes and QKV projection output. It does not assume numerical parity.
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
MAGIC = 0x304c5453
MAX_SIZE = 1 << 20
COMPONENTS = ("embedding", "hc_mixed", "q8_input", "gdn_qkv")


def check(ok: bool, msg: str) -> None:
    if not ok:
        raise parity.ParityError(msg)


def load_probe(path: Path) -> dict:
    check(path.is_file(), f"missing layer-zero arithmetic record: {path}")
    size = path.stat().st_size
    check(HEADER.size + 4 <= size <= MAX_SIZE, f"invalid STL0 record size: {path}")
    b = path.read_bytes()
    (magic, version, stage, N, C, Q, rows, layers,
     request, window, pos0, fingerprint, crc, reserved) = HEADER.unpack_from(b)
    check(magic == MAGIC and version == 1 and reserved == 0 and
          stage == 0 and rows == 1 and layers == 48 and
          0 < N <= 16384 and N % 32 == 0 and
          0 < C <= 65536 and Q == (N // 32) * 36 and
          request == 1 and window > 0 and pos0 >= 0 and fingerprint > 0,
          "invalid STL0 header/geometry/identity")
    expect = HEADER.size + 4 + 4 * (2 * N + C) + Q
    check(size == expect, "STL0 payload length mismatch")
    h = 2166136261
    for i, byte in enumerate(b):
        if 64 <= i < 68:
            continue
        h = ((h ^ byte) * 16777619) & 0xffffffff
    check(h == crc, "STL0 checksum mismatch")
    token, = struct.unpack_from("<i", b, HEADER.size)
    check(token >= 0, "invalid STL0 input token")
    pos = HEADER.size + 4
    def floats(n: int):
        nonlocal pos
        ar = array("f")
        ar.frombytes(b[pos:pos + 4 * n])
        if sys.byteorder != "little":
            ar.byteswap()
        pos += 4 * n
        check(len(ar) == n and all(math.isfinite(v) for v in ar),
              "STL0 nonfinite or incomplete FP32 component")
        return ar
    emb = floats(N)
    mixed = floats(N)
    q8 = b[pos:pos + Q]
    pos += Q
    qkv = floats(C)
    check(pos == size, "STL0 trailing bytes")
    return {"window": window, "position": pos0, "plan_fingerprint": fingerprint,
            "token": token, "n_embd": N, "qkv_channels": C,
            "embedding": emb, "hc_mixed": mixed,
            "q8_input": q8, "gdn_qkv": qkv}


def read_run(prefix: Path) -> list[tuple[dict, dict]]:
    parsed = parity.read_run(prefix)
    result = []
    for meta, packet in parsed:
        check(meta["rows"] == 1, "STL0 requires exactly one input row per window")
        p = load_probe(Path(f"{prefix}.stage0.w{meta['window_id']}.stl0"))
        check(p["window"] == packet["window"] and
              p["position"] == meta["pos0"] and
              p["plan_fingerprint"] == packet["fingerprint"] and
              p["token"] == meta["token_ids"][0] and
              p["n_embd"] == 2560, "STL0 model/window identity mismatch")
        result.append((meta, p))
    return result


def floats_metric(a: array, b: array) -> dict:
    check(len(a) == len(b) and len(a) > 0, "component shape differs")
    absmax = 0.0
    diff2 = 0.0
    ref2 = 0.0
    neq = 0
    first = None
    for i, (x, y) in enumerate(zip(a, b)):
        d = float(x) - float(y)
        if x != y:
            neq += 1
            if first is None:
                first = i
        absmax = max(absmax, abs(d))
        diff2 += d * d
        ref2 += float(x) * float(x)
    return {"mismatched_values": neq, "first_mismatch_index": first,
            "max_abs": absmax, "rmse": math.sqrt(diff2 / len(a)),
            "relative_l2": math.sqrt(diff2) / max(math.sqrt(ref2), 1e-30)}


def metric(a, b) -> dict:
    check(a["n_embd"] == b["n_embd"] and
          a["qkv_channels"] == b["qkv_channels"],
          "different embedding/QKV dimensions")
    components = {}
    for key in COMPONENTS:
        if key == "q8_input":
            x, y = a[key], b[key]
            check(len(x) == len(y), "different raw Q8 input lengths")
            first = next((i for i, (v, w) in enumerate(zip(x, y)) if v != w), None)
            components[key] = {"byte_count": len(x),
                               "mismatched_bytes": sum(v != w for v, w in zip(x, y)),
                               "first_mismatch_index": first}
        else:
            components[key] = floats_metric(a[key], b[key])
    first_component = next((key for key in COMPONENTS
                            if (components[key]["mismatched_bytes"] if key == "q8_input"
                                else components[key]["mismatched_values"]) > 0), None)
    return {"first_nonidentical_component": first_component, "components": components}


def compare(reference: list[tuple[dict, dict]],
            control: list[tuple[dict, dict]],
            hetero: list[tuple[dict, dict]]) -> dict:
    check(len(reference) == len(control) == len(hetero) and bool(reference),
          "missing or mismatched complete window sequence")
    rp = reference[0][0]["plan"]
    cp = control[0][0]["plan"]
    hp = hetero[0][0]["plan"]
    check(rp.startswith("hip:") and rp.endswith(":0-48") and "," not in rp and
          cp == "hip:0:0-12,hip:1:12-48" and
          hp == "cuda:0:0-12,hip:0:12-30,hip:1:30-48",
          "expected unsplit HIP, HIP/HIP control and original mixed plan")
    output = []
    for (rm, r), (cm, c), (hm, h) in zip(reference, control, hetero):
        for key in ("model_fingerprint", "pack_index_sha256", "gguf_shards",
                    "profile_path", "pos0", "window_id", "token_ids", "n_keep"):
            check(rm[key] == cm[key] == hm[key],
                  f"cross-run model/input mismatch: {key}")
        check(r["token"] == c["token"] == h["token"] and
              r["position"] == c["position"] == h["position"],
              "STL0 token or position mismatch")
        results = {
            "hip_control_vs_reference": metric(r, c),
            "cuda_vs_reference": metric(r, h),
            "cuda_vs_hip_control": metric(c, h)
        }
        output.append({
            "position": rm["pos0"], "input_token": r["token"],
            "comparisons": results,
        })
    return {"status": "DIAGNOSTIC_ONLY", "windows": len(output),
            "note": "Exact STL0 component mismatches localize differences, "
                    "but do not prove correctness or numeric parity of complete layer",
            "positions": output}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-prefix", type=Path, required=True)
    parser.add_argument("--hip-control-prefix", type=Path, required=True)
    parser.add_argument("--heterogeneous-prefix", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        report = compare(read_run(args.reference_prefix),
                         read_run(args.hip_control_prefix),
                         read_run(args.heterogeneous_prefix))
        print("M2.12 layer-zero arithmetic DIAGNOSTIC ONLY")
        for position in report["positions"]:
            print(f"position={position['position']} input={position['input_token']}")
            for name, result in position["comparisons"].items():
                print(f"  {name}: first_nonidentical={result['first_nonidentical_component']}")
                for comp, stats in result["components"].items():
                    if comp == "q8_input":
                        print(f"    {comp}: bytes_differ={stats['mismatched_bytes']} "
                              f"first_byte={stats['first_mismatch_index']}")
                    else:
                        print(f"    {comp}: relL2={stats['relative_l2']:.8g} "
                              f"rmse={stats['rmse']:.8g} maxAbs={stats['max_abs']:.8g} "
                              f"values_differ={stats['mismatched_values']}")
        if args.report:
            args.report.write_text(json.dumps(report, indent=2) + "\n",
                                   encoding="utf-8")
        return 0
    except (parity.ParityError, ValueError, OSError, MemoryError,
            OverflowError, struct.error) as exc:
        print(f"M2.12 layer-zero diagnostic failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
