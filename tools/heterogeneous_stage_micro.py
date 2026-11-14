#!/usr/bin/env python3
"""M2.17 exact native shared gate/down and pre-GDN convolution history.

STM0/v1 is a completed-window, opt-in capture. Existing STB0/v1 records
MUST agree bit-for-bit for its three overlapping fields. Nothing in this
diagnostic changes the M2.10 final-logit parity requirement.
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
import heterogeneous_stage_branch as branch

HEADER = struct.Struct("<8IQQqQII")
MAGIC = 0x304D5453  # STM0
REF_PLAN = branch.REF_PLAN
HET_PLAN = branch.HET_PLAN
FIELDS = ("shared_input", "gate_raw", "up_raw", "down_unscaled",
          "gate_scale", "shared_final", "layer6_qkv",
          "layer6_conv_history", "layer6_conv_output")
Q8_FIELDS = ("shared_input_q8", "swiglu_q8")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise parity.ParityError(message)


def field_counts(n: int, ff: int, c: int, d: int) -> tuple[int, ...]:
    return (n, ff, ff, n, 1, n, c, c * (d - 1), c)


def load_packet(path: Path) -> dict:
    require(path.is_file(), f"missing STM0 microprobe packet: {path}")
    size = path.stat().st_size
    require(HEADER.size + 4 <= size <= 4 * 1048576,
            "STM0 packet out of bounds")
    raw = path.read_bytes()
    (magic, version, stage, n, ff, c, d, layers,
     request, window, pos0, fingerprint, checksum, reserved) = HEADER.unpack_from(raw)
    require(magic == MAGIC and version == 1 and stage == 0 and
            n == 2560 and 0 < ff <= 65536 and ff % 32 == 0 and
            0 < c <= 16384 and d == 4 and layers == 48 and
            request == 1 and window > 0 and pos0 >= 0 and
            fingerprint > 0 and reserved == 0, "STM0 header, geometry or identity invalid")
    sizes = field_counts(n, ff, c, d)
    input_q8_len, swiglu_q8_len = (n // 32) * 36, (ff // 32) * 36
    expected = HEADER.size + 4 + sum(sizes) * 4 + input_q8_len + swiglu_q8_len
    require(size == expected, "STM0 component payload length mismatch")
    digest = 2166136261
    for i, byte in enumerate(raw):
        if i >= 64 and i < 68:
            continue
        digest = ((digest ^ byte) * 16777619) & 0xffffffff
    require(digest == checksum, "STM0 checksum mismatch")
    token, = struct.unpack_from("<i", raw, HEADER.size)
    require(0 <= token < 248320, "STM0 invalid token")
    pos = HEADER.size + 4
    result = {"n_embd": n, "n_ff": ff, "conv_channels": c, "conv_kernel": d,
              "request": request, "window": window, "pos0": pos0,
              "fingerprint": fingerprint, "token": token}
    for name, count in zip(FIELDS, sizes):
        vals = array("f")
        vals.frombytes(raw[pos:pos + count * 4])
        if sys.byteorder != "little":
            vals.byteswap()
        require(len(vals) == count and all(math.isfinite(x) for x in vals),
                f"STM0 nonfinite or incomplete {name}")
        result[name] = vals
        pos += count * 4
    result[Q8_FIELDS[0]] = raw[pos:pos + input_q8_len]
    pos += input_q8_len
    result[Q8_FIELDS[1]] = raw[pos:pos + swiglu_q8_len]
    pos += swiglu_q8_len
    require(pos == len(raw), "STM0 unexpected trailing bytes")
    return result


def read_run(prefix: Path) -> list[tuple[dict, dict]]:
    completed = parity.read_run(prefix)
    branches = branch.read_run(prefix)
    require(len(completed) == len(branches), "STM0 / STB0 window count mismatch")
    output = []
    for (meta, stfr), (branch_meta, old) in zip(completed, branches):
        require(meta["rows"] == meta["n_keep"] == 1 and
                meta["window_id"] == branch_meta["window_id"],
                "STM0 requires complete one-row STB0 windows")
        item = load_packet(Path(f"{prefix}.stage0.w{meta['window_id']}.stm0"))
        require(item["request"] == stfr["request"] and
                item["window"] == stfr["window"] and
                item["pos0"] == meta["pos0"] and
                item["fingerprint"] == stfr["fingerprint"] and
                item["token"] == meta["token_ids"][0],
                "STM0 model/window token/request/position identity mismatch")
        require(item["shared_final"] == old["layer5_shared"] and
                item["layer6_qkv"] == old["layer6_qkv"] and
                item["layer6_conv_output"] == old["layer6_conv"],
                "STM0/STB0 overlap mismatch: instrumentation or field layout inconsistent")
        require(item["n_embd"] == old["n_embd"] and
                item["conv_channels"] == old["conv_channels"],
                "STM0/STB0 cross-record geometry mismatch")
        output.append((meta, item))
    return output


def metric(a: array, b: array) -> dict:
    require(len(a) == len(b) and bool(a), "STM0 component width mismatch")
    ref2 = diff2 = peak = 0.0
    first = None
    count = 0
    for i, (x, y) in enumerate(zip(a, b)):
        delta = float(x) - float(y)
        ref2 += float(x) * float(x)
        diff2 += delta * delta
        peak = max(peak, abs(delta))
        if delta:
            count += 1
            if first is None:
                first = i
    return {"relative_l2": math.sqrt(diff2) / max(math.sqrt(ref2), 1e-30),
            "rmse": math.sqrt(diff2 / len(a)), "max_abs": peak,
            "different_values": count, "first_different_index": first}


def compare(reference: list[tuple[dict, dict]],
            heterogeneous: list[tuple[dict, dict]], threshold: float = 0.001) -> dict:
    require(bool(reference) and len(reference) == len(heterogeneous),
            "STM0 missing or mismatched committed windows")
    require(math.isfinite(threshold) and threshold > 0,
            "STM0 invalid diagnostic relative-L2 threshold")
    require(reference[0][0]["plan"] == REF_PLAN and
            heterogeneous[0][0]["plan"] == HET_PLAN,
            "STM0 requires full HIP reference and original mixed plan")
    rows = []
    for (rm, ref), (hm, het) in zip(reference, heterogeneous):
        for key in ("model_fingerprint", "pack_index_sha256", "gguf_shards",
                    "profile_path", "window_id", "pos0", "token_ids", "rows", "n_keep"):
            require(rm[key] == hm[key], f"STM0 model/input mismatch: {key}")
        for key in ("n_embd", "n_ff", "conv_channels", "conv_kernel"):
            require(ref[key] == het[key], f"STM0 cross-backend geometry mismatch: {key}")
        measurements = {key: metric(ref[key], het[key]) for key in FIELDS}
        for key in Q8_FIELDS:
            a, b = ref[key], het[key]
            require(len(a) == len(b), f"STM0 Q8 length mismatch: {key}")
            first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
            measurements[key] = {"mismatched_bytes": sum(x != y for x, y in zip(a, b)),
                                 "first_different_byte": first, "total_bytes": len(a)}
        ordered = ("shared_input", "shared_input_q8", "gate_raw", "up_raw",
                   "swiglu_q8", "down_unscaled", "gate_scale", "shared_final",
                   "layer6_qkv", "layer6_conv_history", "layer6_conv_output")
        def mismatched(key: str) -> bool:
            return (measurements[key]["mismatched_bytes"] > 0 if key in Q8_FIELDS
                    else measurements[key]["different_values"] > 0)
        first_nonzero = next((key for key in ordered if mismatched(key)), None)
        first_large = next((key for key in ordered if key not in Q8_FIELDS and
                            measurements[key]["relative_l2"] > threshold), None)
        rows.append({"window": rm["window_id"], "position": rm["pos0"],
                     "token": ref["token"], "first_nonidentical": first_nonzero,
                     "first_float_above_threshold": first_large,
                     "components": measurements})
    return {"status": "DIAGNOSTIC_ONLY", "threshold": threshold,
            "note": "Native microtrace localizes differences but does not prove kernel defect "
                    "or M2.10 final-logit parity. History mismatch implies prior-state investigation.",
            "positions": rows}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-prefix", type=Path, required=True)
    p.add_argument("--heterogeneous-prefix", type=Path, required=True)
    p.add_argument("--relative-threshold", type=float, default=0.001)
    p.add_argument("--report", type=Path)
    o = p.parse_args()
    try:
        r = compare(read_run(o.reference_prefix), read_run(o.heterogeneous_prefix),
                    o.relative_threshold)
        print("M2.17 EXACT SHARED/GDN MICROTRACE — DIAGNOSTIC ONLY")
        for item in r["positions"]:
            print(f"position={item['position']} token={item['token']} "
                  f"first_nonidentical={item['first_nonidentical']} "
                  f"first_above_threshold={item['first_float_above_threshold']}")
            for name, values in item["components"].items():
                if name in Q8_FIELDS:
                    print(f"  {name}: differing_bytes={values['mismatched_bytes']}")
                else:
                    print(f"  {name}: relativeL2={values['relative_l2']:.9g} "
                          f"rmse={values['rmse']:.9g} maxAbs={values['max_abs']:.9g}")
        if o.report:
            require(not o.report.exists(), "STM0 report already exists")
            o.report.write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
        return 0
    except (parity.ParityError, OSError, ValueError, OverflowError,
            MemoryError, struct.error) as e:
        print(f"M2.17 microtrace comparison failed: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
