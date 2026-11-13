#!/usr/bin/env python3
"""Create an all-HIP two-stage control for M2.12 (CPU-only).

Reference: hip:0:0-48
Control:   hip:0:0-12,hip:1:12-48
Original:  cuda:0:0-12,hip:0:12-30,hip:1:30-48

Outputs are exclusive-create; existing production configs are never modified.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys


def fields(path: Path) -> dict[str, str]:
    if not path.is_absolute() or not path.is_file() or path.stat().st_size > 16384:
        raise ValueError(f"invalid absolute source config: {path}")
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        if line.count("=") != 1:
            raise ValueError(f"malformed v2 source line: {path}")
        key, value = line.split("=", 1)
        if not key or not value or key in result:
            raise ValueError(f"duplicate or empty config key: {path}: {key}")
        result[key] = value
    return result


def prepare(a: Path, b: Path, x: Path, y: Path, hip0: int = 0, hip1: int = 1) -> tuple[Path, Path]:
    if any(not p.is_absolute() for p in (a, b, x, y)):
        raise ValueError("all paths must be absolute")
    if len({a, b, x, y}) != 4:
        raise ValueError("source and output paths must be distinct")
    if x.exists() or y.exists():
        raise FileExistsError("HIP control output already exists; use fresh paths")
    if not x.parent.is_dir() or not y.parent.is_dir():
        raise ValueError("output parent directories must exist")
    if not (0 <= hip0 <= 2147483647 and 0 <= hip1 <= 2147483647) or hip0 == hip1:
        raise ValueError("distinct HIP ordinals required")
    first, rest = fields(a), fields(b)
    for f in (first, rest):
        if f.get("version") != "2":
            raise ValueError("only version 2 native configs are supported")
        for key in ("pack_dir", "profile_path", "index_sha256", "gguf_shards",
                    "model_fingerprint", "n_layers", "n_vocab", "hc", "n_embd",
                    "max_cells", "max_t", "free_reserve_bytes", "arena_budget_bytes",
                    "cpu_workers", "max_cache_slots"):
            if not f.get(key):
                raise ValueError(f"missing required native source field: {key}")
    if (first.get("stage_index"), first.get("first_layer"), first.get("end_layer")) != ("0", "0", "12") or (
            rest.get("stage_index"), rest.get("first_layer"), rest.get("end_layer")) != ("1", "12", "30"):
        raise ValueError("require canonical staged source [0,12) and [12,30)")
    if first.get("n_layers") != "48" or rest.get("n_layers") != "48" or not first.get("ple_gguf_path"):
        raise ValueError("expected Q2_0 48-layer stage zero with PLE shard")
    if rest.get("ple_gguf_path") is not None:
        raise ValueError("source stage one must not declare layer-one PLE")
    if first["ple_gguf_path"] not in first["gguf_shards"].split(";"):
        raise ValueError("PLE shard is not in original GGUF inputs")
    for key in ("pack_dir", "profile_path", "index_sha256", "gguf_shards",
                "model_fingerprint", "n_layers", "n_vocab", "hc", "n_embd",
                "max_cells", "max_t"):
        if first[key] != rest[key]:
            raise ValueError(f"cross-stage canonical model source differs: {key}")
    first.update({"device_ordinal": str(hip0), "stage_index": "0",
                  "first_layer": "0", "end_layer": "12",
                  "arena_budget_bytes": "0"})
    rest.update({"device_ordinal": str(hip1), "stage_index": "1",
                 "first_layer": "12", "end_layer": "48",
                 "arena_budget_bytes": "0"})
    def encode(f: dict[str, str]) -> str:
        out = "# M2.12 HIP-only split control; old stage-specific arena budget removed.\n"
        out += "".join(f"{key}={value}\n" for key, value in f.items())
        if len(out.encode()) > 16384:
            raise ValueError("HIP control config exceeds native parser limit")
        return out
    payload1, payload2 = encode(first), encode(rest)
    with x.open("x", encoding="utf-8") as out:
        out.write(payload1)
    try:
        with y.open("x", encoding="utf-8") as out:
            out.write(payload2)
    except Exception:
        x.unlink()
        raise
    return x, y


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-first", type=Path, required=True)
    ap.add_argument("--source-rest", type=Path, required=True)
    ap.add_argument("--output-first", type=Path, required=True)
    ap.add_argument("--output-rest", type=Path, required=True)
    ap.add_argument("--hip-first-ordinal", type=int, default=0)
    ap.add_argument("--hip-rest-ordinal", type=int, default=1)
    ns = ap.parse_args()
    try:
        result = prepare(ns.source_first, ns.source_rest, ns.output_first,
                         ns.output_rest, ns.hip_first_ordinal, ns.hip_rest_ordinal)
        print("CREATED: CPU-only HIP split control:", *result)
        return 0
    except (OSError, KeyError, UnicodeError, ValueError) as e:
        print(f"FAIL: HIP split control: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
