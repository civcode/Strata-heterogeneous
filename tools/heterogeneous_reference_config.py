#!/usr/bin/env python3
"""Prepare an unsplit [0,n_layers) HIP reference config from the trusted stage-0 v2 config.

CPU-only; does not modify model inputs or inspect GPU runtimes.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def make_reference(source: Path, target: Path, ordinal: int = 0) -> Path:
    if not source.is_absolute() or not target.is_absolute() or source == target:
        raise ValueError("source and distinct destination paths must be absolute")
    if target.exists():
        raise FileExistsError(f"reference config already exists: {target}")
    if ordinal < 0 or ordinal > 2147483647:
        raise ValueError("backend-local HIP device ordinal outside range")
    if not source.is_file() or source.stat().st_size > 16384:
        raise ValueError("stage-0 source config missing or too large")
    fields: dict[str, str] = {}
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        if line.count("=") != 1:
            raise ValueError("malformed source config")
        key, value = line.split("=", 1)
        if not key or not value or key in fields:
            raise ValueError("duplicate or empty source config field")
        fields[key] = value
    if (fields.get("version") != "2" or fields.get("stage_index") != "0" or
            fields.get("first_layer") != "0" or
            int(fields.get("end_layer", "0")) <= 1 or
            int(fields.get("n_layers", "0")) < int(fields["end_layer"])):
        raise ValueError("source must be a valid version-2 layer-zero stage configuration")
    for key in ("pack_dir", "profile_path", "index_sha256", "model_fingerprint",
                "gguf_shards", "ple_gguf_path", "arena_budget_bytes",
                "free_reserve_bytes", "max_cells", "max_t", "n_vocab",
                "hc", "n_embd", "cpu_workers", "max_cache_slots"):
        if key not in fields:
            raise ValueError(f"source missing required field {key}")
    if fields["ple_gguf_path"] not in fields["gguf_shards"].split(";"):
        raise ValueError("PLE source must be an original GGUF shard")
    # The staged arena budget is intentionally not reused: it was sized for
    # 12 layers, not all 48. Zero means no artificial cap, NOT no VRAM check.
    # Native loaders still check actual free memory and the safety reserve.
    fields.update({"stage_index": "0", "device_ordinal": str(ordinal),
                   "first_layer": "0", "end_layer": fields["n_layers"],
                   "arena_budget_bytes": "0"})
    text = "# Generated for M2.10 unsplit HIP reference; runtime remains opt-in via --run.\n"
    text += "".join(f"{key}={value}\n" for key, value in fields.items())
    if len(text.encode()) > 16384:
        raise ValueError("reference config exceeds native parser size limit")
    if not target.parent.is_dir():
        raise ValueError("destination directory does not exist")
    with target.open("x", encoding="utf-8") as output:
        output.write(text)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ordinal", type=int, default=0)
    args = parser.parse_args()
    try:
        print("CREATED: CPU-only full-model reference config:",
              make_reference(args.source, args.output, args.ordinal))
        return 0
    except (OSError, ValueError, KeyError, UnicodeError) as exc:
        print(f"FAIL: reference config: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
