#!/usr/bin/env python3
"""CPU-only Q2_0 cross-stage model/plan preflight. Does not open a GPU."""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

from gguf_reader import GGUFFile

EXPECTED_EXPERTS = 48 * 512 * 1_382_400
EXPECTED_GEOMETRY = (48, 4, 2560, 248320)
EXPERT_PARTS = ("gate", "up", "down")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def plan_stages(spec: str, layers: int):
    stages = []
    cursor = 0
    used = set()
    for entry in spec.split(","):
        match = re.fullmatch(r"(cuda|hip):([0-9]+):([0-9]+)-([0-9]+)", entry)
        require(match is not None, f"invalid stage descriptor: {entry!r}")
        backend, ordinal, first, end = match.groups()
        ordinal, first, end = map(int, (ordinal, first, end))
        require(first == cursor and first < end <= layers,
                "stage ranges must partition [0,n_layers) in order")
        require((backend, ordinal) not in used, "duplicate backend-local device ordinal")
        used.add((backend, ordinal))
        stages.append((backend, ordinal, first, end))
        cursor = end
    require(cursor == layers and {s[0] for s in stages} == {"cuda", "hip"},
            "plan must cover all layers and include both CUDA and HIP")
    return stages


def config(path: Path) -> dict[str, str]:
    require(path.is_absolute() and path.is_file(), f"config is not an absolute file: {path}")
    require(path.stat().st_size <= 16384, "oversized worker config")
    fields = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if not line or line.startswith("#"):
            continue
        require(line.count("=") == 1, "invalid key=value line")
        key, value = line.split("=", 1)
        require(key and value and key not in fields, "empty or duplicate config key")
        fields[key] = value
    return fields


def read_profile(path: Path) -> None:
    require(path.is_file() and path.stat().st_size == 196632,
            "canonical 48x512 STRP profile size mismatch")
    with path.open("rb") as stream:
        header = stream.read(24)
        require(len(header) == 24 and header[:4] == b"STRP" and
                struct.unpack("<5I", header[4:]) == (1, 48, 512, 24576, 24576),
                "canonical 48x512 STRP/v1 header mismatch")
        seen = set()
        inverse = {}
        for slot in range(24576):
            pair = stream.read(4)
            require(len(pair) == 4, "truncated STRP pair table")
            layer, expert = struct.unpack("<HH", pair)
            require(layer < 48 and expert < 512 and (layer, expert) not in seen,
                    "duplicate or invalid expert profile pair")
            seen.add((layer, expert))
            inverse[(layer, expert)] = slot
        for layer in range(48):
            for expert in range(512):
                slot = stream.read(4)
                require(len(slot) == 4 and struct.unpack("<I", slot)[0] == inverse[(layer, expert)],
                        "STRP inverse table does not match pair ranking")


def check_gguf(shards: list[Path]) -> Path:
    require(0 < len(shards) <= 64, "no original GGUF shards")
    original_names = set()
    roles = set()
    ple = None
    for i, shard in enumerate(shards):
        require(shard.is_absolute() and shard.is_file(), "GGUF shard must be an absolute file")
        obj = GGUFFile(shard)  # original Strata's bounded GGUF v3 header reader
        if i == 0:
            require(obj.metadata.get("general.architecture") == "qwen4exp",
                    "primary GGUF architecture must be qwen4exp")
        for tensor in obj.tensors:
            require(tensor.name not in original_names, "duplicate GGUF tensor across shards")
            original_names.add(tensor.name)
            match = re.fullmatch(r"blk\.([0-9]+)\.ffn_(gate|up|down)_exps\.weight", tensor.name)
            if match:
                layer = int(match.group(1))
                role = match.group(2)
                require(layer < 48 and match.group(1) == str(layer) and
                        tensor.type_name == "Q2_0", "noncanonical routed Q2_0 GGUF expert")
                roles.add((layer, role))
            if tensor.name == "per_layer_token_embd.weight":
                require(ple is None and tensor.type_name == "IQ4_NL" and
                        tensor.shape == [160, 320001536], "invalid/duplicate PLE GGUF tensor")
                ple = shard
    require(ple is not None, "no PLE table in original GGUF shards")
    require(roles == {(layer, part) for layer in range(48) for part in EXPERT_PARTS},
            "missing Q2_0 gate/up/down expert tensor(s) in original GGUF")
    return ple


def check_pack(root: Path, stages) -> None:
    require(root.is_absolute() and root.is_dir(), "pack_dir is not an absolute directory")
    require(not (root / "native_experts.txt").exists() and
            not (root / "native_experts.txt").is_symlink(),
            "native/IQ expert pack cannot be used as canonical Q2_0")
    for file in ("dense.bin", "embd.bin", "experts.bin", "index.txt"):
        p = root / file
        require(p.is_file() and p.stat().st_size > 0, f"missing or empty {file}")
    require((root / "experts.bin").stat().st_size == EXPECTED_EXPERTS,
            "experts.bin is not canonical 48x512x1382400 Q2_0 bytes")
    index = root / "index.txt"
    require(index.stat().st_size <= 8 * 1024 * 1024, "oversized index.txt")
    names = set()
    coverage = set()
    lines = index.read_text(encoding="ascii").splitlines()
    header = re.compile(r"# align ([0-9]+) pool ([0-9]+) tensors ([0-9]+)")
    tensor_count = None
    seen_rows = False
    for line in lines:
        # Match the actual pack_index.py layout: a descriptive comment,
        # then '# align ...', then a column-label comment and tensor rows.
        # Strata's stage_weights.cpp also permits other leading comments.
        require(bool(line) and len(line) < 1023,
                "blank or overlong canonical pack index row")
        if line.startswith("#"):
            match = header.fullmatch(line)
            if match:
                require(tensor_count is None and not seen_rows,
                        "duplicate or misplaced canonical index header")
                alignment, pool_bytes, tensor_count = map(int, match.groups())
                require(0 < alignment <= 1 << 20 and pool_bytes > 0 and
                        tensor_count > 0, "invalid canonical index header values")
            elif line.startswith("# align "):
                raise ValueError("malformed canonical index header")
            continue
        require(tensor_count is not None,
                "canonical pack index has tensor rows before its header")
        seen_rows = True
        fields = line.split()
        require(len(fields) == 19 and fields[0] not in names,
                "index row must have 19 columns and unique tensor name")
        names.add(fields[0])
        match = re.match(r"^blk\.([0-9]+)\.", fields[0])
        if match:
            layer = int(match.group(1))
            require(layer < 48 and str(layer) == match.group(1), "noncanonical index layer")
            coverage.add(layer)
    require(tensor_count is not None and seen_rows,
            "missing canonical pack index header or tensor rows")
    require(len(names) == tensor_count and coverage == set(range(48)),
            "canonical pack index does not cover all 48 model layers")
    for _backend, _ordinal, first, end in stages:
        require(all(layer in coverage for layer in range(first, end)),
                "stage has an uncovered layer in canonical pack index")


def check_preflight(binary: Path, cfg: Path) -> None:
    require(binary.is_absolute() and binary.is_file() and os.access(binary, os.X_OK),
            "optional CPU-only preflight executable must be an executable absolute file")
    result = subprocess.run([str(binary), "--check-config", str(cfg)],
                            capture_output=True, text=True, timeout=60, check=False)
    require(result.returncode == 0, f"CPU file preflight refused {cfg.name}: {result.stderr.strip()}")


def validate(spec: str, layers: int, hc: int, n_embd: int, fingerprint: int,
             configs: list[Path], preflight: Path | None = None) -> None:
    stages = plan_stages(spec, layers)
    require((layers, hc, n_embd, 248320) == EXPECTED_GEOMETRY and fingerprint > 0,
            "only canonical 48-layer Q2_0 with nonzero fingerprint is supported")
    require(len(stages) == len(configs), "expected exactly one ordered config per stage")
    fields = []
    for i, (stage, path) in enumerate(zip(stages, configs)):
        if preflight is not None:
            check_preflight(preflight, path)
        c = config(path)
        require(c.get("version") == "2", "stage config must be version 2")
        backend, ordinal, first, end = stage
        for key, expected in (("stage_index", i), ("device_ordinal", ordinal),
                              ("first_layer", first), ("end_layer", end),
                              ("n_layers", layers), ("hc", hc), ("n_embd", n_embd),
                              ("n_vocab", 248320), ("model_fingerprint", fingerprint)):
            require(c.get(key) == str(expected), f"stage {i} {backend} {key} does not match plan")
        slots = c.get("max_cache_slots")
        require(slots is not None and re.fullmatch(r"[0-9]{1,5}", slots) is not None
                and 1 <= int(slots) <= (end - first) * 512,
                f"stage {i} expert cache slots exceed canonical stage bounds")
        if fields:
            for key in ("pack_dir", "profile_path", "index_sha256", "model_fingerprint",
                        "gguf_shards", "n_layers", "hc", "n_embd", "n_vocab"):
                require(c.get(key) == fields[0].get(key),
                        f"stage {i} model identity differs in {key}")
        fields.append(c)
    first = fields[0]
    shards = [Path(p) for p in first["gguf_shards"].split(";")]
    ple = check_gguf(shards)
    for i, ((_, _, begin, end), c) in enumerate(zip(stages, fields)):
        require((c.get("ple_gguf_path") == str(ple)) if begin <= 1 < end else
                "ple_gguf_path" not in c,
                f"stage {i} PLE path is not its original PLE-bearing shard")
    pack = Path(first["pack_dir"])
    check_pack(pack, stages)
    read_profile(Path(first["profile_path"]))
    exact = hashlib.sha256((pack / "index.txt").read_bytes()).hexdigest()
    require(first["index_sha256"] == exact,
            "Q2_0 index digest changed since config creation")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--hc", type=int, required=True)
    parser.add_argument("--n-embd", type=int, required=True)
    parser.add_argument("--fingerprint", type=int, required=True)
    parser.add_argument("--preflight", type=Path,
                        help="optional additional CPU-only native config preflight executable")
    parser.add_argument("--config", action="append", type=Path, required=True,
                        help="one config per ordered stage; repeat --config")
    args = parser.parse_args()
    try:
        validate(args.plan, args.layers, args.hc, args.n_embd,
                 args.fingerprint, args.config, args.preflight)
    except (ValueError, KeyError, OSError, RuntimeError, TimeoutError, UnicodeError) as exc:
        print(f"FAIL: Q2_0 model/plan gate: {exc}", file=sys.stderr)
        return 1
    print(f"PASS: canonical Q2_0 model/plan input gate ({len(args.config)} stages, CPU only)")
    print("NOTE: not a full-model checksum, native GPU validation or numerical parity")
    return 0


if __name__ == "__main__":
    sys.exit(main())
