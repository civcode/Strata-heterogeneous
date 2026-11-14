#!/usr/bin/env python3
"""M2.7 passive Linux PCI/driver and stage-memory readiness inventory.

Never opens /dev nodes, invokes GPU utilities, loads models, or initializes a
CUDA/HIP runtime. Reads only bounded sysfs/procfs text, configured index.txt,
and optionally a pre-exported kernel log file. Does not write any files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from heterogeneous_model_gate import config, plan_stages

VENDOR = {0x10DE: "nvidia", 0x1002: "amd"}
DISPLAY_CLASSES = {0x0300, 0x0302, 0x0380}
EXPERT_BLOB_BYTES = 1_382_400  # canonical Q2_0; never applies to IQ3/native packs
EXPERT_RESIDENCY_BYTES = 48 * 512 * 4
MAX_INDEX_BYTES = 8 * 1024 * 1024
MAX_LOG_BYTES = 8 * 1024 * 1024
BDF_PATTERN = re.compile(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]")
HEADER = re.compile(r"# align ([0-9]+) pool ([0-9]+) tensors ([0-9]+)")
BLOCK = re.compile(r"blk\.([0-9]+)\.")
FAULT_PATTERNS = (
    re.compile(r"\bNVRM:.*\bXid\b", re.I),
    re.compile(r"\bGPU.*fallen off the bus\b", re.I),
    re.compile(r"\bGPU has fallen off the bus\b", re.I),
    re.compile(r"\bamdgpu.*\b(GPU fault|ring .*timeout|GPU reset)\b", re.I),
    re.compile(r"\bpcie.*\b(AER|fatal error)\b", re.I),
)


def need(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def read_text(path: Path, limit: int = 4096) -> str | None:
    """Bounded regular sysfs/procfs text read; missing/unreadable => unknown."""
    try:
        # sysfs attributes often report size=0; do not require nonzero st_size.
        with path.open("rb") as file:
            contents = file.read(limit + 1)
        if len(contents) > limit:
            return None
        return contents.decode("ascii", errors="replace").strip()
    except (OSError, ValueError):
        return None


def read_hex(path: Path) -> int | None:
    s = read_text(path, 64)
    if s is None or not re.fullmatch(r"0[xX][0-9a-fA-F]+", s):
        return None
    return int(s, 16)


def gpu_inventory(sys_root: Path) -> list[dict[str, Any]]:
    """Enumerate PCI graphics functions; no /dev or GPU driver API calls."""
    base = sys_root / "bus/pci/devices"
    if not base.is_dir():
        raise ValueError(f"PCI sysfs root unavailable: {base}")
    inventory: list[dict[str, Any]] = []
    for entry in sorted(base.iterdir()):
        if not BDF_PATTERN.fullmatch(entry.name):
            continue
        vendor, dev_id, klass = (read_hex(entry / key) for key in ("vendor", "device", "class"))
        if vendor not in VENDOR or klass is None or (klass >> 8) not in DISPLAY_CLASSES:
            continue
        driver = None
        try:
            driver = (entry / "driver").resolve(strict=True).name
        except (OSError, RuntimeError):
            pass
        cards = sorted(c.name for c in (entry / "drm").iterdir()
                       if re.fullmatch(r"card[0-9]+", c.name)) if (entry / "drm").is_dir() else []
        inventory.append({
            "pci_bdf": entry.name.lower(), "vendor": VENDOR[vendor],
            "pci_vendor_id": f"0x{vendor:04x}",
            "pci_device_id": f"0x{dev_id:04x}" if dev_id is not None else None,
            "pci_class": f"0x{klass:06x}",
            "bound_driver": driver, "drm_cards": cards,
            "numa_node": read_text(entry / "numa_node", 16),
        })
    return inventory


def driver_inventory(sys_root: Path, proc_root: Path) -> dict[str, Any]:
    module_text = read_text(proc_root / "modules", 1024 * 1024)
    present = set()
    if module_text is not None:
        for line in module_text.splitlines():
            name = line.split(maxsplit=1)[0] if line.strip() else ""
            if name in {"nvidia", "nvidia_uvm", "nvidia_modeset", "amdgpu", "kfd"}:
                present.add(name)
    return {
        "loaded_modules": sorted(present) if module_text is not None else None,
        "nvidia_module_version": read_text(sys_root / "module/nvidia/version", 256),
        "amdgpu_module_version": read_text(sys_root / "module/amdgpu/version", 256),
        "nvidia_proc_version": read_text(proc_root / "driver/nvidia/version", 1024),
        "visibility_is_not_runtime_health": True,
    }


def planned_weights(index_path: Path, stages: list[tuple[str, int, int, int]]) -> list[int]:
    """Reproduce stage_weights.cpp selected packed arena accounting (CPU)."""
    need(index_path.is_file() and index_path.stat().st_size <= MAX_INDEX_BYTES,
         "missing or oversized pack index")
    lines = index_path.read_text(encoding="ascii").splitlines()
    alignment = pool = declared = None
    seen_rows = False
    names = set()
    covered = set()
    totals = [0 for _ in stages]
    last_end = 0
    for line in lines:
        need(line and len(line) < 1023, "blank or overlong index row")
        if line.startswith("#"):
            m = HEADER.fullmatch(line)
            if m:
                need(declared is None and not seen_rows, "duplicate or misplaced pack header")
                alignment, pool, declared = map(int, m.groups())
                need(0 < alignment <= 1 << 20 and 0 < pool and declared > 0,
                     "invalid pack header values")
            elif line.startswith("# align "):
                raise ValueError("malformed pack header")
            continue
        need(declared is not None, "tensor before pack index header")
        seen_rows = True
        f = line.split()
        need(len(f) == 19 and f[0] not in names, "invalid or duplicate pack index tensor")
        names.add(f[0])
        try:
            file_id, kind = int(f[1]), int(f[2])
            src_off, src_bytes, dst_off, dst_bytes = map(int, f[3:7])
            dims = list(map(int, f[7:9]))
            _ = list(map(int, f[9:19]))
        except ValueError as exc:
            raise ValueError("non-integer pack index field") from exc
        need(0 <= file_id <= 3 and 0 <= kind <= 5 and
             min(src_off, src_bytes, dst_off, dst_bytes) >= 0 and
             dims[0] > 0 and dims[1] >= 0 and
             dst_off % alignment == 0 and dst_off >= last_end and
             dst_off + dst_bytes <= pool, "invalid pack row bounds")
        last_end = dst_off + dst_bytes
        m = BLOCK.match(f[0])
        layer = None
        if m:
            layer = int(m.group(1))
            need(m.group(1) == str(layer) and 0 <= layer < stages[-1][3],
                 "noncanonical pack layer")
            covered.add(layer)
        for i, (_backend, _ord, first, end) in enumerate(stages):
            if layer is None or first <= layer < end:
                need(dst_bytes > 0, "selected tensor has no destination bytes")
                totals[i] += ((dst_bytes + alignment - 1) // alignment) * alignment
    need(declared is not None and seen_rows and len(names) == declared,
         "pack tensor count/header mismatch")
    need(covered == set(range(stages[-1][3])), "pack index does not cover every layer")
    need(all(0 < n <= pool for n in totals), "invalid selected pack memory")
    return totals


def planned_memory(spec: str, cfg_paths: list[Path], layers: int) -> list[dict[str, Any]]:
    stages = plan_stages(spec, layers)
    need(len(stages) == len(cfg_paths), "stage config count does not match plan")
    configs = [config(p) for p in cfg_paths]
    for key in ("pack_dir", "model_fingerprint", "index_sha256", "gguf_shards",
                "profile_path", "hc", "n_embd", "n_vocab"):
        need(all(c.get(key) and c.get(key) == configs[0].get(key)
                 for c in configs), f"inconsistent cross-stage {key}")
    pack_paths = {c.get("pack_dir") for c in configs}
    need(len(pack_paths) == 1 and None not in pack_paths, "different or missing stage pack_dir")
    pack = Path(next(iter(pack_paths)))
    need((pack / "experts.bin").is_file() and not (pack / "native_experts.txt").exists() and
         not (pack / "native_experts.txt").is_symlink(),
         "M2.7 memory estimate requires canonical Q2_0 experts.bin, not a native pack")
    for idx, (stage, c) in enumerate(zip(stages, configs)):
        backend, ordinal, first, end = stage
        for key, expected in (("version", 2), ("stage_index", idx),
                              ("device_ordinal", ordinal), ("first_layer", first),
                              ("end_layer", end), ("n_layers", layers)):
            need(c.get(key) == str(expected), f"stage {idx} {key} disagrees with plan")
    index = pack / "index.txt"
    need(index.is_file() and index.stat().st_size <= MAX_INDEX_BYTES,
         "pack index missing or oversized")
    need(hashlib.sha256(index.read_bytes()).hexdigest() == configs[0]["index_sha256"],
         "pack index SHA-256 disagrees with stage configs")
    weights = planned_weights(index, stages)
    out = []
    for i, ((backend, ordinal, first, end), c) in enumerate(zip(stages, configs)):
        slots = int(c["max_cache_slots"])
        reserve = int(c["free_reserve_bytes"])
        budget = int(c["arena_budget_bytes"])
        max_cells = int(c["max_cells"])
        max_t = int(c["max_t"])
        need(max_cells > 0 and max_t > 0, "invalid stage session/window limits")
        need(1 <= slots <= (end - first) * 512 and 0 <= reserve < (1 << 64)
             and 0 <= budget < (1 << 64),
             "invalid configured stage slot count/reserve/budget")
        cache = slots * EXPERT_BLOB_BYTES + EXPERT_RESIDENCY_BYTES
        lower_bound = weights[i] + cache
        out.append({
            "stage": i, "backend": backend, "ordinal": ordinal,
            "first_layer": first, "end_layer": end,
            "selected_pack_weight_bytes": weights[i],
            "configured_expert_slots": slots,
            "max_cells": max_cells, "max_t": max_t,
            "canonical_expert_cache_plus_table_bytes": cache,
            "known_allocations_floor_bytes": lower_bound,
            "safety_reserve_bytes": reserve,
            "known_allocations_plus_reserve_bytes": lower_bound + reserve,
            "configured_weight_session_arena_budget_bytes": budget if budget else None,
            "unaccounted": ["stage session arena (core::session_bytes)",
                            "native GGUF dense/PLE allocations", "verifier staging and streams",
                            "runtime allocator overhead and any host/device staging"],
        })
    return out


def validate_bindings(tokens: list[str], stages: list[tuple[str, int, int, int]],
                      pci: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    expected = {(kind, idx) for kind, idx, _, _ in stages}
    found = {d["pci_bdf"]: d for d in pci}
    supplied = {}
    errors = []
    for token in tokens:
        m = re.fullmatch(r"(cuda|hip):([0-9]+)=([0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7])", token)
        if not m:
            errors.append(f"invalid --bind syntax: {token!r}")
            continue
        key = (m.group(1), int(m.group(2)))
        bdf = m.group(3).lower()
        if key not in expected or key in supplied:
            errors.append(f"unexpected/duplicate backend:ordinal binding: {key}")
            continue
        supplied[key] = bdf
    reverse = set()
    result = []
    for backend, ordinal, _, _ in stages:
        key = (backend, ordinal)
        bdf = supplied.get(key)
        device = found.get(bdf)
        valid = bdf is not None and device is not None
        if bdf is not None:
            if bdf in reverse:
                errors.append(f"PCI function assigned to multiple stages: {bdf}")
            reverse.add(bdf)
            if device is None:
                errors.append(f"explicit mapping for {backend}:{ordinal} is not in sysfs: {bdf}")
            else:
                expected_vendor = "nvidia" if backend == "cuda" else "amd"
                if device["vendor"] != expected_vendor:
                    errors.append(f"vendor mismatch: {backend}:{ordinal} at {bdf}")
                if device["bound_driver"] not in ({"nvidia"} if backend == "cuda" else {"amdgpu"}):
                    errors.append(f"driver not bound as expected: {backend}:{ordinal} at {bdf}")
        result.append({"backend": backend, "ordinal": ordinal, "declared_bdf": bdf,
                       "pci_device_found": valid,
                       "runtime_ordinal_confirmed": False})
    return result, errors


def scan_preexported_log(file: Path | None) -> dict[str, Any]:
    if file is None:
        return {"source": None, "fault_markers": None, "interpretation": "not evaluated"}
    need(file.is_file() and file.stat().st_size <= MAX_LOG_BYTES,
         "kernel log input missing or larger than 8 MiB")
    markers: list[str] = []
    # Show only a redacted prefix; exact PCI IDs and timestamps may be useful,
    # but kernel logs can contain unrelated private information.
    with file.open("r", encoding="utf-8", errors="replace") as source:
        for line in source:
            if any(p.search(line) for p in FAULT_PATTERNS):
                markers.append(line.strip()[:240])
                if len(markers) == 20:
                    break
    return {"source": str(file), "fault_markers": markers,
            "interpretation": "historical signals only; absence is not a health test"}


def assess(plan: str, layers: int, cfg: list[Path], sys_root: Path, proc_root: Path,
           bindings: list[str], kernel_log: Path | None) -> dict[str, Any]:
    stages = plan_stages(plan, layers)
    gpu = gpu_inventory(sys_root)
    mapping, problems = validate_bindings(bindings, stages, gpu)
    mem = planned_memory(plan, cfg, layers)
    # Runtime ordinals are backend-local handles used directly by the native
    # workers. Physical PCI identity is useful for diagnostics/pinning, but
    # the absence of an optional --bind declaration is not a failed passive
    # hardware assessment (nor a prerequisite for starting a staged test).
    nvidia_count = sum(d["vendor"] == "nvidia" for d in gpu)
    amd_count = sum(d["vendor"] == "amd" for d in gpu)
    if nvidia_count < len({i for b, i, _, _ in stages if b == "cuda"}):
        problems.append("fewer NVIDIA graphics functions than planned CUDA ordinals")
    if amd_count < len({i for b, i, _, _ in stages if b == "hip"}):
        problems.append("fewer AMD graphics functions than planned HIP ordinals")
    return {"schema": "strata-heterogeneous-m2.7-readiness-v1", "passive_only": True,
            "runtime_initialized": False, "plan": plan, "stage_memory": mem,
            "pci_graphics": gpu, "driver_metadata": driver_inventory(sys_root, proc_root),
            "ordinal_declarations": mapping, "offline_kernel_log": scan_preexported_log(kernel_log),
            "readiness": "UNVERIFIED" if not problems else "BLOCKED",
            "blockers": problems,
            "limits": ["No CUDA/HIP runtime ordinal enumeration or memory availability measurement",
                       "No GPU health, IOMMU topology, peer access or link bandwidth measurement",
                       "Memory excludes session, native dense/PLE, dynamic and allocator buffers",
                       "Explicit --bind declares intended mapping, NOT a runtime ordinal guarantee",
                       "Kernel log input is optional and historical; no journal/dmesg invoked"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--config", action="append", type=Path, required=True)
    parser.add_argument("--sysfs-root", type=Path, default=Path("/sys"), help=argparse.SUPPRESS)
    parser.add_argument("--procfs-root", type=Path, default=Path("/proc"), help=argparse.SUPPRESS)
    parser.add_argument("--bind", action="append", default=[],
                        help="declared mapping, e.g. cuda:0=0000:01:00.0 (NOT runtime-confirmed)")
    parser.add_argument("--kernel-log-file", type=Path,
                        help="optional PRE-EXPORTED kernel log; never reads dmesg/journal")
    parser.add_argument("--json", action="store_true", help="emit structured JSON to stdout")
    args = parser.parse_args(argv)
    try:
        result = assess(args.plan, args.layers, args.config, args.sysfs_root,
                        args.procfs_root, args.bind, args.kernel_log_file)
    except (OSError, ValueError, KeyError, UnicodeError, OverflowError) as exc:
        print(f"FAIL: M2.7 passive hardware readiness: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("M2.7 passive PCI / driver assessment (no device nodes opened)")
        for d in result["pci_graphics"]:
            print(f"  PCI {d['pci_bdf']} {d['vendor']} device={d['pci_device_id']} "
                  f"driver={d['bound_driver'] or 'unbound/unknown'} "
                  f"drm={','.join(d['drm_cards']) or '-'}")
        drivers = result["driver_metadata"]
        print("  Kernel modules: " +
              (", ".join(drivers["loaded_modules"]) if drivers["loaded_modules"] is not None else "unknown"))
        print("  NVIDIA module version: " +
              (drivers["nvidia_module_version"] or "unknown"))
        for s in result["stage_memory"]:
            mib = 1024 * 1024
            print(f"  Stage {s['stage']} {s['backend']}:{s['ordinal']} "
                  f"[{s['first_layer']},{s['end_layer']}): "
                  f"weights={s['selected_pack_weight_bytes']/mib:.1f}MiB "
                  f"Q2cache={s['canonical_expert_cache_plus_table_bytes']/mib:.1f}MiB "
                  f"reserve={s['safety_reserve_bytes']/mib:.1f}MiB "
                  f"max_cells={s['max_cells']} max_t={s['max_t']} "
                  "(session/native/runtime additional, unknown)")
        for s in result["ordinal_declarations"]:
            print(f"  Declared mapping {s['backend']}:{s['ordinal']} "
                  f"-> {s['declared_bdf'] or 'UNRESOLVED'} "
                  "(runtime ordinal NOT checked)")
        for problem in result["blockers"]:
            print(f"  BLOCKER: {problem}")
        log = result["offline_kernel_log"]
        print("  Kernel fault evidence: " +
              ("not inspected" if log["fault_markers"] is None else
               f"{len(log['fault_markers'])} matching lines in supplied offline log"))
        print("READINESS: " + result["readiness"] +
              " (NEVER authorizes live GPU use or establishes driver health)")
    # Exit 0 means this *passive inventory* had no contradictions. Runtime
    # device discovery, free-memory checks and inference are separate gates.
    return 0 if not result["blockers"] else 1


if __name__ == "__main__":
    sys.exit(main())
