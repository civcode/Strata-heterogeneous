#!/usr/bin/env python3
"""M2.8 offline CUDA/HIP ordinal-to-PCI evidence reconciliation.

Consumes already-exported JSON only. No sysfs, device nodes, CUDA/HIP runtime,
GPU utilities, subprocesses, or model assets are accessed by this program.
Evidence files are NOT cryptographic attestations or proof of device health.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

INVENTORY_SCHEMA = "strata-heterogeneous-m2.7-readiness-v1"
PROBE_SCHEMA = "strata-heterogeneous-runtime-identity-v1"
BDF = re.compile(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]")
MAX_JSON = 1 << 20


def require(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError(why)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and path.stat().st_size <= MAX_JSON,
            f"missing/oversized JSON evidence: {path}")
    with path.open("rb") as source:
        data = source.read(MAX_JSON + 1)
    require(len(data) <= MAX_JSON, f"oversized JSON evidence: {path}")
    root = json.loads(data)
    require(type(root) is dict, "evidence root must be a JSON object")
    return root


def normalize_bdf(value: Any) -> str:
    require(type(value) is str and BDF.fullmatch(value.lower()) is not None,
            "invalid PCI BDF in evidence")
    return value.lower()


def parse_inventory(report: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    require(report.get("schema") == INVENTORY_SCHEMA and report.get("passive_only") is True
            and report.get("runtime_initialized") is False,
            "expected a passive M2.7 inventory (never a runtime claim)")
    pci = report.get("pci_graphics")
    stages = report.get("stage_memory")
    require(type(pci) is list and type(stages) is list and bool(stages),
            "missing M2.7 PCI inventory or stage plan")
    devices: dict[str, dict[str, Any]] = {}
    for device in pci:
        require(type(device) is dict, "invalid PCI inventory entry")
        bdf = normalize_bdf(device.get("pci_bdf"))
        require(bdf not in devices and device.get("vendor") in ("nvidia", "amd"),
                "duplicate PCI BDF or invalid vendor")
        devices[bdf] = device
    expected_stages = []
    stage_keys = set()
    for idx, stage in enumerate(stages):
        require(type(stage) is dict and type(stage.get("stage")) is int
                and stage["stage"] == idx and stage.get("backend") in ("cuda", "hip")
                and type(stage.get("ordinal")) is int and stage["ordinal"] >= 0,
                "malformed/unsorted M2.7 stage record")
        key = (stage["backend"], stage["ordinal"])
        require(key not in stage_keys, "duplicate backend:ordinal in M2.7 stage plan")
        stage_keys.add(key)
        expected_stages.append({"stage": idx, "backend": key[0], "ordinal": key[1]})
    return devices, expected_stages


def parse_probe(probe: dict[str, Any]) -> tuple[str, dict[int, str]]:
    require(probe.get("schema") == PROBE_SCHEMA and
            probe.get("backend") in ("cuda", "hip") and
            probe.get("method") == "runtime_device_get_pci_bus_id" and
            probe.get("complete") is True,
            "unsupported or incomplete runtime identity evidence")
    backend = probe["backend"]
    records = probe.get("devices")
    require(type(records) is list and len(records) <= 64 and bool(records),
            "identity evidence must include a bounded nonempty device list")
    mapping: dict[int, str] = {}
    seen = set()
    for record in records:
        require(type(record) is dict and type(record.get("ordinal")) is int and
                record["ordinal"] >= 0 and record["ordinal"] < 64,
                "invalid runtime ordinal record")
        ordinal = record["ordinal"]
        bdf = normalize_bdf(record.get("pci_bdf"))
        require(ordinal not in mapping and bdf not in seen,
                "repeated runtime ordinal or PCI function in probe")
        mapping[ordinal] = bdf
        seen.add(bdf)
    require(set(mapping) == set(range(len(mapping))),
            "probe must enumerate contiguous runtime ordinals starting at zero")
    return backend, mapping


def reconcile(inventory: dict[str, Any], probes: list[dict[str, Any]]) -> dict[str, Any]:
    """Pure evidence validation; even full agreement is not a runtime health gate."""
    pci, stages = parse_inventory(inventory)
    mappings: dict[str, dict[int, str]] = {}
    for probe in probes:
        backend, mapping = parse_probe(probe)
        require(backend not in mappings, f"duplicate identity probe for {backend}")
        mappings[backend] = mapping
    problems: list[str] = []
    # An identity cross-check must not erase an unrelated passive M2.7 blocker.
    # The single ordinal-missing marker can be addressed by complete probe
    # evidence; all other M2.7 blockers survive this stage unchanged.
    prior = inventory.get("blockers", [])
    require(type(prior) is list and all(type(x) is str for x in prior),
            "malformed M2.7 blocker list")
    for reason in prior:
        if reason != "unresolved backend-local runtime ordinal mapping":
            problems.append("M2.7 retained blocker: " + reason)
    declarations = inventory.get("ordinal_declarations", [])
    require(type(declarations) is list, "malformed M2.7 ordinal declarations")
    declared = {}
    for d in declarations:
        require(type(d) is dict and d.get("backend") in ("cuda", "hip")
                and type(d.get("ordinal")) is int, "malformed M2.7 ordinal declaration")
        key = (d["backend"], d["ordinal"])
        require(key not in declared, "duplicate M2.7 ordinal declaration")
        declared[key] = (normalize_bdf(d["declared_bdf"])
                         if d.get("declared_bdf") is not None else None)
    resolved: list[dict[str, Any]] = []
    seen_bdf = set()
    for stage in stages:
        backend, ordinal = stage["backend"], stage["ordinal"]
        mapping = mappings.get(backend)
        bdf = mapping.get(ordinal) if mapping is not None else None
        entry = pci.get(bdf) if bdf is not None else None
        required_vendor = "nvidia" if backend == "cuda" else "amd"
        required_driver = "nvidia" if backend == "cuda" else "amdgpu"
        if mapping is None:
            problems.append(f"missing {backend} runtime identity evidence")
        elif bdf is None:
            problems.append(f"{backend}:{ordinal} absent from runtime identity evidence")
        elif entry is None:
            problems.append(f"{backend}:{ordinal} maps to PCI {bdf}, absent from M2.7 inventory")
        else:
            if entry.get("vendor") != required_vendor:
                problems.append(f"{backend}:{ordinal} PCI vendor mismatch at {bdf}")
            if entry.get("bound_driver") != required_driver:
                problems.append(f"{backend}:{ordinal} bound-driver mismatch at {bdf}")
        recorded = declared.get((backend, ordinal))
        if recorded is not None and bdf is not None and recorded != bdf:
            problems.append(f"{backend}:{ordinal} runtime PCI {bdf} disagrees with M2.7 declaration {recorded}")
        if bdf is not None:
            if bdf in seen_bdf:
                problems.append(f"PCI {bdf} claimed by multiple runtime stages")
            seen_bdf.add(bdf)
        resolved.append({**stage, "runtime_reported_bdf": bdf,
                         "passive_pci_matched": entry is not None and
                         entry.get("vendor") == required_vendor and
                         entry.get("bound_driver") == required_driver})
    return {
        "schema": "strata-heterogeneous-m2.8-identity-reconciliation-v1",
        "runtime_called_by_this_tool": False,
        "evidence_is_authenticated": False,
        "readiness": "BLOCKED" if problems else "IDENTITY_CORRELATED_ONLY",
        "stage_mappings": resolved,
        "blockers": list(dict.fromkeys(problems)),
        "remaining_gates": [
            "Verify evidence was collected on the intended host, boot and environment",
            "Assess NVIDIA/AMD driver faults separately; absence of logs is not proof of health",
            "Establish free VRAM and complete session/native/PLE/verifier/runtime budget",
            "Obtain separate explicit authorization before any live GPU execution",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", type=Path, required=True,
                   help="pre-exported M2.7 --json output")
    p.add_argument("--probe", type=Path, action="append", default=[],
                   help="offline evidence from a separately authorized runtime-identity probe")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    try:
        output = reconcile(load_json(args.inventory), [load_json(path) for path in args.probe])
    except (OSError, ValueError, TypeError, UnicodeError, json.JSONDecodeError) as error:
        print(f"FAIL: M2.8 offline identity reconciliation: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(output, sort_keys=True, indent=2))
    else:
        for stage in output["stage_mappings"]:
            print(f"  Stage {stage['stage']} {stage['backend']}:{stage['ordinal']} "
                  f"-> {stage['runtime_reported_bdf'] or 'UNRESOLVED'} "
                  f"(passive PCI match={stage['passive_pci_matched']})")
        for blocker in output["blockers"]:
            print("  BLOCKER: " + blocker)
        print("READINESS: " + output["readiness"] +
              " (offline evidence only; NEVER authorizes GPU use)")
    return 1 if output["blockers"] else 0


if __name__ == "__main__":
    sys.exit(main())
