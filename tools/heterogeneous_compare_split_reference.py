#!/usr/bin/env python3
"""Compare tagged one-row STFR captures from a split vanilla oracle and workers.

Capture producers, not this reader, must establish the source/build provenance.
This does not change the legacy unsplit parity contract.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys

from heterogeneous_compare_kv import metrics, require

HIP = "hip:0:0-24,hip:1:24-48"
MIXED = "cuda:0:0-15,hip:0:15-33,hip:1:33-48"


def compare(reference, candidate, max_kl=0.001):
    require(math.isfinite(max_kl) and 0 <= max_kl <= 0.001, "invalid or relaxed KL threshold")
    require(len(reference) == len(candidate) and 0 < len(reference) <= 128,
            "invalid capture lengths")
    first_ref, first_het = reference[0][0], candidate[0][0]
    require(first_ref.get("plan") == HIP and first_het.get("plan") in (HIP, MIXED),
            "expected split HIP oracle and canonical HIP or mixed candidate")
    mode = first_ref.get("kv_mode")
    require(mode in ("f16", "int8"), "explicit KV mode required")
    for run, producer in ((reference, "vanilla"), (candidate, "heterogeneous")):
        first = run[0][0]
        revision = first.get("source_revision")
        binary = first.get("binary_sha256")
        require(isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision) and
                isinstance(binary, str) and re.fullmatch(r"[0-9a-f]{64}", binary),
                "missing producer build identity")
        for i, (meta, packet) in enumerate(run):
            require(meta.get("capture_producer") == producer and meta.get("source_revision") == revision and
                    meta.get("binary_sha256") == binary and meta.get("kv_mode") == mode and
                    meta.get("plan") == first["plan"], "inconsistent producer, placement or KV mode")
            require(meta["window_id"] == i+1 and meta["pos0"] == i and
                    meta["rows"] == meta["n_keep"] == packet["rows"] == 1,
                    "expected one committed teacher-forced row per position")
    rows = []
    for (rm, rp), (hm, hp) in zip(reference, candidate):
        for key in ("model_fingerprint", "pack_index_sha256", "gguf_shards", "profile_path", "token_ids"):
            require(rm[key] == hm[key] == first_ref[key] if key != "token_ids" else rm[key] == hm[key],
                    f"source or input differs: {key}")
        require(rp["vocab"] == hp["vocab"] == len(rp["logits"]) == len(hp["logits"]),
                "different or incomplete logits shape")
        m = metrics(rp["logits"], hp["logits"])
        rows.append(dict(position=rm["pos0"], input=rm["token_ids"][0],
                         reference_top1=m["f16_top1"], candidate_top1=m["int8_top1"],
                         reference_prediction=rp["predictions"][0], candidate_prediction=hp["predictions"][0],
                         kl=m["kl_f16_to_int8"], max_abs=m["max_abs"], rmse=m["rmse"]))
    failures = [r for r in rows if r["reference_top1"] != r["candidate_top1"] or
                r["reference_prediction"] != r["candidate_prediction"] or r["kl"] > max_kl]
    return dict(status="FAIL" if failures else "PASS", kv_mode=mode,
                reference_plan=first_ref["plan"], candidate_plan=first_het["plan"],
                reference_revision=first_ref["source_revision"], candidate_revision=first_het["source_revision"],
                reference_binary_sha256=first_ref["binary_sha256"], candidate_binary_sha256=first_het["binary_sha256"],
                rows=len(rows), max_kl=max(r["kl"] for r in rows), kl_threshold=max_kl,
                first_failure=failures[0] if failures else None, row_metrics=rows)


def main():
    from heterogeneous_compare_parity import read_run
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vanilla-prefix", required=True, type=Path)
    p.add_argument("--candidate-prefix", required=True, type=Path)
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    try:
        require(args.vanilla_prefix.resolve() != args.candidate_prefix.resolve(), "distinct captures required")
        result = compare(read_run(args.vanilla_prefix, placement=True),
                         read_run(args.candidate_prefix, placement=True))
        encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.report:
            args.report.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
        return 0 if result["status"] == "PASS" else 1
    except (OSError, ValueError, KeyError, TypeError, OverflowError, MemoryError) as exc:
        print(f"Invalid split reference capture: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
