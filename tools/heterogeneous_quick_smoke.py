#!/usr/bin/env python3
"""Small, opt-in HTTP smoke against an already-running local Strata server.

The wall budget is an admission budget, not a hard process deadline:
an in-flight HTTP operation may take longer. Never launch or stop a server.
"""
import argparse
import json
import os
from pathlib import Path
import time


def run_suite(cases, base, model, key, wall_budget, per_request, emit,
              case_runner, sender, clock=time.monotonic):
    if not 0 < wall_budget <= 300 or not 0 < per_request <= 60:
        raise ValueError("wall budget must be <=300s and per-request timeout <=60s")
    if not 1 <= len(cases) <= 8 or any(type(c["max_tokens"]) is not int
                                       or not 1 <= c["max_tokens"] <= 64 for c in cases):
        raise ValueError("quick smoke permits 1..8 cases and 1..64 output tokens per request")
    deadline = clock() + wall_budget
    rows = []

    def bounded_send(url, body, auth, timeout, cancel_after_first=False):
        remaining = deadline - clock()
        if remaining <= 0:
            return {"status": "BUDGET_EXHAUSTED", "elapsed_s": 0.0}
        return sender(url, body, auth, min(timeout, per_request, remaining), cancel_after_first)

    for case in cases:
        if clock() >= deadline:
            break
        result = case_runner(base, model, case, key, per_request, sender=bounded_send)
        stages = result["stages"]
        expected = "CLIENT_CANCELLED" if case.get("cancel") else "COMPLETE"
        transport_ok = bool(stages) and all(s["status"] == expected for s in stages)
        row = {"kind": "case", "id": case["id"], "transport_ok": transport_ok, **result}
        emit(row)
        rows.append(row)
        if not transport_ok:
            break
    summary = {"kind": "summary", "planned": len(cases), "attempted": len(rows),
               "not_attempted": [c["id"] for c in cases[len(rows):]],
               "transport_failed": sum(not r["transport_ok"] for r in rows),
               "quality_failed": sum(r["quality"] == "FAIL" for r in rows),
               "quality_passed": sum(r["quality"] == "PASS" for r in rows),
               "unscored": sum(r["quality"] == "UNSCORED" for r in rows),
               "production_admission": "NOT_QUALIFIED"}
    emit(summary)
    return summary


def main():
    from tools.heterogeneous_workload_bench import endpoint, request, run_case, validate_manifest
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8081")
    p.add_argument("--model", default="qwen3.8-flash-next-q2_0")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--wall-budget", type=float, default=90)
    p.add_argument("--timeout", type=float, default=20)
    a = p.parse_args()
    endpoint(a.url)
    manifest = Path(__file__).resolve().parents[1] / "experimental/heterogeneous/workloads/quick-smoke.json"
    doc = json.loads(manifest.read_text())
    validate_manifest(doc)
    os.umask(0o077)
    with a.output.open("x", encoding="utf-8") as f:
        def emit(row):
            f.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")
            f.flush()
        emit({"kind": "header", "schema": "strata-heterogeneous-quick-smoke/v1",
              "model": a.model, "wall_budget_s": a.wall_budget, "request_timeout_s": a.timeout})
        summary = run_suite(doc["cases"], a.url, a.model, os.environ.get("STRATA_KEY", ""),
                            a.wall_budget, a.timeout, emit, run_case, request)
    print(json.dumps(summary, indent=2))
    return int(bool(summary["not_attempted"] or summary["transport_failed"] or summary["quality_failed"]))


if __name__ == "__main__":
    raise SystemExit(main())
