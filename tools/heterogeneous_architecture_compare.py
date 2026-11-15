#!/usr/bin/env python3
"""Compare complete, matched workload JSONL runs; CPU only, no inference.

Manifest identity is checked; actual model/backend/options must be established
from launch evidence. Ratios are observational, never causal attribution.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median

SCHEMA = "strata-heterogeneous-workloads/v1"
METRICS = ("ttft_s", "elapsed_s", "e2e_completion_tps", "server_prefill_tps", "server_decode_tps")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def count(value):
    return type(value) is int and value >= 0


def load_run(path, manifest_hash, case_ids):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    require(len(rows) >= 3, "incomplete run")
    head, tail = rows[0], rows[-1]
    require(head.get("kind") == "header" and head.get("schema") == SCHEMA and
            head.get("manifest_sha256") == manifest_hash and tail.get("kind") == "summary",
            "missing completion summary or mismatched manifest/schema")
    repeats = head.get("repeats")
    require(type(repeats) is int and 1 <= repeats <= 20, "invalid repetition count")
    records = {}
    for row in rows[1:-1]:
        key = (row.get("id"), row.get("repeat"))
        require(row.get("kind") == "case" and key[0] in case_ids and
                type(key[1]) is int and 0 <= key[1] < repeats and key not in records and
                row.get("quality") in ("PASS", "FAIL", "UNSCORED") and
                isinstance(row.get("stages"), list) and row["stages"] and
                all(isinstance(stage, dict) for stage in row["stages"]), "invalid or duplicate attempt")
        records[key] = row
    require(set(records) == {(c, r) for c in case_ids for r in range(repeats)}, "missing attempts")
    return head, records


def sample(row, min_decode):
    # Only the first HTTP stage has the same input in every arm. Tool follow-up
    # input depends on generated calls; its quality is retained, not its timing.
    s = row["stages"][0]
    if s.get("status") != "COMPLETE":
        return None
    u, t = s.get("usage") or {}, s.get("server_timings") or {}
    prompt, completion = u.get("prompt_tokens"), u.get("completion_tokens")
    if not count(prompt) or not prompt or not count(completion):
        return None
    elapsed = s.get("elapsed_s")
    source = t.get("timing_source") or "unlabeled_server"
    uncached = t.get("cache_n") == 0 and t.get("prompt_n") == prompt
    return dict(prompt=prompt, completion=completion, finish=s.get("finish_reason"),
                timing_source=source, uncached=uncached,
                ttft_s=s.get("ttft_s") if positive(s.get("ttft_s")) else None,
                elapsed_s=elapsed if positive(elapsed) else None,
                e2e_completion_tps=completion / elapsed if positive(elapsed) and completion else None,
                server_prefill_tps=t.get("prompt_per_second") if uncached and positive(t.get("prompt_per_second")) else None,
                server_decode_tps=t.get("predicted_per_second") if completion >= min_decode and
                positive(t.get("predicted_per_second")) else None)


def stats(values):
    return dict(n=len(values), median=median(values) if values else None,
                minimum=min(values) if values else None, maximum=max(values) if values else None)


def comparison(manifest_path, arms, reference, warmup=1, min_decode=128):
    raw = Path(manifest_path).read_bytes()
    doc = json.loads(raw)
    require(doc.get("schema") == SCHEMA and isinstance(doc.get("cases"), list), "invalid manifest")
    cases = [c["id"] for c in doc["cases"]]
    require(cases and len(set(cases)) == len(cases), "manifest case IDs must be unique")
    require(reference in arms and len(arms) >= 2, "need reference and candidate arms")
    require(type(warmup) is int and warmup >= 0 and type(min_decode) is int and min_decode >= 1,
            "invalid analysis bounds")
    digest = hashlib.sha256(raw).hexdigest()
    runs = {name: load_run(path, digest, cases) for name, path in arms.items()}
    base_head, base = runs[reference]
    repeats = base_head["repeats"]
    require(warmup < repeats, "warmup excludes all samples")
    for h, _ in runs.values():
        require(h.get("model") == base_head.get("model") and h["repeats"] == repeats,
                "model label/repetition mismatch; use same served model name")
    reports = {}
    for name, (head, rows) in runs.items():
        report = {}
        for cid in cases:
            measured = [rows[cid, r] for r in range(warmup, repeats)]
            pairs = [(sample(base[cid, r], min_decode), sample(rows[cid,r], min_decode))
                     for r in range(warmup, repeats)]
            current = [b for _, b in pairs if b is not None]
            comparable = [(a,b) for a,b in pairs if a and b and a["prompt"] == b["prompt"]]
            result = dict(attempts=len(measured), quality={q:sum(x["quality"]==q for x in measured)
                          for q in ("PASS", "FAIL", "UNSCORED")},
                          first_statuses={k:sum(x["stages"][0].get("status")==k for x in measured)
                          for k in sorted({x["stages"][0].get("status", "MISSING") for x in measured})},
                          matched_prompt_pairs=len(comparable),
                          output_tokens=stats([b["completion"] for b in current]),
                          timing_sources=sorted({b["timing_source"] for b in current}),
                          metrics={}, paired_speedup={})
            for metric in METRICS:
                result["metrics"][metric] = stats([b[metric] for b in current if positive(b[metric])])
                ratios = []
                for a,b in comparable:
                    if not positive(a[metric]) or not positive(b[metric]):
                        continue
                    if metric.startswith("server_") and (a["timing_source"] != b["timing_source"] or
                            a["timing_source"] == "unlabeled_server"):
                        continue  # different or undocumented server timing denominators
                    if metric in ("elapsed_s", "e2e_completion_tps", "server_decode_tps") and (
                            a["completion"] != b["completion"] or a["finish"] != b["finish"]):
                        continue  # different amount of generated work / terminal reason
                    ratios.append(a[metric]/b[metric] if metric.endswith("_s") and not metric.endswith("_tps") else b[metric]/a[metric])
                result["paired_speedup"][metric] = stats(ratios)
            report[cid] = result
        reports[name] = report
    return dict(schema="strata-architecture-comparison/v1", manifest_sha256=digest,
                reference=reference, discarded_warmup_per_case=warmup, min_decode_tokens=min_decode,
                evidence="matched manifest and first-stage prompt token counts only; verify model, token IDs, settings and timing sources separately",
                qualification="DIAGNOSTIC_ONLY", arms=reports)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--arm", action="append", required=True, metavar="NAME=JSONL")
    p.add_argument("--reference", required=True)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--min-decode-tokens", type=int, default=128)
    p.add_argument("--output", type=Path)
    a = p.parse_args(argv)
    try:
        arms = {}
        for pair in a.arm:
            name, sep, path = pair.partition("=")
            require(sep and name and path and name not in arms, "invalid or duplicate arm")
            arms[name] = path
        report = comparison(a.manifest, arms, a.reference, a.warmup, a.min_decode_tokens)
        rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if a.output:
            with a.output.open("x", encoding="utf-8") as f:
                f.write(rendered)
        else:
            print(rendered, end="")
    except (OSError, ValueError, KeyError, TypeError) as e:
        p.error(str(e))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
