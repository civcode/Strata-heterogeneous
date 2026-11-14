"""Join real committed coordinator/worker spans; wall time, not GPU occupancy."""
import argparse
import json
from collections import defaultdict
from statistics import median


def fields(line, prefix):
    result = {}
    for item in line[len(prefix):].split():
        key, sep, value = item.partition("=")
        if not sep or not key or not value or key in result:
            raise ValueError(f"malformed {prefix.strip()} record")
        result[key] = value
    return result


def integer(record, key, minimum=0):
    value = int(record[key])
    if value < minimum:
        raise ValueError(f"invalid {key}")
    return value


def union(intervals):
    total = 0
    end = None
    for a, b in sorted(intervals):
        if end is None or a > end:
            total += b - a
            end = b
        elif b > end:
            total += b - end
            end = b
    return total


def distribution(values):
    values = sorted(v / 1e6 for v in values)
    if not values:
        return {"n": 0, "p50_ms": None, "p90_ms": None, "p99_ms": None}
    def quantile(p):
        i = (len(values) - 1) * p
        lo = int(i)
        return values[lo] + (i - lo) * (values[min(lo + 1, len(values) - 1)] - values[lo])
    return {"n": len(values), "p50_ms": median(values), "p90_ms": quantile(.9),
            "p99_ms": quantile(.99)}


def summarize(lines, stage_count):
    if not 1 <= stage_count <= 64:
        raise ValueError("invalid stage count")
    spans, commits = defaultdict(dict), {}
    for line in lines:
        line = line.strip()
        if line.startswith("HETERO_PERF "):
            r = fields(line, "HETERO_PERF ")
            key = integer(r, "request", 1), integer(r, "window")
            tag = r["component"], integer(r, "stage"), r["phase"]
            n, a, b = (integer(r, k) for k in ("rows", "start_ns", "end_ns"))
            if not key[1]:  # reset, not a generation window
                continue
            if not n or b < a or tag in spans[key]:
                raise ValueError(f"invalid or duplicate span {key} {tag}")
            spans[key][tag] = (a, b, n)
        elif line.startswith("HETERO_PERF_COMMITTED "):
            r = fields(line, "HETERO_PERF_COMMITTED ")
            key = integer(r, "request", 1), integer(r, "window", 1)
            phase = r["phase"]
            rows, keep, pos = (integer(r, k) for k in ("rows", "keep", "pos0"))
            if key in commits or phase not in ("prefill", "decode") or not 0 < keep <= rows:
                raise ValueError(f"invalid or duplicate committed window {key}")
            if phase == "prefill" and keep != rows:
                raise ValueError("partial prefill commit")
            commits[key] = (phase, rows, keep, pos)
    if not commits:
        raise ValueError("no committed generation markers; use instrumented native trace")
    windows, incomplete = [], []
    previous = {}
    for key, (phase, rows, keep, pos) in sorted(commits.items()):
        data = spans.get(key, {})
        required = [("coordinator", i, "stage_roundtrip") for i in range(stage_count)]
        required += [("coordinator", stage_count, "commit_barrier")]
        if any(tag not in data for tag in required) or any(x[2] != rows for x in data.values()):
            incomplete.append(dict(request=key[0], window=key[1], reason="missing spans or inconsistent rows"))
            previous.pop(key[0], None)
            continue
        ordered = [data[tag][:2] for tag in required]
        if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:])):
            incomplete.append(dict(request=key[0], window=key[1], reason="overlapping serial coordinator spans"))
            previous.pop(key[0], None)
            continue
        start, end = ordered[0][0], ordered[-1][1]
        item = dict(request=key[0], window=key[1], phase=phase, rows=rows, keep=keep,
                    pos0=pos, critical_ns=end-start,
                    commit_ns=ordered[-1][1]-ordered[-1][0],
                    dispatch_gap_ns=end-start-union(ordered), inter_window_gap_ns=None,
                    stages=[])
        prev = previous.get(key[0])
        if prev and key[1] == prev[0] + 1:
            if pos != prev[1] or start < prev[2]:
                raise ValueError(f"non-contiguous committed sequence {key}")
            if phase == prev[3]:
                item["inter_window_gap_ns"] = start-prev[2]
        previous[key[0]] = (key[1], pos+keep, end, phase)
        for i, (a, b) in enumerate(ordered[:-1]):
            worker, coordinator = {}, {}
            for (component, stage, name), (x, y, _) in data.items():
                if stage != i or name in ("stage_roundtrip", "commit_ack"):
                    continue
                if component == "worker" and "commit" not in name:
                    worker[name] = (x, y)
                elif component == "coordinator":
                    coordinator[name] = (x, y)
            # Host spans may end just after the coordinator receives a reply.
            # Clip to the enclosing round trip and disclose that clipping.
            clipped = [(max(a, x), min(b, y)) for x, y in worker.values() if max(a, x) < min(b, y)]
            covered = union(clipped)
            item["stages"].append(dict(stage=i, roundtrip_ns=b-a,
                worker_covered_ns=covered, not_covered_by_worker_spans_ns=b-a-covered,
                worker_clipped_spans=sum(x < a or y > b for x, y in worker.values()),
                worker_phase_ns={name: y-x for name, (x,y) in worker.items()},
                coordinator_phase_ns={name: y-x for name, (x,y) in coordinator.items()}))
        windows.append(item)
    summaries = {}
    for phase in ("prefill", "decode"):
        group = [w for w in windows if w["phase"] == phase]
        summaries[phase] = dict(windows=len(group), accepted_rows=sum(w["keep"] for w in group),
            evaluated_rows=sum(w["rows"] for w in group),
            timing={k: distribution([w[k] for w in group if w[k] is not None]) for k in
                    ("critical_ns", "commit_ns", "dispatch_gap_ns", "inter_window_gap_ns")},
            stages=[dict(stage=i, **{k: distribution([w["stages"][i][k] for w in group]) for k in
                ("roundtrip_ns", "worker_covered_ns", "not_covered_by_worker_spans_ns")}) for i in range(stage_count)])
    uncommitted = sorted(set(spans)-set(commits))
    return dict(schema="strata-heterogeneous-critical-path/v1", stages=stage_count,
        committed_markers=len(commits), complete_windows=len(windows), incomplete=incomplete,
        uncommitted=[list(k) for k in uncommitted], summaries=summaries, windows=windows,
        notes="Host monotonic spans only; same host/boot. Coverage is a union, not GPU busy time. "
              "Uncovered time is not necessarily IPC. Individual nested phase spans are not additive. "
              "Inter-window gaps exclude phase transitions and missing windows. Trace overhead is not clean throughput.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--stages", required=True, type=int)
    args = parser.parse_args(argv)
    with open(args.trace, encoding="utf-8") as stream:
        result = summarize(stream, args.stages)
    print(json.dumps(result, indent=2, allow_nan=False))
    return int(bool(result["incomplete"] or result["uncommitted"]))


if __name__ == "__main__":
    raise SystemExit(main())
