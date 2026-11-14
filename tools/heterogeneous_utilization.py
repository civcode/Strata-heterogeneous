"""Phase-aligned, passive utilization summary for one same-host resident benchmark."""
import argparse
import json
import math
from collections import defaultdict
from statistics import mean, median


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def stats(values):
    return {"samples": len(values), "mean": mean(values) if values else None,
            "median": median(values) if values else None,
            "peak": max(values) if values else None}


def fraction(values, predicate):
    return 100 * sum(predicate(x) for x in values) / len(values) if values else None


def phase(name, start, end, request=None, feedback=None):
    if not number(start) or not number(end) or start > end:
        raise ValueError(f"invalid phase bounds: {name}")
    return dict(name=name, start=start, end=end, request=request, feedback=feedback)


def phases(benchmark):
    if benchmark.get("status") != "complete":
        raise ValueError("completed resident benchmark required")
    bounds = benchmark["phase_monotonic_s"]
    result = [phase("startup", bounds["startup_start"], bounds["startup_end"])]
    previous = bounds["startup_end"]
    for record in benchmark["completions"]:
        t = record["timing"]
        b = t["phase_monotonic_s"]
        req = record["request_id"]
        if record["finish_reason"] not in ("stop", "length"):
            raise ValueError("only successful benchmark requests are supported")
        result += [phase("idle", previous, b["request_start"]),
                   phase("prefill_including_reset", b["request_start"], b["prefill_end"], req),
                   phase("decode", b["prefill_end"], b["decode_end"], req, t["decode_feedback_tokens"]),
                   phase("completion_tail", b["decode_end"], b["request_end"], req)]
        previous = b["request_end"]
    result += [phase("idle", previous, bounds["shutdown_start"]),
               phase("shutdown", bounds["shutdown_start"], bounds["shutdown_end"])]
    return result


def summarize(benchmark, samples, devices):
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("explicit distinct monitor devices are required")
    groups = defaultdict(dict)
    interval = None
    for sample in samples:
        device = f'{sample["backend"]}:{sample["monitor_index"]}'
        if device not in devices:
            raise ValueError(f"unexpected monitor device {device}")
        cycle = sample["cycle"]
        start, end, step = (sample[k] for k in ("read_start_s", "monotonic_s", "interval_s"))
        if not (type(cycle) is int and cycle > 0 and number(start) and number(end)
                and start <= end and number(step) and 0.25 <= step <= 5):
            raise ValueError("invalid sample timing")
        if interval is None:
            interval = step
        if interval != step or device in groups[cycle]:
            raise ValueError("mixed sampling intervals or duplicate cycle/device")
        groups[cycle][device] = sample
    if interval is None:
        raise ValueError("empty telemetry input")
    # Group by recorded cycle, not rounded wall clock. Sensor queries are
    # sequential, so even aligned pairs are only a sampled overlap proxy.
    def belongs(row, p):
        return p["start"] <= row["read_start_s"] <= row["monotonic_s"] < p["end"]

    def valid_util(row):
        u = row.get("util_percent")
        return number(u) and 0 <= u <= 100

    def summarize_phases(ps):
        elapsed = sum(p["end"] - p["start"] for p in ps)
        rows = {d: [] for d in devices}
        complete = []
        complete_power = []
        for group in groups.values():
            for d, row in group.items():
                if any(belongs(row, p) for p in ps):
                    rows[d].append(row)
            aligned = (len(group) == len(devices)
                       and any(all(belongs(row, p) for row in group.values()) for p in ps)
                       and max(row["monotonic_s"] for row in group.values())
                       - min(row["read_start_s"] for row in group.values()) <= interval / 2)
            if aligned and all(valid_util(row) for row in group.values()):
                complete.append(group)
            if aligned and all(number(row.get("power_w")) and row["power_w"] >= 0
                               for row in group.values()):
                complete_power.append(sum(row["power_w"] for row in group.values()))
        gpu = {}
        for d, records in rows.items():
            utilization = [row["util_percent"] for row in records if valid_util(row)]
            vram = [row["vram_used_bytes"] for row in records if number(row.get("vram_used_bytes"))
                    and row["vram_used_bytes"] >= 0]
            power = [row["power_w"] for row in records if number(row.get("power_w")) and row["power_w"] >= 0]
            limits = [row["power_limit_w"] for row in records
                      if number(row.get("power_limit_w")) and row["power_limit_w"] > 0]
            temp = [row["temperature_c"] for row in records if number(row.get("temperature_c"))]
            gpu[d] = {"observed_samples": len(records), "utilization_percent": stats(utilization),
                      "samples_at_least_90_percent": fraction(utilization, lambda x: x >= 90),
                      "samples_at_least_95_percent": fraction(utilization, lambda x: x >= 95),
                      "samples_below_10_percent": fraction(utilization, lambda x: x < 10),
                      "vram_used_bytes": stats(vram), "power_w": stats(power),
                      "power_limit_w": stats(limits), "temperature_c": stats(temp),
                      "vram_total_bytes": max((row["vram_total_bytes"] for row in records
                                               if number(row.get("vram_total_bytes")) and row["vram_total_bytes"] > 0),
                                              default=None)}
        mins = [min(row["util_percent"] for row in group.values()) for group in complete]
        feedback = sum(p["feedback"] or 0 for p in ps)
        expected = elapsed / interval
        return {"duration_s": elapsed, "expected_cycles_approx": expected,
                "feedback_tokens": feedback, "feedback_tok_s": feedback / elapsed if feedback and elapsed else None,
                "gpu": gpu, "aligned_complete_cycles": len(complete),
                "total_gpu_power_w": stats(complete_power),
                "power_low_sample_warning": len(complete_power) < 10 or len(complete_power) < expected * 0.8,
                "all_gpus_at_least_10_percent": fraction(mins, lambda x: x >= 10),
                "all_gpus_at_least_90_percent": fraction(mins, lambda x: x >= 90),
                "all_gpus_at_least_95_percent": fraction(mins, lambda x: x >= 95),
                "low_sample_warning": len(complete) < 10 or len(complete) < expected * 0.8}

    ps = phases(benchmark)
    decode = [p for p in ps if p["name"] == "decode"]
    warm = decode[1:]
    return {"schema": "strata-heterogeneous-utilization/v1", "interval_s": interval,
            "devices": devices, "phases": [dict(p, **summarize_phases([p])) for p in ps],
            "warm_decode": summarize_phases(warm),
            "notes": "Same Linux host/boot only. Unweighted sensor samples, not exact time fractions or kernel occupancy. "
                     "Driver averaging windows differ; aligned cycles are not proof of instantaneous overlap. "
                     "Decode excludes startup, prefill/reset and final completion tail; includes IPC/consumer pacing. "
                     "First request excluded from warm decode. Missing readings are not zero. "
                     "Total GPU power sums only complete aligned power cycles; sampled board/sensor readings, not wall power."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--telemetry", required=True)
    parser.add_argument("--device", action="append", required=True, help="mapped cuda:INDEX or hip:INDEX")
    args = parser.parse_args(argv)
    with open(args.benchmark, encoding="utf-8") as stream:
        bench = json.load(stream)
    with open(args.telemetry, encoding="utf-8") as stream:
        samples = [json.loads(line) for line in stream if line.strip()]
    print(json.dumps(summarize(bench, samples, args.device), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
