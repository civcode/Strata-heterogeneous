#!/usr/bin/env python3
"""Summarize frozen bounded replay JSON; timing is a native feedback-rate proxy."""
import argparse
import json
import math
import statistics
from pathlib import Path


def summarize(manifest, run, warmup=1):
    turns = manifest["turns"]
    repeats = manifest["runs_per_turn"]
    results = run["completions"]
    if not turns or not 0 <= warmup < repeats or len(results) != len(turns) * repeats:
        raise ValueError("replay manifest/result shape mismatch")
    task_turn_rates = {}
    detail = []
    for index, turn in enumerate(turns):
        samples = [results[r * len(turns) + index] for r in range(repeats)]
        expected = turn["prompt_token_ids"]
        if any(s.get("prompt_token_ids") != expected for s in samples):
            raise ValueError("replay prompt identity mismatch")
        warm = samples[warmup:]
        rates = [s["timing"].get("decode_tokens_per_second") for s in warm]
        valid = [x for x in rates if isinstance(x, (int, float)) and not isinstance(x, bool)
                 and math.isfinite(x) and x > 0]
        rate = statistics.median(valid) if len(valid) == len(warm) else None
        task_turn_rates.setdefault(turn["task"], []).append(rate)
        detail.append({"task": turn["task"], "turn": turn["turn"], "warm_samples": len(warm),
                       "valid_rates": len(valid), "median_feedback_tok_s": rate,
                       "outputs": [len(s["token_ids"]) for s in warm],
                       "finish_reasons": [s["finish_reason"] for s in warm],
                       "median_prefill_ms": statistics.median(s["timing"]["prefill_ms"] for s in warm),
                       "median_elapsed_ms": statistics.median(s["timing"]["elapsed_ms"] for s in warm)})
    tasks = {key: statistics.median(values) if all(v is not None for v in values) else None
             for key, values in task_turn_rates.items()}
    complete = all(v is not None for v in tasks.values())
    overall = statistics.median(tasks.values()) if complete else None
    def percentile(fraction):
        if not complete:
            return None
        values = sorted(tasks.values())
        point = (len(values) - 1) * fraction
        lo = int(point)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (point - lo)
    return {"metric": "native feedback tokens/s; not measured user-visible streaming rate",
            "warmup_per_turn": warmup, "turns": detail, "task_medians": tasks,
            "equal_weighted_task_median": overall,
            "task_p10": percentile(0.1), "task_p90": percentile(0.9)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(summarize(json.loads(args.manifest.read_text()),
                               json.loads(args.run.read_text()), args.warmup), indent=2))


if __name__ == "__main__":
    main()
