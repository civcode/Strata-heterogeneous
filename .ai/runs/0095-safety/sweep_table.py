#!/usr/bin/env python3
"""Table of full-replay summaries: sweep_table.py PREFIX (e.g. sweep-full-q) -> one row per run, medians per arm."""
import json, re, statistics as S, sys
from pathlib import Path
runs = sorted(Path('.ai/runs').glob(f'0095-{sys.argv[1]}*'))
arms = {}
print('| run | generated | engine decode tok/s | acceptance % | ttft median s | overall | thinking | answering | tool calls | pauses (n, total s, max s) |')
print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---|')
for d in runs:
    p = d / 'replay/summary.json'
    if not p.exists():
        continue
    s = json.load(open(p)); m = s['metrics_phases']; r = json.load(open(d / 'runner.json'))
    ps = [float(re.search(r'for (\d+) ms', l)[1]) / 1000 for l in r.get('gpu_pause_lines', []) if re.search(r'for (\d+) ms', l)]
    row = [s['generated_tokens'], s['engine_decode_tps'], s['draft_acceptance_pct'], s['ttft_median_s'], m['overall']['tps'],
           m['thinking']['tps'], m['answering']['tps'], m['writing tool calls']['tps']]
    arm = re.sub(r'-r\d+$', '', d.name.replace('0095-', ''))
    arms.setdefault(arm, []).append(row)
    print(f'| {d.name[5:]} | ' + ' | '.join(str(x) for x in row) + f' | {len(ps)}, {sum(ps):.1f}, {max(ps, default=0):.1f} |')
print('\n| arm (median) | runs | engine decode | acceptance | ttft | overall | thinking | answering | tool calls |')
print('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
for a, rows in arms.items():
    med = [round(S.median(c), 2) for c in zip(*rows)]
    print(f'| {a} | {len(rows)} | ' + ' | '.join(str(x) for x in med[1:]) + ' |')
