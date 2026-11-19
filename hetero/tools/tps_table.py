#!/usr/bin/env python3
"""Prefill / decode tok/s and first-token time per request from engine_drive.py results:
    tps_table.py RUN_DIR PREFIX:LABEL [PREFIX:LABEL ...]
Each PREFIX names RUN_DIR/PREFIX-<i>.json for every sample i found; prints the samples in order and the median."""
import glob, json, os, statistics as st, sys

run = sys.argv[1]
for arg in sys.argv[2:]:
    prefix, label = arg.split(':', 1)
    files = sorted(glob.glob(os.path.join(run, prefix + '-[0-9]*.json')), key=lambda p: int(p.rsplit('-', 1)[1][:-5]))
    names = {}
    for f in files:
        for r in json.load(open(f)).get('requests', []):
            names.setdefault(r['name'], []).append(r)
    for name, rs in names.items():
        def col(key, scale=1.0):
            xs = [r[key] * scale for r in rs if r.get(key) is not None]
            return '%s (median %.1f)' % (' / '.join('%.1f' % x for x in xs), st.median(xs)) if xs else '-'
        print('%-14s %-10s prefill %s | decode %s | first token ms %s' % (label, name, col('prefill_tps'), col('decode_tps'),
                                                                          col('first_token_s', 1000.0)))
