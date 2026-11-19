#!/usr/bin/env python3
"""Prefill / decode tok/s and drafts per prompt from serve_session.py results (suite measure):
    session_table.py RUN_DIR PREFIX:LABEL [PREFIX:LABEL ...]
Each PREFIX names RUN_DIR/PREFIX-<i>-result.json for every sample i found; prints a Markdown row per prompt with the
samples in order and their median."""
import glob, json, os, re, statistics as st, sys

run = sys.argv[1]
print('| Prompt | Config | Prefill tok/s | Decode tok/s | Drafts accepted / drafted |')
print('| --- | --- | --- | --- | --- |')
rows = {}
for arg in sys.argv[2:]:
    prefix, label = arg.split(':', 1)
    files = sorted(glob.glob(os.path.join(run, prefix + '-[0-9]*-result.json')),
                   key=lambda p: int(re.search(r'-(\d+)-result\.json$', p).group(1)))
    for k in ('code', 'long', 'long16k'):
        ts = [json.load(open(f))[k]['timings'] for f in files]
        def col(key):
            xs = [t[key] for t in ts]
            return '**%.1f** (%s)' % (st.median(xs), ' / '.join('%.1f' % x for x in xs))
        dr = ', '.join('%d/%d' % (t['draft_n_accepted'], t['draft_n']) for t in ts)
        rows.setdefault(k, []).append('| %s (%d) | %s | %s | %s | %s |' % (k, ts[0]['prompt_n'], label, col('prompt_per_second'),
                                                                        col('predicted_per_second'), dr))
for k in ('code', 'long', 'long16k'):
    print('\n'.join(rows.get(k, [])))
