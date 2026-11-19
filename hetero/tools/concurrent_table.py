#!/usr/bin/env python3
"""Tables of serve_session.py ... concurrent results:
    concurrent_table.py RUN_DIR PREFIX:LABEL [PREFIX:LABEL ...]
Each PREFIX names RUN_DIR/PREFIX-<i>-result.json for every sample i.  Per config: aggregate decode tok/s (tokens / the
wall time with at least one request decoding), aggregate output tok/s (tokens / load wall time), load wall time; then
per prompt kind the median over all requests of the samples: client-side decode tok/s and time to first token."""
import glob, json, os, re, statistics as st, sys

run = sys.argv[1]
agg, per = [], []
for arg in sys.argv[2:]:
    prefix, label = arg.split(':', 1)
    files = sorted(glob.glob(os.path.join(run, prefix + '-[0-9]*-result.json')),
                   key=lambda p: int(re.search(r'-(\d+)-result\.json$', p).group(1)))
    cs = [json.load(open(f))['concurrent'] for f in files]
    def col(xs, f='%.1f'):
        return ('**' + f + '** (%s)') % (st.median(xs), ' / '.join(f % x for x in xs))
    agg.append('| %s | %s | %s | %s |' % (label, col([c['aggregate_decode_tps'] for c in cs]), col([c['aggregate_output_tps'] for c in cs]),
                                         col([c['wall_s'] for c in cs])))
    for k in ('code', 'long'):
        rs = [r for c in cs for r in c['requests'] if r['prompt'] == k and 'error' not in r]
        dec = [r['client_decode_tps'] for r in rs if r.get('client_decode_tps')]
        tt = [r['ttft_s'] * 1000 for r in rs if r.get('ttft_s') is not None]
        srv = [r['timings']['predicted_per_second'] for r in rs if r.get('timings')]
        per.append('| %s | %s | %.1f (%.1f to %.1f) | %.1f | %.0f (%.0f to %.0f) | %d |' % (
            label, k, st.median(dec), min(dec), max(dec), st.median(srv) if srv else float('nan'), st.median(tt), min(tt), max(tt), len(rs)))
print('| Config | Aggregate decode tok/s | Aggregate output tok/s | Load wall s |\n| --- | --- | --- | --- |')
print('\n'.join(agg))
print('\n| Config | Prompt | Decode tok/s per request, client (min to max) | Server decode tok/s | First token ms (min to max) | Requests |')
print('| --- | --- | --- | --- | --- | --- |')
print('\n'.join(per))
