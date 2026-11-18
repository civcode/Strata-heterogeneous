#!/usr/bin/env python3
"""Compare the generated token ids of two engine_drive.py results:  cmp_tokens.py A.json B.json
Per request (by name): identical, or the first differing position.  Exit status 1 when any differs."""
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
rb = {r['name'] + r.get('keys', ''): r for r in b['requests']}
bad = 0
for r in a['requests']:
    o = rb.get(r['name'] + r.get('keys', ''))
    if o is None:
        print(r['name'], 'missing in', sys.argv[2]); bad = 1; continue
    ta, tb = r['tokens'], o['tokens']
    d = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
    if d is None and len(ta) != len(tb):
        d = min(len(ta), len(tb))
    print('%-10s %-30s %4d / %4d tokens: %s' % (r['name'], r.get('keys', ''), len(ta), len(tb), 'identical' if d is None else 'first difference at %d' % d))
    bad |= d is not None
sys.exit(1 if bad else 0)
