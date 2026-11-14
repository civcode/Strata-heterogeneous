#!/usr/bin/env python3
"""Compare completed bounded MTP state trace events, byte-for-byte (no GPU)."""
import argparse
import json
from pathlib import Path
import sys

PARTS = ('k', 'v', 'ks', 'vs')
PHASES = ('prime_before', 'prime_after', 'draft_before', 'draft_after', 'reprime_after')


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def load(path):
    path = Path(path)
    prefix, marker, event = str(path).rpartition('.mtp.')
    require(marker and event.endswith('.json') and event[:-5].isdigit(), 'invalid trace manifest name')
    completion = json.loads(Path(prefix + '.complete.json').read_text())
    require(completion['schema'] == 'strata-heterogeneous-parity/v1' and
            type(completion['model_fingerprint']) is int and completion['model_fingerprint'] > 0 and
            isinstance(completion['plan'], str) and completion['plan'], 'invalid completed target run')
    m = json.loads(path.read_text())
    require(m['schema'] == 'strata-mtp-state/v1' and m['event'] == int(event[:-5]) and
            1 <= m['event'] <= 64 and m['phase'] in PHASES, 'invalid trace identity')
    cell, upto, tokens = m['cell'], m['upto'], m['tokens']
    require(type(cell) is int and type(upto) is int and isinstance(tokens, list) and
            1 <= len(tokens) <= 8 and 0 <= cell < cell + len(tokens) <= 16 and
            all(type(t) is int and 0 <= t < 248320 for t in tokens), 'invalid trace position/tokens')
    require(upto == (cell if m['phase'].endswith('before') else cell + len(tokens)), 'invalid trace extent')
    require(m['residual_stride'] == 40960 and type(m['format']) is int, 'invalid trace geometry')
    strides = m['kv_strides']
    require(isinstance(strides, list) and len(strides) == 4 and
            all(type(s) is int and 0 <= s <= 65536 for s in strides) and
            (upto == 0 and not any(strides) or upto > 0 and strides[0] > 0 and strides[1] > 0),
            'invalid draft KV strides')
    def part(name, size):
        p = path.with_suffix('.' + name + '.bin')
        require(p.is_file() and p.stat().st_size == size, 'missing or wrong extent: ' + str(p))
        raw = p.read_bytes()
        require(len(raw) == size, 'trace changed during read')
        return raw
    data = {'r': part('r', len(tokens) * m['residual_stride'])}
    data.update({name: part(name, upto * stride) for name, stride in zip(PARTS, strides)})
    return completion, m, data


def compare(a, b, upto=None, kv_only=False):
    ca, ma, da = load(a)
    cb, mb, db = load(b)
    require((ca['model_fingerprint'], ca['plan']) == (cb['model_fingerprint'], cb['plan']), 'different model/plan')
    require(ma['format'] == mb['format'] and ma['kv_strides'] == mb['kv_strides'], 'different draft KV layouts')
    if upto is None:
        require(ma['upto'] == mb['upto'], 'different committed extents; select --upto explicitly')
        upto = ma['upto']
    require(type(upto) is int and 0 <= upto <= min(ma['upto'], mb['upto']), 'comparison outside committed cells')
    if not kv_only:
        require(ma['cell'] == mb['cell'] and ma['tokens'] == mb['tokens'], 'different input pair; use --kv-only explicitly')
    fields = {}
    def check(name, stride, count, pos):
        if not stride:
            return
        x, y = da[name], db[name]
        cells = [pos + i for i in range(count) if x[i*stride:(i+1)*stride] != y[i*stride:(i+1)*stride]]
        fields[name] = {'compared_cells': count, 'different_cells': cells, 'identical': not cells}
    if not kv_only:
        check('r', ma['residual_stride'], len(ma['tokens']), ma['cell'])
    for name, stride in zip(PARTS, ma['kv_strides']):
        check(name, stride, upto, 0)
    require(any(v['compared_cells'] for v in fields.values()), 'empty comparison')
    return {'status': 'IDENTICAL' if all(v['identical'] for v in fields.values()) else 'DIFFERENT',
            'a': str(a), 'b': str(b), 'upto': upto, 'kv_only': kv_only, 'fields': fields}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('a', type=Path)
    p.add_argument('b', type=Path)
    p.add_argument('--upto', type=int)
    p.add_argument('--kv-only', action='store_true')
    args = p.parse_args()
    try:
        result = compare(args.a, args.b, args.upto, args.kv_only)
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f'INVALID: {e}', file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'IDENTICAL' else 1


if __name__ == '__main__':
    sys.exit(main())
