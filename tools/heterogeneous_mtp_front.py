#!/usr/bin/env python3
"""Exact byte compare of bounded one-row native MTP front traces."""
import argparse
import json
from pathlib import Path
import struct
import sys

FIELDS = (
    ('token', 4), ('step', 16), ('position', 96), ('rest_step', 16),
    ('rest_position', 96), ('selected_row', 8), ('residual', 40960),
    ('embedding', 10240), ('embedding_norm', 10240), ('embedding_fc', 10240),
    ('hidden_norm', 40960), ('hidden_fc', 40960), ('mixed_residual', 40960),
    ('inject', 16), ('mixed', 10240), ('q8_mixed', 2880),
    ('k_project', 2048), ('v_project', 2048), ('k_append', 2048), ('v_append', 2048),
    ('source_before', 40960), ('source_after', 40960),
)
BYTES = sum(size for _, size in FIELDS)


def split(raw):
    if len(raw) != BYTES:
        raise ValueError('invalid native front trace extent')
    result, at = {}, 0
    for name, size in FIELDS:
        result[name] = raw[at:at+size]
        at += size
    return result


def staging(fields, cell, token, residual):
    step = struct.pack('<4i', cell, cell+1, (cell+1)//4, cell+1)
    pos = struct.pack('<24i', *([cell]*24))
    expected = {'token': struct.pack('<i', token), 'step': step, 'position': pos,
                'rest_step': step, 'rest_position': pos,
                'selected_row': struct.pack('<2i', 0, 0), 'residual': residual,
                'source_before': residual, 'source_after': residual}
    return [name for name, raw in expected.items() if fields[name] != raw]


def differences(a, b):
    return {name: sum(x != y for x, y in zip(a[name], b[name]))
            for name, _ in FIELDS}


def load_front(path):
    from heterogeneous_mtp_state import load, require
    completion, m, state = load(path)
    require(m['phase'] == 'draft_after' and len(m['tokens']) == 1 and
            m.get('front_schema') == 2 and m.get('front_bytes') == BYTES,
            'manifest not a completed one-row front trace')
    raw_path = Path(path).with_suffix('.front.bin')
    require(raw_path.stat().st_size == BYTES, 'invalid front trace extent')
    return completion, m, state, split(raw_path.read_bytes())


def compare(a, b):
    from heterogeneous_mtp_state import require
    ca, ma, sa, fa = load_front(a)
    cb, mb, sb, fb = load_front(b)
    require((ca['model_fingerprint'], ca['plan'], ma['cell'], ma['tokens'],
             ma['format'], ma['kv_strides']) ==
            (cb['model_fingerprint'], cb['plan'], mb['cell'], mb['tokens'],
             mb['format'], mb['kv_strides']), 'different model/plan/input/layout')
    aa = staging(fa, ma['cell'], ma['tokens'][0], sa['r'])
    ab = staging(fb, mb['cell'], mb['tokens'][0], sb['r'])
    counts = differences(fa, fb)
    order = ('source_before', 'source_after') + tuple(name for name, _ in FIELDS if name not in ('source_before', 'source_after'))
    first = next((name for name in order if counts[name]), None)
    return {'status': 'IDENTICAL' if not first and not aa and not ab else 'DIFFERENT',
            'cell': ma['cell'], 'token': ma['tokens'][0],
            'staging_mismatches_a': aa, 'staging_mismatches_b': ab,
            'external_residual_identical': sa['r'] == sb['r'],
            'first_different_field': first, 'different_bytes': counts,
            'committed_kv_identical': all(sa[k] == sb[k] for k in ('k','v','ks','vs'))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('a', type=Path)
    p.add_argument('b', type=Path)
    args = p.parse_args()
    try:
        result = compare(args.a, args.b)
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f'INVALID: {e}', file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'IDENTICAL' and result['committed_kv_identical'] else 1


if __name__ == '__main__':
    sys.exit(main())
