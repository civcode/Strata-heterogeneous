#!/usr/bin/env python3
"""Gate real single-relay acceptance against an independent fresh replay."""
import argparse
import json
from pathlib import Path
import sys

PROMPT = [248045, 846, 198, 2523, 513, 32767, 264]
MAX_PREFIX = 12  # seven prompt + one anchor + at most four accepted drafts
STRIDE = 40960
KV = ('k', 'v')
KV_STRIDE = 1024  # approved dense BF16, report 0021


def require(ok, message):
    if not ok:
        raise ValueError(message)


def relay_prefix(run):
    require(len(run) == 8, 'expected seven one-row prompt windows and one live relay')
    for i, (meta, packet) in enumerate(run[:-1]):
        require(meta['pos0'] == i and meta['rows'] == meta['n_keep'] == 1 and
                meta['token_ids'] == [PROMPT[i]], 'wrong acceptance prompt')
    meta, packet = run[-1]
    ids, preds = meta['token_ids'], packet['predictions']
    require(meta['pos0'] == len(PROMPT) and 2 <= len(ids) == len(preds) == meta['rows'] <= 5 and
            ids[0] == run[-2][1]['predictions'][0], 'wrong live relay anchor or extent')
    keep = 1
    while keep < len(ids) and ids[keep] == preds[keep-1]:
        keep += 1
    require(keep == meta['n_keep'], 'wrong greedy acceptance count')
    prefix = PROMPT + ids[:keep]
    require(len(prefix) <= MAX_PREFIX, 'fresh replay bound exceeded')
    return {'prefix': prefix, 'accepted_drafts': keep-1, 'rejected_drafts': len(ids)-keep,
            'live_offer': ids, 'next_anchor': preds[keep-1]}


def trace_history(events, prefix, anchor, draft_cells):
    """Validate every accepted pair and persistent KV transition, including batched prime."""
    require(0 < len(prefix) <= MAX_PREFIX and len(events) % 2 == 0, 'invalid trace history extent')
    pairs = prefix[1:] + [anchor]
    rows, old = [], {k: b'' for k in KV}
    drafts = []
    for i in range(0, len(events), 2):
        before, a = events[i]
        after, b = events[i+1]
        cell = len(rows)
        tokens = before['tokens']
        end = cell + len(tokens)
        phase = 'draft' if cell in draft_cells else 'prime'
        require(before['event'] == i+1 and after['event'] == i+2 and
                before['phase'] == phase+'_before' and after['phase'] == phase+'_after' and
                before['cell'] == after['cell'] == cell and before['upto'] == cell and after['upto'] == end and
                0 < len(tokens) <= 8 and end <= len(prefix) and tokens == after['tokens'] == pairs[cell:end] and
                (phase != 'draft' or len(tokens) == 1) and
                not any(cell < d < end for d in draft_cells), 'wrong trace accepted-pair sequence')
        for meta, data, upto in ((before, a, cell), (after, b, end)):
            require(meta['format'] == 0 and meta['kv_strides'] == ([KV_STRIDE, KV_STRIDE, 0, 0] if upto else [0]*4) and
                    len(data['r']) == len(tokens)*STRIDE and all(len(data[k]) == upto*KV_STRIDE for k in KV) and
                    not data['ks'] and not data['vs'], 'unexpected dense KV or residual geometry')
        require(a['r'] == b['r'], f'residual changed during {phase} at cell {cell}')
        for k in KV:
            for j in range(cell):
                section = slice(j*KV_STRIDE, (j+1)*KV_STRIDE)
                require(old[k][section] == a[k][section] == b[k][section],
                        f'previously committed {k} changed at cell {j} around {phase} at {cell}')
        if phase == 'draft':
            drafts.append(cell)
        for j, token in enumerate(tokens):
            pos = cell+j
            rows.append({'cell': pos, 'token': token, 'r': b['r'][j*STRIDE:(j+1)*STRIDE],
                         **{k: b[k][pos*KV_STRIDE:(pos+1)*KV_STRIDE] for k in KV}})
        old = {k: b[k] for k in KV}
    require(len(rows) == len(prefix) and drafts == list(draft_cells), 'incomplete accepted-pair history')
    return rows


def differences(fresh, reused):
    require(len(fresh) == len(reused), 'different history lengths')
    result = []
    for a, b in zip(fresh, reused):
        require((a['cell'], a['token']) == (b['cell'], b['token']), 'different history pair')
        for field in ('r', *KV):
            if a[field] != b[field]:
                result.append({'cell': a['cell'], 'field': field})
    return result


def load_trace(prefix, committed, anchor, draft_cells):
    from heterogeneous_mtp_state import load
    paths = [p for p in prefix.parent.iterdir() if p.name.startswith(prefix.name+'.mtp.') and p.suffix == '.json']
    require(0 < len(paths) <= 64, 'missing or excessive trace manifests')
    paths.sort(key=lambda p: int(p.name.rsplit('.', 2)[1]))
    events = []
    for p in paths:
        _, meta, data = load(p)
        events.append((meta, data))
    return trace_history(events, committed, anchor, draft_cells), paths[-1]


def analyze(fresh_prefix, reused_prefix, fresh_log, reused_log, traced=False):
    from heterogeneous_compare_parity import read_run, compare
    from heterogeneous_mtp_reuse import validate
    require(fresh_prefix.resolve() != reused_prefix.resolve(), 'fresh and reused capture must be distinct')
    reused = read_run(reused_prefix, allow_partial=True)
    info = relay_prefix(reused)
    fresh = read_run(fresh_prefix, mtp_fresh_replay=True)
    f, r, accepted, count = validate(fresh, reused, fresh_log, reused_log, traced=traced)
    require(count == info['accepted_drafts'], 'inconsistent acceptance count')
    require(len({p['fingerprint'] for _, p in fresh + reused}) == 1, 'different plan fingerprints')
    parity = compare(fresh, accepted, 0.001, prefill=True)
    result = dict(info, status='PASS' if count > 0 and f == r and parity['status'] == 'PASS' and
                  parity['max_abs_logit_difference'] == 0 else 'FAIL', fresh=f, reused=r, target_parity=parity,
                  committed_kv='UNMEASURED', multi_round_admission='NOT QUALIFIED')
    if traced:
        from heterogeneous_mtp_front import compare as compare_front
        end = len(info['prefix'])-1
        a, ap = load_trace(fresh_prefix, info['prefix'], info['next_anchor'], (end,))
        b, bp = load_trace(reused_prefix, info['prefix'], info['next_anchor'], (len(PROMPT)-1, end))
        diff = differences(a, b)
        front = compare_front(ap, bp)
        result.update(committed_kv='DIFFERENT' if any(d['field'] in KV for d in diff) else 'IDENTICAL',
                      accepted_pair_differences=diff, front=front)
        if diff or front['status'] != 'IDENTICAL' or not front['committed_kv_identical'] or not front['external_residual_identical']:
            result['status'] = 'FAIL'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    derive = sub.add_parser('prefix', help='derive actual accepted prefix from completed relay STFR')
    derive.add_argument('--reused-prefix', required=True, type=Path)
    gate = sub.add_parser('compare', help='qualify actual acceptance against fresh replay')
    for name in ('fresh-prefix', 'reused-prefix', 'fresh-log', 'reused-log'):
        gate.add_argument('--'+name, required=True, type=Path)
    gate.add_argument('--traced', action='store_true')
    args = parser.parse_args()
    try:
        if args.action == 'prefix':
            from heterogeneous_compare_parity import read_run
            result = relay_prefix(read_run(args.reused_prefix, allow_partial=True))
            result['status'] = 'PASS' if result['accepted_drafts'] > 0 else 'FAIL'
        else:
            result = analyze(args.fresh_prefix, args.reused_prefix, args.fresh_log.read_text(),
                             args.reused_log.read_text(), args.traced)
        print(json.dumps(result, indent=2))
        return 0 if result['status'] == 'PASS' else 1
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        print(f'INVALID: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
