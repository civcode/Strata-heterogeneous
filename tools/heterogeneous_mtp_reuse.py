#!/usr/bin/env python3
"""Qualify a bounded trace-free MTP follow-up against a fresh accepted prefix."""
import argparse
import json
from pathlib import Path
import re
import sys

MODE = re.compile(r'^HETERO_MTP_PROBE_MODE state_trace=(0|1) front_trace=(0|1) repeat_round=(0|1) relay_rounds=(\d+)$', re.M)
DRAFT = re.compile(r'^HETERO_MTP_PROBE_DRAFT cell=(\d+) verified_next=(\d+) count=(\d+) tokens=([\d,]+) \(not submitted to target\)$', re.M)


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def read_offer(log, relay_rounds, vocab):
    modes = MODE.findall(log)
    require(modes == [('0', '0', '0', str(relay_rounds))], 'not a single trace-free probe of the requested mode')
    matches = DRAFT.findall(log)
    require(len(matches) == 1, 'missing or ambiguous final diagnostic offer')
    cell, anchor, count = map(int, matches[0][:3])
    raw = matches[0][3]
    require(raw.endswith(',') and all(x.isdecimal() for x in raw[:-1].split(',')), 'malformed offer')
    tokens = [int(x) for x in raw[:-1].split(',')]
    require(1 <= count == len(tokens) <= 4 and 0 <= anchor < vocab and
            all(0 <= t < vocab for t in tokens), 'invalid offer extent or vocabulary')
    return {'cell': cell, 'anchor': anchor, 'tokens': tokens}


def accepted_rows(run):
    rows = []
    for meta, packet in run:
        for i in range(meta['n_keep']):
            vocab = packet['vocab']
            rows.append((dict(meta, pos0=meta['pos0']+i, rows=1, n_keep=1,
                              token_ids=[meta['token_ids'][i]]),
                         dict(packet, rows=1, predictions=[packet['predictions'][i]],
                              logits=packet['logits'][i*vocab:(i+1)*vocab])))
    return rows


def validate(fresh, reused, fresh_log, reused_log):
    require(bool(fresh) and len(reused) >= 2, 'missing prompt or relay')
    require(all(m['rows'] == m['n_keep'] == 1 for m, _ in fresh), 'fresh run must be one-row full commits')
    m, p = reused[-1]
    prior_m, prior_p = reused[-2]
    require(2 <= m['rows'] <= 5 and m['token_ids'][0] == prior_p['predictions'][prior_m['n_keep']-1],
            'relay anchor does not match target prediction')
    keep = 1
    while keep < m['rows'] and m['token_ids'][keep] == p['predictions'][keep-1]:
        keep += 1
    require(keep == m['n_keep'], 'relay commit did not follow greedy accept policy')
    accepted = accepted_rows(reused)
    require(len(fresh) == len(accepted) and all(a[0]['token_ids'] == b[0]['token_ids']
            for a, b in zip(fresh, accepted)), 'fresh run does not reproduce actual accepted prefix')
    vocab = p['vocab']
    f = read_offer(fresh_log, 0, vocab)
    r = read_offer(reused_log, 1, vocab)
    for offer, run in ((f, fresh), (r, accepted)):
        last_m, last_p = run[-1]
        require(offer['cell'] == last_m['pos0'] and offer['anchor'] == last_p['predictions'][0],
                'diagnostic offer is not paired with final committed row and target-selected anchor')
    return f, r, accepted, keep-1


def analyze(fresh_prefix, reused_prefix, fresh_log, reused_log):
    from heterogeneous_compare_parity import read_run, compare
    fresh = read_run(fresh_prefix)
    reused = read_run(reused_prefix, allow_partial=True)
    f, r, accepted, count = validate(fresh, reused, fresh_log, reused_log)
    parity = compare(fresh, accepted, 0.001, prefill=True)
    # Keep the existing KL/top1 gate and additionally require exact logit values.
    passed = f == r and parity['status'] == 'PASS' and parity['max_abs_logit_difference'] == 0
    return {'status': 'PASS' if passed else 'FAIL', 'fresh': f, 'reused': r,
            'accepted_drafts': count, 'target_parity': parity,
            'committed_kv': 'UNMEASURED (no MTP state/front trace)',
            'multi_round_admission': 'NOT QUALIFIED'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fresh-prefix', required=True, type=Path)
    parser.add_argument('--reused-prefix', required=True, type=Path)
    parser.add_argument('--fresh-log', required=True, type=Path)
    parser.add_argument('--reused-log', required=True, type=Path)
    args = parser.parse_args()
    try:
        result = analyze(args.fresh_prefix, args.reused_prefix,
                         args.fresh_log.read_text(), args.reused_log.read_text())
        print(json.dumps(result, indent=2))
        return 0 if result['status'] == 'PASS' else 1
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        print(f'INVALID: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
