#!/usr/bin/env python3
"""Offline qualification of the opt-in three-round native relay; no GPU APIs."""
import argparse
import json
from pathlib import Path
import sys

PROMPT = [248045, 846, 198, 2523, 513, 32767, 264]
MAX_PREFIX = 12
ROUNDS = 3


def require(ok, message):
    if not ok:
        raise ValueError(message)


def checkpoints(run):
    """Derive four actual replay prefixes and the three live greedy commits."""
    require(len(run) == len(PROMPT)+ROUNDS, 'expected exactly seven prompt and three relay windows')
    prefix = []
    for i, (m, p) in enumerate(run[:len(PROMPT)]):
        require(m['pos0'] == i and m['rows'] == m['n_keep'] == 1 and
                m['token_ids'] == [PROMPT[i]] and len(p['predictions']) == 1, 'wrong prompt capture')
        prefix.append(PROMPT[i])
    out = []
    anchor = run[len(PROMPT)-1][1]['predictions'][0]
    for i, (m, p) in enumerate(run[len(PROMPT):]):
        ids, preds = m['token_ids'], p['predictions']
        require(m['pos0'] == len(prefix) and 2 <= len(ids) == len(preds) == m['rows'] <= 5 and
                ids[0] == anchor and len(prefix)+len(ids) <= 16, 'wrong live offer extent or anchor')
        keep = 1
        while keep < len(ids) and ids[keep] == preds[keep-1]:
            keep += 1
        require(keep == m['n_keep'], 'commit did not follow target greedy acceptance')
        require(len(prefix)+keep+ROUNDS-1-i <= MAX_PREFIX, 'unrepresentable committed prefix')
        out.append(dict(prefix=prefix.copy(), cell=len(prefix)-1, anchor=anchor, candidates=ids[1:],
                        accepted_drafts=keep-1, rejected_drafts=len(ids)-keep))
        prefix.extend(ids[:keep])
        anchor = preds[keep-1]
    out.append(dict(prefix=prefix.copy(), cell=len(prefix)-1, anchor=anchor))
    return out


def offer_equal(point, offer):
    return (point['cell'], point['anchor'], point['candidates']) == (offer['cell'], offer['anchor'], offer['tokens'])


def outcome(points):
    return (sum(p['accepted_drafts'] for p in points[:-1]), sum(p['rejected_drafts'] for p in points[:-1]))


def trace_events(prefix):
    from heterogeneous_mtp_state import load
    paths = [p for p in prefix.parent.iterdir() if p.name.startswith(prefix.name+'.mtp.') and p.suffix == '.json']
    require(0 < len(paths) <= 64, 'missing or excessive trace manifests')
    paths.sort(key=lambda p: int(p.name.rsplit('.', 2)[1]))
    events, drafts = [], []
    for path in paths:
        _, meta, data = load(path)
        events.append((meta, data))
        if meta['phase'] == 'draft_after':
            drafts.append(path)
    return events, drafts


def analyze(reused_prefix, reused_log, fresh_pairs, traced=False):
    from heterogeneous_compare_parity import read_run, compare
    from heterogeneous_mtp_reuse import read_offer, accepted_rows
    require(len(fresh_pairs) == ROUNDS+1 and len({p.resolve() for p, _ in fresh_pairs} | {reused_prefix.resolve()}) == ROUNDS+2,
            'four distinct fresh captures and a distinct reused capture required')
    reused = read_run(reused_prefix, allow_partial=True, mtp_three_round=True)
    points = checkpoints(reused)
    vocab = reused[0][1]['vocab']
    final = read_offer(reused_log, ROUNDS, vocab, traced=traced)
    require((final['cell'], final['anchor']) == (points[-1]['cell'], points[-1]['anchor']), 'wrong terminal diagnostic pair')
    points[-1]['candidates'] = final['tokens']
    accepted = accepted_rows(reused)
    plan_fp = reused[0][1]['fingerprint']
    require(all(p['fingerprint'] == plan_fp and p['vocab'] == vocab for _, p in reused), 'reused identity changed')
    reused_history, reused_drafts = None, None
    if traced:
        from heterogeneous_mtp_acceptance import trace_history
        events, reused_drafts = trace_events(reused_prefix)
        require(len(reused_drafts) == ROUNDS+1, 'missing per-round native front captures')
        reused_history = trace_history(events, points[-1]['prefix'], points[-1]['anchor'], tuple(p['cell'] for p in points))
    gates = []
    for i, ((path, log), point) in enumerate(zip(fresh_pairs, points)):
        fresh = read_run(path, mtp_fresh_replay=True)
        require(all(p['fingerprint'] == plan_fp and p['vocab'] == vocab for _, p in fresh), 'fresh identity changed')
        require([m['token_ids'] for m, _ in fresh] == [[t] for t in point['prefix']], 'wrong independent fresh prefix')
        require(fresh[-1][1]['predictions'][0] == point['anchor'], 'wrong fresh target anchor')
        offer = read_offer(log, 0, vocab, traced=traced)
        parity = compare(fresh, accepted[:len(fresh)], 0.001, prefill=True)
        passed = offer_equal(point, offer) and parity['status'] == 'PASS' and parity['max_abs_logit_difference'] == 0
        gate = dict(point, checkpoint=i, fresh_offer=offer, target_parity=parity, committed_kv='UNMEASURED')
        if traced:
            from heterogeneous_mtp_acceptance import trace_history, differences
            from heterogeneous_mtp_front import compare as compare_front
            events, drafts = trace_events(path)
            require(len(drafts) == 1, 'wrong fresh front history')
            history = trace_history(events, point['prefix'], point['anchor'], (point['cell'],))
            diff = differences(history, reused_history[:len(history)])
            front = compare_front(drafts[0], reused_drafts[i])
            gate.update(accepted_pair_differences=diff, front=front,
                        committed_kv='DIFFERENT' if any(d['field'] in ('k','v') for d in diff) else 'IDENTICAL')
            passed = passed and not diff and front['status'] == 'IDENTICAL' and front['committed_kv_identical'] and front['external_residual_identical']
        gate['status'] = 'PASS' if passed else 'FAIL'
        gates.append(gate)
    a, r = outcome(points)
    return dict(status='PASS' if a and r and all(g['status']=='PASS' for g in gates) else 'FAIL',
                accepted_drafts=a, rejected_drafts=r, traced=traced, checkpoints=gates,
                production_admission='NOT QUALIFIED')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    derive = sub.add_parser('prefixes')
    derive.add_argument('--reused-prefix', type=Path, required=True)
    gate = sub.add_parser('compare')
    gate.add_argument('--reused-prefix', type=Path, required=True)
    gate.add_argument('--reused-log', type=Path, required=True)
    gate.add_argument('--fresh', nargs=2, metavar=('PREFIX','LOG'), action='append', required=True, type=Path)
    gate.add_argument('--traced', action='store_true')
    args = parser.parse_args()
    try:
        if args.action == 'prefixes':
            from heterogeneous_compare_parity import read_run
            points = checkpoints(read_run(args.reused_prefix, allow_partial=True, mtp_three_round=True))
            result = dict(status='PASS', checkpoints=points)
        else:
            result = analyze(args.reused_prefix, args.reused_log.read_text(),
                             [(p, log.read_text()) for p, log in args.fresh], args.traced)
        print(json.dumps(result, indent=2))
        return 0 if result['status']=='PASS' else 1
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        print(f'INVALID: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
