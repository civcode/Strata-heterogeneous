#!/usr/bin/env python3
"""Correlate existing raw bounded CUDA prefill state and bridge/logit captures."""
import argparse
import json
from pathlib import Path
import sys


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def exact(result):
    return result['status'] == 'PASS' and result['max_abs_logit_difference'] == 0


def fields_different(fields):
    return sorted(name for name, item in fields.items()
                  if not item['bytes_equal'] or item.get('nonfinite_values', 0))


def summarize(state, bridge, comparisons):
    history = state['observed_ple_prev']
    expected = state['expected_ple_prev']
    diffs = {name: fields_different(fields) for name, fields in state['comparisons'].items()}
    stable = (exact(comparisons['serial-a_vs_serial-b']) and not diffs['serial-a_vs_serial-b'] and
              history['serial-a'] == history['serial-b'] == expected)
    state_equal = (state['status'] == 'IDENTICAL' and all(prev == expected for prev in history.values()) and
                   not any(diffs.values()))
    logits = all(comparisons[k]['status'] == 'PASS' for k in ('serial-a_vs_native','serial-b_vs_native'))
    return dict(serial_control_exact=stable, bridge_exact=bridge['status']=='PASS',
                cuda_state_exact=state_equal, native_logits_gate=logits,
                native_logits_exact=all(exact(c) for c in comparisons.values()),
                different_state_fields=diffs, expected_ple_prev=expected, observed_ple_prev=history)


def analyze(prefixes):
    from heterogeneous_prefill_parity import load, MAX_KL
    from heterogeneous_prefill_state import analyze as analyze_state
    from heterogeneous_compare_parity import compare
    require(1 <= len(prefixes) <= 16 and len({p.resolve() for p in prefixes}) == len(prefixes),
            'expected distinct bounded cut captures')
    rows, source = [], None
    tokens = []
    for prefix in prefixes:
        marker, runs, bridge = load(prefix)
        cut = marker['prefix_rows']
        require(1 <= cut <= 16, 'CUDA state capture cut must be 1..16')
        m, p = runs['serial-a'][0]
        identity = tuple(marker[k] for k in ('plan','model_fingerprint','chunk','hc','n_embd')) + tuple(
            m[k] for k in ('pack_index_sha256','gguf_shards','profile_path')) + (p['fingerprint'],p['vocab'])
        seq = marker['token_ids']
        if source is None:
            source = identity
            tokens = seq
        require(identity == source and tokens[:min(len(tokens),len(seq))] == seq[:min(len(tokens),len(seq))],
                'cuts must use the same chunk, model, plan and token sequence')
        if len(seq) > len(tokens):
            tokens = seq
        state = analyze_state(prefix)
        require(state['position'] == cut, 'state and target cuts differ')
        comparisons = {f'{a}_vs_{b}': compare(runs[a],runs[b],MAX_KL,prefill=True)
                       for a,b in (('serial-a','serial-b'),('serial-a','native'),('serial-b','native'))}
        rows.append(dict(cut=cut,prefix=str(prefix),bridge=bridge,comparisons=comparisons,
                         **summarize(state,bridge,comparisons)))
    rows.sort(key=lambda r: r['cut'])
    require(len({r['cut'] for r in rows}) == len(rows), 'duplicate cut')
    keys = ('serial_control_exact','bridge_exact','cuda_state_exact','native_logits_gate')
    earliest = {key: next((r['cut'] for r in rows if not r[key]),None) for key in keys}
    return dict(status='PASS' if all(v is None for v in earliest.values()) else 'FAIL',
                scope='OBSERVED_CUTS_ONLY', earliest_failed_cut=earliest, cuts=rows,
                causal_operation='NOT ESTABLISHED', production_admission='NOT QUALIFIED')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', type=Path, required=True, action='append')
    args = parser.parse_args()
    try:
        result = analyze(args.prefix)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result['status']=='PASS' else 1
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
        print(f'INVALID: {exc}',file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
