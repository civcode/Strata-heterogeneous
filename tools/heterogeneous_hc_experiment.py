#!/usr/bin/env python3
"""Compare a fresh bounded HC intervention with its unchanged native baseline.

This is an intervention analysis, not a production or whole-model admission gate.
"""
import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np

OWNERS = (('cuda', 0, 12), ('hip', 12, 30), ('hip', 30, 48))
LOG = re.compile(r'STRATA_HC_EXPERIMENT backend=(cuda|hip) layers=(-?\d+)-(-?\d+) '
                 r'requested=(bf16x2|fp32) scope=(all|cuda|hip) effective=(off|bf16x2|fp32)\b')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def check_log(text, identity):
    seen = set()
    for line in text.splitlines():
        if 'STRATA_HC_EXPERIMENT backend=' not in line:
            continue
        match = LOG.search(line)
        require(match is not None and identity is not None, 'unexpected/malformed HC experiment log')
        backend, lo, hi, mode, scope, effective = match.groups()
        owner = (backend, int(lo), int(hi))
        require(owner in OWNERS and mode == identity['mode'] and scope == identity['backend'] and
                effective == (mode if scope in ('all', backend) else 'off'), 'wrong effective HC experiment owner/mode')
        seen.add(owner)
    require(seen == (set(OWNERS) if identity else set()), 'missing HC experiment owner records')


def compare_evidence(b, e, identity):
    a, z = b['marker'], e['marker']
    require(a.get('hc_weight_contract') == z.get('hc_weight_contract'), 'activation intervention changed weight contract')
    require(a.get('hc_experiment') is None and z.get('hc_experiment') == identity and identity is not None,
            'expected default baseline and explicitly labeled intervention')
    require(all(a[k] == z[k] for k in ('plan', 'prefix_rows', 'chunk', 'token_ids', 'transport')) and
            b['source'] == e['source'] and not a['stage_reference'] and not z['stage_reference'] and
            not a.get('numerical_trace') and not z.get('numerical_trace'), 'baseline/candidate identity or trace mode differs')
    for c in (b, e):
        require(c['bridge']['status'] == 'PASS' and c['local_links_exact'] and c['serial_controls_exact'],
                'local bridge, adoption, or serial integrity failed')
    require(all(b['logits'][m] == e['logits'][m] for m in ('serial-a', 'serial-b')),
            'intervention changed serial continuation controls')
    require(b['residuals'].keys() == e['residuals'].keys() and b['states'].keys() == e['states'].keys(),
            'stage capture inventory differs')
    for stage, state in b['states'].items():
        require(state['identity'] == e['states'][stage]['identity'], 'native checkpoint metadata differs')
    for key, digest in b['residuals'].items():
        if key.startswith('stage0.') and (key.endswith('.in') or identity['backend'] == 'hip'):
            require(digest == e['residuals'][key], 'untouched CUDA input/output changed')
    if identity['backend'] == 'hip':
        require(b['states']['0'] == e['states']['0'], 'HIP-only intervention changed CUDA checkpoint')


def metric(a, b):
    require(a.shape == b.shape and a.size and np.isfinite(a).all() and np.isfinite(b).all(), 'invalid residual values')
    a, b = a.astype(np.float64), b.astype(np.float64)
    d = b - a
    denom = float(np.linalg.norm(a))
    return {'max_abs': float(np.max(np.abs(d))), 'rmse': float(np.sqrt(np.mean(d*d))),
            'relative_l2_to_baseline': float(np.linalg.norm(d))/denom if denom else None}


def serial_state_signatures(prefix, state_load, signature):
    _, captures, expected = state_load(prefix)
    result = {}
    for mode in ('serial-a', 'serial-b'):
        prev, fields = captures[mode]
        require(prev == expected, 'serial PLE history differs from expected tokens')
        result[mode] = {name: (kind, size, signature(path, size, kind))
                        for name, (kind, size, path) in fields.items()}
    require(result['serial-a'] == result['serial-b'], 'unstable serial CUDA checkpoint')
    return result['serial-a']


def analyze(baseline, experiment, baseline_log, experiment_log, reference=None, reference_log=None):
    from heterogeneous_prefill_stage_gate import (read_capture, experiment_identity, signature, pair, ROW_BYTES, MAX_KL)
    from heterogeneous_prefill_state import load as state_load
    require(baseline.resolve() != experiment.resolve(), 'capture prefixes must differ')
    b, e = read_capture(baseline), read_capture(experiment)
    identity = experiment_identity(e['marker'])
    compare_evidence(b, e, identity)
    check_log(baseline_log.read_text(errors='replace'), None)
    check_log(experiment_log.read_text(errors='replace'), identity)
    require(serial_state_signatures(baseline, state_load, signature) == serial_state_signatures(experiment, state_load, signature),
            'intervention changed serial CUDA checkpoint')
    cut, chunk = e['marker']['prefix_rows'], e['marker']['chunk']
    residuals = {}
    for key, old_digest in b['residuals'].items():
        pos = int(key.split('.')[1][1:])
        rows = min(chunk, cut-pos)
        values = []
        for prefix, digest in ((baseline, old_digest), (experiment, e['residuals'][key])):
            path = Path(f'{prefix}.native.{key}.f32')
            require(signature(path, rows*ROW_BYTES, 'f32') == digest, 'residual changed during analysis')
            values.append(np.fromfile(path, '<f4').reshape(rows, ROW_BYTES//4))
        residuals[key] = {'byte_exact': old_digest == e['residuals'][key],
                          'rows': [metric(a, z) for a, z in zip(*values)]}
    states = {s: [k for k,v in old['sha256'].items() if v != e['states'][s]['sha256'][k]]
              for s, old in b['states'].items()}
    require((reference is None) == (reference_log is None), 'reference prefix and log must be supplied together')
    interop = 'NOT_RUN'
    if reference is not None:
        check_log(reference_log.read_text(errors='replace'), identity)
        interop = pair(experiment, reference)['interop']
    return {'status': 'DIAGNOSTIC_ONLY' if interop != 'FAIL' else 'REPLAY_MISMATCH',
            'case': [cut, chunk], 'transport': e['marker']['transport'], 'hc_experiment': identity,
            'serial_controls_exact': True, 'intervention_wire_reference': interop,
            'kl_threshold': MAX_KL, 'original_quality': b['quality'], 'intervention_quality': e['quality'],
            'native_residual_change': residuals, 'native_state_changed_fields': states,
            'production_admission': 'NOT_QUALIFIED', 'bulk_mtp': 'NOT_ENABLED'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('baseline', 'experiment', 'baseline-log', 'experiment-log'):
        p.add_argument('--'+key, required=True, type=Path)
    p.add_argument('--reference', type=Path)
    p.add_argument('--reference-log', type=Path)
    a = p.parse_args()
    try:
        result = analyze(a.baseline, a.experiment, a.baseline_log, a.experiment_log, a.reference, a.reference_log)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 1 if result['status'] == 'REPLAY_MISMATCH' else 0
    except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError, OverflowError) as exc:
        print(f'INVALID: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
