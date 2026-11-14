#!/usr/bin/env python3
"""Read-only matrix gate for bounded native MTP; never launches workers."""
import argparse
import json
from pathlib import Path
import re
import sys

from heterogeneous_mtp_three_round import PROMPT, analyze, require

SCHEMA = 'strata-mtp-qualification/v1'
TRANSPORT = re.compile(r'^LIVE workers ready, mode=(shared|socket)(?: |$)', re.M)
MODES = {(transport, traced) for transport in ('shared', 'socket') for traced in (False, True)}


def read_transport(log, transport):
    require(TRANSPORT.findall(log) == [transport], 'missing, duplicate or mismatched worker transport')


def load_manifest(path):
    obj = json.loads(path.read_text())
    require(obj.get('schema') == SCHEMA, 'invalid matrix schema')
    cases = obj.get('cases')
    require(type(cases) is list and 2 <= len(cases) <= 4, 'expected two to four bounded prompt cases')
    seen_prompts, seen_paths = set(), set()
    result = []

    def distinct(raw):
        require(type(raw) is str and raw, 'missing capture or log path')
        p = (path.parent / raw).resolve()
        require(p not in seen_paths, 'capture/log path reused in matrix')
        seen_paths.add(p)
        return p

    for case in cases:
        prompt, runs = case['prompt_ids'], case['runs']
        require(type(prompt) is list and 1 <= len(prompt) <= 7 and
                all(type(t) is int and 0 <= t < 248320 for t in prompt), 'invalid bounded prompt')
        key = tuple(prompt)
        require(key not in seen_prompts, 'duplicate prompt case')
        seen_prompts.add(key)
        require(type(runs) is list and 4 <= len(runs) <= 8, 'expected four to eight matched sets per case')
        modes = set()
        parsed = []
        for run in runs:
            transport, traced = run['transport'], run['traced']
            require(type(traced) is bool and transport in ('shared', 'socket'), 'invalid run mode')
            modes.add((transport, traced))
            fresh = run['fresh']
            require(type(fresh) is list and len(fresh) == 4, 'four fresh checkpoint captures required')
            parsed.append((transport, traced, distinct(run['prefix']), distinct(run['log']),
                           [(distinct(f['prefix']), distinct(f['log'])) for f in fresh]))
        require(modes == MODES, 'each case needs trace-free and traced shared/socket runs')
        result.append((prompt, parsed))
    require(tuple(PROMPT) in seen_prompts and tuple(PROMPT[:-1]) in seen_prompts,
            'known acceptance and rejection-start prompts are required')
    return result


def exact(result):
    return result['status'] == 'PASS' and result['max_abs_logit_difference'] == 0


def identity(run):
    m, p = run[0]
    return tuple(m[k] for k in ('plan', 'model_fingerprint', 'pack_index_sha256', 'gguf_shards', 'profile_path')) + (p['fingerprint'], p['vocab'])


def compare_matrix(path):
    from heterogeneous_compare_parity import read_run, compare
    cases = load_manifest(path)
    coverage = {mode: [0, 0] for mode in MODES}
    patterns = {mode: set() for mode in MODES}
    source = None
    summaries = []
    all_pass = True
    for prompt, runs in cases:
        reference = None
        reference_offers = None
        for transport, traced, prefix, log_path, fresh in runs:
            log = log_path.read_text()
            read_transport(log, transport)
            fresh_pairs = [(p, l.read_text()) for p, l in fresh]
            for _, fresh_log in fresh_pairs:
                read_transport(fresh_log, transport)
            gate = analyze(prefix, log, fresh_pairs, traced=traced, prompt=prompt)
            reused = read_run(prefix, allow_partial=True, mtp_three_round=True)
            current = identity(reused)
            if source is None:
                source = current
            require(current == source, 'model/plan/source changed across matrix cases')
            offers = [(p['cell'], p['anchor'], p['candidates']) for p in gate['checkpoints']]
            if reference is None:
                reference, reference_offers = reused, offers
                repeated = None
                equivalent = True
            else:
                # Includes discarded rows; incompatible offers/windows are invalid evidence.
                repeated = compare(reference, reused, 0.001, prefill=True)
                equivalent = exact(repeated) and offers == reference_offers
            passed = gate['parity_status'] == 'PASS' and equivalent
            if prompt == PROMPT:
                passed = passed and gate['status'] == 'PASS'
            all_pass &= passed
            mode = (transport, traced)
            pattern = tuple(p['accepted_drafts'] for p in gate['checkpoints'][:-1])
            if passed:
                coverage[mode][0] += gate['accepted_drafts']
                coverage[mode][1] += gate['rejected_drafts']
                patterns[mode].add(pattern)
            summaries.append(dict(prompt_ids=prompt, transport=transport, traced=traced,
                                  prefix=str(prefix), status='PASS' if passed else 'FAIL',
                                  accepted_pattern=pattern, all_evaluated_parity=repeated,
                                  gate=gate))
    modes = [dict(transport=t, traced=b, accepted_drafts=coverage[t,b][0],
                  rejected_drafts=coverage[t,b][1], patterns=sorted(patterns[t,b]),
                  covered=all(coverage[t,b]) and len(patterns[t,b]) >= 2)
             for t,b in sorted(MODES)]
    return dict(status='PASS' if all_pass and all(m['covered'] for m in modes) else 'FAIL',
                scope='BOUNDED_REPLAY_MATRIX_ONLY', modes=modes, runs=summaries,
                production_admission='NOT QUALIFIED', resident_mtp='NOT QUALIFIED', physical_rollback='NOT MEASURED')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    args = parser.parse_args()
    try:
        result = compare_matrix(args.manifest)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result['status'] == 'PASS' else 1
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
        print(f'INVALID: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
