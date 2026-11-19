#!/usr/bin/env python3
"""The summary of agent_session.py runs (`.ai/RC.md`), one column per run directory.

    agent_table.py RUN_DIR [RUN_DIR ...] [--bucket 32768] [--markdown OUT.md]

Prints, per run: totals; decode and prefill tok/s (medians) and time to first token by context bucket for tool and
answer requests; each compaction; probe replay results (probes/*-replay.json); resources from samples.jsonl (VRAM per
card and available RAM at the end of cycle 0 and at the end, swap). With two or more runs it also says whether their
request sequences are identical (the SHA-256 column of requests.jsonl).
"""
import argparse, glob, json, os, statistics, sys


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 1) if xs else None


def load(d):
    rows = [json.loads(l) for l in open(os.path.join(d, 'requests.jsonl'))]
    sess = json.load(open(os.path.join(d, 'session.json'))) if os.path.exists(os.path.join(d, 'session.json')) else {}
    samples = []
    if os.path.exists(os.path.join(d, 'samples.jsonl')):
        samples = [json.loads(l) for l in open(os.path.join(d, 'samples.jsonl')) if l.strip()]
    replays = {}
    for f in sorted(glob.glob(os.path.join(d, 'probes', '*-replay.json'))):
        r = json.load(open(f))
        verdict = r.get('verdict')
        replays[r['why']] = (verdict.get('verdict') if isinstance(verdict, dict) else verdict) or (
            'equal' if r['equal'] else 'DIFFERENT')
    return dict(name=os.path.basename(os.path.normpath(d)), rows=rows, sess=sess, samples=samples, replays=replays)


def table(head, lines):
    out = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    out += ['| ' + ' | '.join('' if v is None else str(v) for v in l) + ' |' for l in lines]
    return '\n'.join(out)


def vram(s):
    v = {}
    for k in ('cuda_vram_mib', 'hip_vram_mib'):
        for card, mib in (s.get(k) or {}).items():
            v[f'{k[:3]}{card}'] = mib
    return v


def summary(runs, bucket):
    parts = []
    # totals
    keys = ('requests', 'prompt_tokens', 'cached_tokens', 'completion_tokens', 'prunes', 'pruned_outputs', 'compactions',
            'cut_outputs', 'errors', 'probes')
    lines = [[k] + [r['sess'].get('totals', {}).get(k) for r in runs] for k in keys]
    lines.append(['wall time (min)'] + [round(r['sess'].get('wall_s', 0) / 60, 1) for r in runs])
    lines.append(['new tokens read'] + [sum((x['prompt_tokens'] or 0) - (x['cached_tokens'] or 0) for x in r['rows'])
                                        for r in runs])
    parts.append('### Totals\n\n' + table(['', *[r['name'] for r in runs]], lines))
    if len(runs) > 1:
        same = all([x['sha256'] for x in r['rows']] == [x['sha256'] for x in runs[0]['rows']] for r in runs[1:])
        common = min(len(r['rows']) for r in runs)
        first_diff = next((i for i in range(common) if len({r['rows'][i]['sha256'] for r in runs}) > 1), None)
        parts.append(f'Request sequences identical: **{"yes" if same else "no"}**'
                     + ('' if same else f' (first difference at request {first_diff + 1 if first_diff is not None else common + 1})'))
    # by bucket
    top = max((x['prompt_tokens'] or 0) for r in runs for x in r['rows'])
    lines = []
    for lo in range(0, top + 1, bucket):
        line = [f'{lo // 1024}K-{(lo + bucket) // 1024}K']
        for r in runs:
            sel = [x for x in r['rows'] if x['kind'] in ('tool', 'answer') and lo <= (x['prompt_tokens'] or 0) < lo + bucket]
            line += [len(sel), med(x['decode_tok_s'] for x in sel), med(x['prompt_tok_s'] for x in sel),
                     med(x['ttft_s'] for x in sel)]
        lines.append(line)
    head = ['context'] + [f'{r["name"]} {c}' for r in runs for c in ('n', 'decode tok/s', 'prefill tok/s', 'ttft s')]
    parts.append(f'### Tool and answer requests by context (medians)\n\n' + table(head, lines))
    # compactions
    lines = []
    for r in runs:
        for x in r['rows']:
            if x['kind'] == 'compaction':
                lines.append([r['name'], x['cycle'], x['prompt_tokens'], x['cached_tokens'], x['ttft_s'], x['decode_tok_s'],
                              x['completion_tokens'], x['wall_s']])
    parts.append('### Compactions\n\n' + table(['run', 'cycle', 'prompt', 'cached', 'ttft s', 'decode tok/s', 'generated',
                                                'wall s'], lines))
    # probes
    whys = sorted({w for r in runs for w in r['replays']} | {x['kind'][6:] for r in runs for x in r['rows']
                                                              if x['kind'].startswith('probe-')})
    lines = [[w] + [r['replays'].get(w, 'not replayed') for r in runs] for w in whys]
    parts.append('### Probes (fresh-server replay)\n\n' + table(['probe', *[r['name'] for r in runs]], lines))
    # resources
    lines = []
    for r in runs:
        s = r['samples']
        if not s:
            lines.append([r['name'], 'no samples'] + [None] * 4)
            continue
        comp = next((x for x in r['rows'] if x['cycle'] == 0 and x['kind'] == 'compaction' and not x.get('error')), None)
        c0_end = comp['t_start'] + (comp.get('wall_s') or 0) if comp else None
        base = next((x for x in s if c0_end and x['t'] >= c0_end), None)
        if base is None:
            avail = [x.get('mem_available_mib') for x in s if x.get('mem_available_mib') is not None]
            lines.append([r['name'], 'first cycle incomplete / no post-cycle sample', None,
                          s[-1].get('mem_available_mib'), min(avail) if avail else None,
                          max((x.get('swap_used_mib') or 0) for x in s)])
            continue
        end = s[-1]
        vb, ve = vram(base), vram(end)
        growth = {k: round(100 * (ve[k] - vb[k]) / vb[k], 2) for k in vb if k in ve and vb[k]}
        avail = [x.get('mem_available_mib') for x in s if x.get('mem_available_mib') is not None]
        lines.append([r['name'], json.dumps(growth), base.get('mem_available_mib'), end.get('mem_available_mib'),
                      min(avail) if avail else None, max((x.get('swap_used_mib') or 0) for x in s)])
    parts.append('### Resources (after cycle 0 to the end)\n\n' + table(
        ['run', 'VRAM growth % per card', 'RAM available after cycle 0 (MiB)', 'at the end', 'lowest', 'swap max (MiB)'],
        lines))
    errs = [(r['name'], x['n'], x['error']) for r in runs for x in r['rows'] if x.get('error')]
    parts.append('### Errors\n\n' + ('none' if not errs else '\n'.join(f'- {n} #{i}: {e}' for n, i, e in errs)))
    return '\n\n'.join(parts) + '\n'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+')
    p.add_argument('--bucket', type=int, default=32768)
    p.add_argument('--markdown')
    a = p.parse_args()
    text = summary([load(d) for d in a.runs], a.bucket)
    if a.markdown:
        open(a.markdown, 'w').write(text)
    print(text)


if __name__ == '__main__':
    sys.exit(main())
