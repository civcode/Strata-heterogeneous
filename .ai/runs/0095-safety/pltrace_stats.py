#!/usr/bin/env python3
"""Handoff 0095: intervals of the pipelined decode loop from STRATA_PIPELINE_TRACE (host clock, ms).

    pltrace_stats.py PLTRACE [--json OUT]

Events (generate.cpp, tre()): L0 seq T st (st>=10: speculative launch on stage 0), F0 seq fr spec (stage 0 done),
L1 seq T, F1 seq T (last stage done), V seq a flag (1 on the path, -1 miss with B launched, 0 no B), CL seq kind
(1: chain over A forced through B, 2: fresh), CE seq T (fresh A decided), CD seq kind p_on (B decided), U seq keep.
"""
import json, statistics as S, sys
from collections import defaultdict


def q(v):
    if not v:
        return None
    v = sorted(v)
    return dict(n=len(v), mean=round(S.fmean(v), 3), med=round(v[len(v) // 2], 3), p10=round(v[len(v) // 10], 3),
                p90=round(v[(9 * len(v)) // 10], 3))


def main():
    reqs, cur = [], []
    for line in open(sys.argv[1]):
        if line.startswith('#'):
            if cur:
                reqs.append(cur)
            cur = []
            continue
        t, ev, *x = line.split()
        cur.append((float(t), ev, *map(int, x)))
    if cur:
        reqs.append(cur)
    st = defaultdict(list)
    for evs in reqs:
        last_v = None          # (t, flag)
        chain = None           # (t, kind)
        l0 = {}                # seq -> (t, spec)
        l1 = {}
        f0_t = {}
        chain_open = []        # [start, end] of chains
        cstart = None
        for t, ev, *x in evs:
            if ev == 'CL':
                chain = (t, x[1])
                cstart = t
            elif ev == 'CE' and chain:
                st['c2_CL_CE'].append(t - chain[0])
            elif ev == 'CD' and chain:
                st[f'c{chain[1]}_CL_CD'].append(t - chain[0])
                chain_open.append((cstart, t))
            elif ev == 'L0':
                spec = x[2] >= 10
                l0[x[0]] = (t, spec)
                if last_v is not None:
                    st[f'V_to_L0_{"spec" if spec else "fresh"}_after_{last_v[1]}'].append(t - last_v[0])
            elif ev == 'F0' and x[0] in l0:
                t0, spec = l0[x[0]]
                f0_t[x[0]] = t
                st['L0_F0_' + ('spec' if spec else 'fresh')].append(t - t0)
            elif ev == 'L1':
                l1[x[0]] = t
                if x[0] in f0_t:
                    st['F0_to_L1'].append(t - f0_t[x[0]])
            elif ev == 'F1' and x[0] in l1:
                d = t - l1[x[0]]
                # did a chain run while this window was on the last card? (CL after L1 or chain still open)
                overl = chain is not None and chain[0] >= l1[x[0]] - 0.05
                st['L1_F1_' + ('chain' if overl else 'nochain')].append(d)
            elif ev == 'U' and last_v is not None:
                st['V_to_U'].append(t - last_v[0])
            elif ev == 'V':
                if last_v is not None:
                    st[f'V_V_prev{last_v[1]}'].append(t - last_v[0])
                last_v = (t, x[2])
    out = {k: q(v) for k, v in sorted(st.items())}
    for k, v in out.items():
        print(f'{k:28s} {v}')
    if len(sys.argv) > 3 and sys.argv[2] == '--json':
        json.dump(out, open(sys.argv[3], 'w'), indent=1)


if __name__ == '__main__':
    main()
