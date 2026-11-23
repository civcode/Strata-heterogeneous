#!/usr/bin/env python3
"""Handoff 0095: a trace-driven model of vanilla's pipelined decode loop (--pipeline-windows 2) on a mixed split.

    plmodel.py PLTRACE [PARAMS.json]

The verdict sequence (on the path / miss with a speculative window launched / no speculative window) is taken from a
real STRATA_PIPELINE_TRACE; the times are the model's.  Two windows are in flight at most: window n+2 is made by the
drafter's chain launched at the verdict of window n, so on the path

    period ~ (c1 + S0 + h + s1) / 2   (when that exceeds s1),   fresh window ~ c2a + S0 + h + s1

where S0 is the stage-0 path (front stage + first local stage, host view), s1 the last stage (with the head), c1 the
chain's time to the next window's decision, c2a the chain's time to a fresh window's decision.
"""
import json, sys

BASE = dict(
    c1=5.08,     # verdict -> speculative launch on stage 0 after an on-path verdict (trace: V_to_L0_spec_after_1)
    cB2=4.35,    # verdict -> B decided after a fresh chain (trace: c2_CL_CD + bookkeeping)
    c2a=1.99,    # verdict -> fresh launch on stage 0 (trace: V_to_L0_fresh_after_0)
    S0=11.1,     # stage-0 path, launch -> done (trace: L0_F0)
    S0d=9.0,     # a doomed speculative window's stage-0 path (doom-skip: HIP0 gets empty plans), fitted
    S0_fresh_extra=0.0,  # added to S0 of a fresh window (A: the chain's tail on the front card)
    S0_b2_extra=0.0,     # added to S0 of a speculative window made by a fresh chain (A: the chain's tail)
    u=0.10,      # undo -> fresh launch
    h=0.08,      # stage 0 done -> last-stage launch
    delta=0.10,  # verdict -> last-stage launch of a ready window
    s1=9.35,     # last stage, launch -> verdict (trace: L1_F1)
)


def flags_from_trace(path):
    reqs, cur = [], []
    for line in open(path):
        if line.startswith('#'):
            if cur:
                reqs.append(cur)
            cur = []
            continue
        p = line.split()
        if p[1] == 'V':
            cur.append(int(p[4]))
    if cur:
        reqs.append(cur)
    return reqs


def simulate(reqs, P):
    sums = {'fresh': [0.0, 0], 'spec': [0.0, 0]}
    idle1 = [0.0, 0]
    for f in reqs:
        n = len(f)
        V = [0.0] * n
        L0 = [None] * (n + 2)
        F0 = [None] * (n + 2)
        # window 0: fresh from t=0
        L0[0] = 0.0
        F0[0] = L0[0] + P['S0'] + P['S0_fresh_extra']
        V[0] = F0[0] + P['h'] + P['s1']
        Dend = None
        # B made by the chain at window 0's start (kind 2)
        def make_b(k, t_chain, kind, a_f0):
            """B (window k) made by a chain launched at t_chain; launched on stage 0 after window k-1 leaves it."""
            c = P['c1'] if kind == 1 else P['cB2']
            L0[k] = max(t_chain + c, a_f0)
            F0[k] = L0[k] + P['S0'] + (P['S0_b2_extra'] if kind == 2 else 0.0)
        if n > 1 and f[0] != 0:
            make_b(1, 0.0 - P['c2a'], 2, F0[0])
        for m in range(n - 1):            # verdict of window m decides window m+1
            k = m + 1
            if f[m] == 1:                 # window k was launched speculatively and is on the path
                L1 = max(V[m] + P['delta'], F0[k] + P['h'])
                idle1[0] += L1 - V[m]
                idle1[1] += 1
                V[k] = L1 + P['s1']
                sums['spec'][0] += V[k] - V[m]
                sums['spec'][1] += 1
                kind = 1
            else:                         # fresh window k
                t = V[m] + P['c2a']
                if f[m] == -1 and Dend is not None:
                    t = max(t, Dend + P['u'])
                L0[k] = t
                F0[k] = t + P['S0'] + P['S0_fresh_extra']
                V[k] = F0[k] + P['h'] + P['s1']
                sums['fresh'][0] += V[k] - V[m]
                sums['fresh'][1] += 1
                kind = 2
            Dend = None
            # the chain at V[m] (kind 1) or V[m]+... (kind 2: the same chain made window k) makes window k+1
            if k < n and f[k] != 0:
                make_b(k + 1, V[m], kind, F0[k])
                if f[k] == -1:            # it will be the doomed window D
                    Dend = L0[k + 1] + P['S0d']
                    L0[k + 1] = F0[k + 1] = None
    r = {c: round(s / max(1, k), 2) for c, (s, k) in sums.items()}
    r['n_fresh'], r['n_spec'] = sums['fresh'][1], sums['spec'][1]
    r['last_idle_on_path'] = round(idle1[0] / max(1, idle1[1]), 2)
    tot = sums['fresh'][0] + sums['spec'][0]
    r['mean_window'] = round(tot / max(1, sums['fresh'][1] + sums['spec'][1]), 2)
    return r


def main():
    reqs = flags_from_trace(sys.argv[1])
    P = dict(BASE)
    if len(sys.argv) > 2:
        P.update(json.load(open(sys.argv[2])))
    print(json.dumps(simulate(reqs, P)))


if __name__ == '__main__':
    main()
