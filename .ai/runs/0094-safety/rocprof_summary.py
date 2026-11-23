#!/usr/bin/env python3
"""Summarise a rocprofv3 kernel trace (CSV) per agent: busy time, top kernels, dispatch counts, idle gaps.
    rocprof_summary.py run_kernel_trace.csv OUT.npz-cache"""
import csv, sys, collections, re, os
import numpy as np

src, cache = sys.argv[1], sys.argv[2]
if not os.path.exists(cache):
    names = {}; ag, st, en, kid = [], [], [], []
    with open(src, newline='') as f:
        r = csv.reader(f); hdr = next(r)
        ia, ik, isx, iex = hdr.index('Agent_Id'), hdr.index('Kernel_Name'), hdr.index('Start_Timestamp'), hdr.index('End_Timestamp')
        for row in r:
            k = names.setdefault(row[ik], len(names))
            ag.append(int(row[ia].split()[-1])); kid.append(k); st.append(int(row[isx])); en.append(int(row[iex]))
    inv = [None] * len(names)
    for n, i in names.items(): inv[i] = n
    np.savez(cache, ag=np.array(ag, np.int8), st=np.array(st, np.int64), en=np.array(en, np.int64), kid=np.array(kid, np.int32), names=np.array(inv, dtype=object))
z = np.load(cache, allow_pickle=True)
ag, st, en, kid, names = z['ag'], z['st'], z['en'], z['kid'], list(z['names'])

def short(n):
    n = n.replace('strata::', '').replace('(anonymous namespace)::', '').replace('kernels::', '')
    n = re.sub(r'\(.*$', '', n)
    n = re.sub(r'^void ', '', n)
    return n[:110]

def is_prefill(n):
    return bool(re.search(r'prefill::|prompt_|Cijk_|gr_write_norm|gr_mix|w11|rocclr|MtpDrafter::prefill', n))

t0 = st.min()
for a in sorted(set(ag.tolist())):
    m = ag == a
    s, e, k = st[m], en[m], kid[m]
    o = np.argsort(s); s, e, k = s[o], e[o], k[o]
    span = (e.max() - s.min()) / 1e9
    # busy = union of intervals
    ce = np.maximum.accumulate(e)
    gaps = s[1:] - ce[:-1]
    busy = span - gaps[gaps > 0].sum() / 1e9
    pre = np.array([is_prefill(names[i]) for i in range(len(names))])
    kp = pre[k]
    print(f'\n=== Agent {a}: {m.sum()} dispatches over {span:.1f} s; busy (union) {busy:.1f} s = {100*busy/span:.1f}%')
    d = (e - s)
    sums = np.bincount(k, weights=d, minlength=len(names)); cnts = np.bincount(k, minlength=len(names))
    order = np.argsort(-sums)
    alltime = d.sum() / 1e9
    print(f'  kernel time sum {alltime:.1f} s; decode-type kernels {d[~kp].sum()/1e9:.1f} s ({(~kp).sum()} dispatches), prompt-type {d[kp].sum()/1e9:.1f} s ({kp.sum()})')
    for i in order[:18]:
        if sums[i] == 0: break
        print(f'  {sums[i]/1e9:7.2f} s {100*sums[i]/1e9/alltime:5.1f}%  {cnts[i]:9d} x {sums[i]/cnts[i]/1e3:9.1f} us  {"P" if pre[i] else "D"} {short(names[i])}')
    # gaps between consecutive kernels (after the running max end), decode-type neighbourhood only
    g = gaps / 1e3
    pos = g[g > 0]
    print('  idle gaps (us) between kernels: count by size: ' + ', '.join(f'{lo}-{hi}: {((pos>=lo)&(pos<hi)).sum()} ({pos[(pos>=lo)&(pos<hi)].sum()/1e6:.1f} s)'
          for lo, hi in ((0, 10), (10, 100), (100, 1000), (1000, 10000), (10000, 100000), (100000, 1e12))))
    wh = [i for i in range(len(names)) if 'wait_handoff' in names[i]]
    for i in wh:
        print(f'  wait_handoff on this agent: {cnts[i]} x {sums[i]/max(cnts[i],1)/1e3:.1f} us = {sums[i]/1e9:.1f} s')
    # candidate once-per-window kernels: decode-type, counts
    cand = [(cnts[i], short(names[i])) for i in range(len(names)) if cnts[i] and not pre[i] and re.search(r'argmax|sample|head|logit|wait_handoff|signal|ready', names[i])]
    for c, n in sorted(cand, reverse=True)[:8]:
        print(f'    count {c:8d}  {n}')
