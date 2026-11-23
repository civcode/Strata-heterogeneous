#!/usr/bin/env python3
"""HIP1-only kernels in steady decode (kernels HIP0 never runs in decode): the output head, the drafter, sampling."""
import sys, re, numpy as np
z = np.load(sys.argv[1], allow_pickle=True)
ag, st, en, kid, names = z['ag'], z['st'], z['en'], z['kid'], list(z['names'])
short = lambda n: re.sub(r'\(.*$', '', n.replace('strata::', '').replace('(anonymous namespace)::', '').replace('kernels::', '')).replace('void ', '')[:110]
pre = np.array([bool(re.search(r'prefill::|prompt_|Cijk_', n)) for n in names]); wh = np.array(['wait_handoff' in n for n in names])
BIN = 50_000_000; t0 = st.min(); b = (st - t0) // BIN; nb = int(b.max()) + 1
hw = np.zeros(nb, bool); hw[b[wh[kid]]] = True; hp = np.zeros(nb, bool); hp[b[pre[kid]]] = True
dec = hw & ~hp; d2 = dec.copy(); d2[1:-1] = dec[1:-1] & dec[:-2] & dec[2:]; sel = d2[b]; T = d2.sum() * BIN / 1e9
nwin = int((wh[kid] & sel).sum())
k0 = set(np.unique(kid[(ag == 1) & sel]).tolist())
m = (ag == 2) & sel; k = kid[m]; d = en[m] - st[m]
sums = np.bincount(k, weights=d, minlength=len(names)); cnts = np.bincount(k, minlength=len(names))
only = [i for i in np.argsort(-sums) if sums[i] > 0 and i not in k0]
tot = sum(sums[i] for i in only)
print(f'steady decode {T:.1f} s, {nwin} HIP0 windows; HIP1 kernels HIP0 does not run: {tot/1e9:.1f} s = {100*tot/1e9/T:.1f}% of decode time = {tot/1e6/nwin:.2f} ms per HIP0 window; HIP1 all kernels {d.sum()/1e9:.1f} s')
for i in only[:15]:
    print(f'  {sums[i]/1e9:6.2f} s {cnts[i]/nwin:5.2f}/win x {sums[i]/cnts[i]/1e3:7.1f} us  {short(names[i])}')
