#!/usr/bin/env python3
"""Decode-only view of the rocprofv3 trace: 50 ms bins that hold a wait_handoff (a decode window on HIP0) and no
prompt-type kernel on either card.  Per card inside those bins: busy share, spin share (wait_handoff), kernel time by
kernel, the sub-100 us gaps between kernels (launch bubbles), and windows per second."""
import sys, re, collections
import numpy as np
z = np.load(sys.argv[1], allow_pickle=True)
ag, st, en, kid, names = z['ag'], z['st'], z['en'], z['kid'], list(z['names'])
def short(n):
    n = n.replace('strata::', '').replace('(anonymous namespace)::', '').replace('kernels::', '')
    return re.sub(r'\(.*$', '', n).replace('void ', '')[:100]
pre = np.array([bool(re.search(r'prefill::|prompt_|Cijk_', n)) for n in names])
wh = np.array(['wait_handoff' in n for n in names])
BIN = 50_000_000
t0 = st.min()
b = (st - t0) // BIN
nb = int(b.max()) + 1
has_wh = np.zeros(nb, bool); has_wh[b[wh[kid]]] = True
has_pre = np.zeros(nb, bool); has_pre[b[pre[kid]]] = True
dec = has_wh & ~has_pre
# a bin counts only when its neighbours are decode too (no edge effects)
dec2 = dec.copy(); dec2[1:-1] = dec[1:-1] & dec[:-2] & dec[2:]
sel = dec2[b]
T = dec2.sum() * BIN / 1e9
print(f'decode bins: {dec2.sum()} x 50 ms = {T:.1f} s of steady decode')
nwin = int((wh[kid] & sel).sum())
print(f'HIP0 windows (wait_handoff) in them: {nwin} = {nwin/T:.1f} windows/s = {1000*T/nwin:.2f} ms per window')
for a in (1, 2):
    m = (ag == a) & sel
    s, e, k = st[m], en[m], kid[m]
    o = np.argsort(s); s, e, k = s[o], e[o], k[o]
    d = (e - s)
    ce = np.maximum.accumulate(e); gaps = (s[1:] - ce[:-1])
    small = gaps[(gaps > 0) & (gaps < 100_000)].sum() / 1e9
    mid = gaps[(gaps >= 100_000) & (gaps < 1_000_000)].sum() / 1e9
    big = gaps[gaps >= 1_000_000].sum() / 1e9
    busy = d.sum() / 1e9            # kernels do not overlap within one card here (one stream at a time) - checked by union below
    union = T - np.clip(gaps, 0, None).sum() / 1e9
    spin = d[wh[k]].sum() / 1e9
    print(f'\n=== Agent {a} ({"HIP0" if a == 1 else "HIP1"}): {m.sum()} kernels = {m.sum()/max(nwin,1):.0f} per HIP0 window; '
          f'kernel time {busy:.1f} s = {100*busy/T:.1f}% of decode time, of it wait_handoff spin {spin:.1f} s ({100*spin/T:.1f}%); '
          f'real work {100*(busy-spin)/T:.1f}%')
    print(f'  gaps between kernels: <100 us {small:.2f} s ({100*small/T:.1f}%, mean {1e6*small/max(((gaps>0)&(gaps<100_000)).sum(),1):.1f} us x {((gaps>0)&(gaps<100_000)).sum()}), '
          f'0.1-1 ms {mid:.2f} s ({100*mid/T:.1f}%), >= 1 ms {big:.2f} s ({100*big/T:.1f}%)')
    sums = np.bincount(k, weights=d, minlength=len(names)); cnts = np.bincount(k, minlength=len(names))
    for i in np.argsort(-sums)[:14]:
        print(f'  {sums[i]/1e9:6.2f} s {100*sums[i]/1e9/T:5.1f}% of time {cnts[i]/max(nwin,1):6.1f}/win x {sums[i]/max(cnts[i],1)/1e3:7.1f} us  {short(names[i])}')
