#!/usr/bin/env python3
"""Per thread: CPU samples attributed to the innermost frame inside the engine binary (its own time plus the library
calls it makes), and the share of samples whose leaf is a wait/poll function.  analyze.py's loaders."""
import collections, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import analyze as A

WAIT = ('WaitRelaxed', 'sched_yield', 'hipEventQuery', 'cuEventQuery', 'cudaEventQuery', 'clock_gettime', '[[vdso]]',
        'steady_clock::now', 'tls_get_addr', 'hipSetDevice', 'hipGetDevice', 'cudaGetDevice', 'lll_mutex', 'pthread_mutex',
        'rwlock', 'hsa_signal_load', 'LoadRelaxed', 'cuCtxGetDevice', 'AsyncEventsLoop', 'libcuda.so', 'libamdhip64')

def main(d, top=25):
    d = Path(d)
    for f in sorted(d.glob('samples-*.bin')):
        if f.stat().st_size == 0:
            continue
        pid = f.stem.split('-')[1]
        maps = A.load_maps(d / f'maps-{pid}.txt'); starts = [m[0] for m in maps]
        import bisect
        def where(pc, leaf):
            a = pc if leaf else pc - 1
            k = bisect.bisect_right(starts, a) - 1
            if k >= 0 and maps[k][0] <= a < maps[k][1]:
                lo, hi, off, mod = maps[k]; return (mod, a - lo + off)
            return ('?', a)
        samples = A.read_samples(f)
        locs = [(t, [where(pc, i == 0) for i, pc in enumerate(p)]) for t, p in samples]
        by = collections.defaultdict(set)
        for _, st in locs:
            for m, o in st: by[m].add(o)
        names = A.symbolise(by)
        nm = lambda l: names.get(l, '?')
        per = collections.defaultdict(list)
        for t, st in locs: per[t].append(st)
        print(f'\n=== pid {pid}', open(d / f'threads-{pid}.txt').readline().split()[2])
        for t, sts in sorted(per.items(), key=lambda x: -len(x[1]))[:4]:
            c = collections.Counter(); w = 0
            for st in sts:
                eng = [A.short(nm(l), 120) for l in st if 'strata' in os.path.basename(l[0])]
                c[eng[0] if eng else '(no engine frame)'] += 1
                leaf = nm(st[0])
                w += any(x in leaf for x in WAIT)
            print(f'--- tid {t}: {len(sts)} samples ({len(sts)/499:.1f} CPU s); leaf in a wait/poll/runtime-query function: {100*w/len(sts):.1f}%')
            for n, k in c.most_common(top):
                print(f'   {100*k/len(sts):5.1f}%  {n}')

if __name__ == '__main__':
    main(sys.argv[1])
