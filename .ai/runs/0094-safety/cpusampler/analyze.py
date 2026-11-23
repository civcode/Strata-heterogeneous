#!/usr/bin/env python3
"""Symbolise and summarise cpusampler output (samples-<pid>.bin, maps-<pid>.txt, threads-<pid>.txt).

    analyze.py DIR [--top 25] [--json OUT]

Per process and per thread group: samples (= CPU time at 499 Hz), the top functions by self time (the sampled pc)
and by inclusive time (anywhere on the stack, once per sample).  Threads are grouped by their start routine: the
outermost frame in the engine binary (or the outermost named frame).
"""
import argparse, bisect, collections, json, os, re, struct, subprocess, sys
from pathlib import Path

SYM = '/opt/rocm/llvm/bin/llvm-symbolizer'


def load_maps(path):
    maps = []
    for line in open(path):
        parts = line.split()
        if len(parts) < 6 or 'x' not in parts[1]:
            continue
        lo, hi = (int(x, 16) for x in parts[0].split('-'))
        maps.append((lo, hi, int(parts[2], 16), parts[5]))
    maps.sort()
    return maps


def read_samples(path):
    data = open(path, 'rb').read()
    i, out = 0, []
    while i + 8 <= len(data):
        tid, depth = struct.unpack_from('<II', data, i)
        i += 8
        pcs = struct.unpack_from(f'<{depth}Q', data, i)
        i += 8 * depth
        out.append((tid, pcs))
    return out


def symbolise(addrs_by_mod):
    """{module: set(file offsets)} -> {(module, off): name}; file offsets are converted to addresses by the module's
    load bias: llvm-symbolizer takes --adjust-vma? We pass virtual addresses computed from the ELF program headers."""
    names = {}
    for mod, offs in addrs_by_mod.items():
        if not os.path.exists(mod):
            for o in offs:
                names[(mod, o)] = f'[{os.path.basename(mod)}]'
            continue
        # file offset -> vaddr via PT_LOAD headers
        loads = []
        try:
            out = subprocess.run(['readelf', '-lW', mod], capture_output=True, text=True, timeout=60).stdout
            for l in out.splitlines():
                l = l.split()
                if l and l[0] == 'LOAD':
                    loads.append((int(l[1], 16), int(l[2], 16), int(l[4], 16)))   # offset, vaddr, filesz
        except Exception:  # noqa: BLE001
            pass
        offl = sorted(offs)
        va = []
        for o in offl:
            v = o
            for fo, vaddr, sz in loads:
                if fo <= o < fo + sz:
                    v = o - fo + vaddr
                    break
            va.append(v)
        inp = '\n'.join(hex(v) for v in va) + '\n'
        r = subprocess.run([SYM, '--obj', mod, '--functions=linkage', '--demangle', '--no-inlines', '--output-style=GNU'],
                           input=inp, capture_output=True, text=True, timeout=600)
        lines = r.stdout.splitlines()
        # GNU style: two lines per address (function, file:line)
        fn = lines[0::2]
        base = os.path.basename(mod)
        for o, f in zip(offl, fn):
            names[(mod, o)] = (f if f and f != '??' else f'??') + ('' if 'strata' in base else f' [{base}]')
    return names


def short(name, n=110):
    name = re.sub(r'\(.*\)( const)?$', '', name) if name.count('(') else name
    return name if len(name) <= n else name[: n - 3] + '...'


def analyze_pid(d, pid, top):
    maps = load_maps(d / f'maps-{pid}.txt')
    starts = [m[0] for m in maps]
    samples = read_samples(d / f'samples-{pid}.bin')
    hdr = open(d / f'threads-{pid}.txt').readline().strip()
    exe = hdr.split()[2] if hdr.startswith('#') else '?'

    def where(pc, leaf):
        a = pc if leaf else pc - 1
        k = bisect.bisect_right(starts, a) - 1
        if k >= 0 and maps[k][0] <= a < maps[k][1]:
            lo, hi, off, mod = maps[k]
            return (mod, a - lo + off)
        return ('?', a)
    locs = []
    by_mod = collections.defaultdict(set)
    for tid, pcs in samples:
        st = [where(pc, i == 0) for i, pc in enumerate(pcs)]
        locs.append((tid, st))
        for m, o in st:
            by_mod[m].add(o)
    names = symbolise(by_mod)

    def nm(loc):
        return names.get(loc, f'[{os.path.basename(loc[0])}]')

    # thread group: the outermost frame inside the engine binary (strata), else outermost named frame
    per_tid = collections.defaultdict(list)
    for tid, st in locs:
        per_tid[tid].append(st)
    group_of = {}
    for tid, sts in per_tid.items():
        votes = collections.Counter()
        for st in sts:
            eng = [nm(l) for l in st if 'strata' in os.path.basename(l[0]) and not nm(l).startswith('??')]
            outer = [x for x in eng if 'main' == short(x) or 'std::thread' in x or 'invoke' in x or 'lambda' in x]
            if eng:
                votes[short(eng[-1], 90)] += 1
            else:
                named = [nm(l) for l in st if not nm(l).startswith('??')]
                votes['(no engine frame) ' + short(named[-1], 70) if named else '(unknown)'] += 1
        group_of[tid] = votes.most_common(1)[0][0]
    res = dict(pid=pid, exe=exe, header=hdr, samples=len(samples), groups={})
    groups = collections.defaultdict(list)
    for tid, st in locs:
        groups[group_of[tid]].append(st)
    for g, sts in sorted(groups.items(), key=lambda x: -len(x[1])):
        selfc, incl = collections.Counter(), collections.Counter()
        for st in sts:
            selfc[short(nm(st[0]))] += 1
            for x in set(short(nm(l)) for l in st):
                incl[x] += 1
        res['groups'][g] = dict(samples=len(sts), cpu_s=round(len(sts) / 499, 1),
                                threads=sorted({t for t in per_tid if group_of[t] == g}).__len__(),
                                self=selfc.most_common(top), inclusive=incl.most_common(top))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir')
    ap.add_argument('--top', type=int, default=25)
    ap.add_argument('--json')
    a = ap.parse_args()
    d = Path(a.dir)
    out = []
    for f in sorted(d.glob('samples-*.bin')):
        pid = f.stem.split('-')[1]
        if f.stat().st_size == 0:
            continue
        r = analyze_pid(d, pid, a.top)
        out.append(r)
        print(f"\n=== pid {pid} {r['exe']}  samples {r['samples']} ({r['samples'] / 499:.1f} CPU s)\n{r['header']}")
        for g, v in r['groups'].items():
            print(f"\n--- group: {g}  threads {v['threads']}  samples {v['samples']} ({v['cpu_s']} CPU s)")
            print('  self:')
            for n, c in v['self'][:a.top]:
                print(f'    {100 * c / v["samples"]:5.1f}%  {n}')
            print('  inclusive:')
            for n, c in v['inclusive'][:a.top]:
                print(f'    {100 * c / v["samples"]:5.1f}%  {n}')
    if a.json:
        json.dump(out, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
