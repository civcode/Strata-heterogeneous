# handoff 0075 task 4: --prefill-pipe on HIP-only (hiponly.example.json, the engine over the serve protocol, fresh prompts,
# per-request `pipe=` keys as tools/calibrate.py sends them).  pipe.py sweep | ab16 | ablong
import sys
from common import *
argv = engine_argv('hetero/configs/hiponly.example.json')
what = sys.argv[1]
n = [0]
def fresh(kind, mx, v):
    i = n[0]; n[0] += 1
    return f'{kind}@{i}:{mx}:pipe={v}+ckpt=0'
if what == 'sweep':
    reqs = [fresh('long', 1, 0)]                       # the first long read after a start faults the weights in
    for v in (0, 1, 700, 2000, 5000):
        reqs += [fresh('long16k', 1, v), fresh('long', 1, v)]
else:
    R = int(sys.argv[2]); kind, mx = ('long16k', 100) if what == 'ab16' else ('long', 200)
    n[0] = 12 if what == 'ab16' else 24
    reqs = [fresh(kind, 1, 0)]
    for r in range(3):
        for v in ((0, R) if r % 2 == 0 else (R, 0)): reqs.append(fresh(kind, mx, v))
ok, _ = drive(f'pipe-{what}', argv, ','.join(reqs))
print('OK' if ok else 'STOP')
