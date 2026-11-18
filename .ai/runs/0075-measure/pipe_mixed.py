# handoff 0075 task 8: mixed.example.json, --prefill-pipe 700 against off on the 16K prompt, untraced, alternating,
# fresh prompts, one engine (per-request pipe= keys)
from common import *
argv = engine_argv('hetero/configs/mixed.example.json')
reqs = ['long16k@31:1:pipe=0+ckpt=0']
i = 32
for r in range(3):
    for v in ((0, 700) if r % 2 == 0 else (700, 0)):
        reqs.append(f'long16k@{i}:100:pipe={v}+ckpt=0'); i += 1
ok, _ = drive('pipe-mixed-ab16', argv, ','.join(reqs))
print('OK' if ok else 'STOP')
