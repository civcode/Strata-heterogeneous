# drive_run.py NAME CONFIG REQUESTS [--no-mtp] [--hiponly]: one direct engine run with drive.py, accounted
import subprocess, sys, time
from common import *
name, config, reqs = sys.argv[1:4]
drop = ('--mtp',) if '--no-mtp' in sys.argv else ()
if '--no-cvec' in sys.argv: drop += ('--control-vector-scaled', '--cvec-mode', '--cvec-dir')
extra = ['--layer-split', 'auto'] if '--hiponly' in sys.argv else []
argv = engine_argv(config, drop=drop)
if '--no-cvec' in sys.argv:
    i = argv.index('--control-vector-layer-range'); del argv[i:i + 3]
if '--range' in sys.argv:   # another --control-vector-layer-range
    j = sys.argv.index('--range'); i = argv.index('--control-vector-layer-range'); argv[i + 1:i + 3] = sys.argv[j + 1:j + 3]
if '--hiponly' in sys.argv:   # the config's layer_split is "auto": vanilla's own split
    i = argv.index('--layer-split'); del argv[i:i + 2]
    argv += extra
if '--cudaonly' in sys.argv:   # the whole model on the CUDA card (the front stage's build), vanilla's single-GPU path
    argv = engine_argv(config, drop=drop + ('--front-exe', '--front-layers', '--layer-split'))
    argv[0] = argv[0].replace('build-0071-hip', 'build-0071-cuda')
own = [sys.executable, str(RUN / 'drive.py'), str(RUN / name), str(RUN / 'prompts.json'), reqs] + (['--shared-arena'] if '--front-exe' in argv else [])
base = kernel(); t0 = time.monotonic()
rc = subprocess.call(own + ['--'] + argv, env=ENV)
ok = account(name, time.monotonic() - t0, base)
sys.exit(0 if ok and rc == 0 else 1)
