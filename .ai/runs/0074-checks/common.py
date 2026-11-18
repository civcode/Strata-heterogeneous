# handoff 0074: shared helpers for the run scripts (run from the checkout)
import json, os, subprocess, sys, time
from pathlib import Path
ROOT = Path.cwd(); RUN = ROOT / '.ai/runs/0074-checks'
sys.path.insert(0, str(ROOT / 'hetero/tools'))
from serve_mixed import expand, REPO  # noqa: E402
ENV = dict(os.environ, STRATA_DATA='/home/chris/workspace/Strata-data', STRATA_MODEL_DIR='/home/chris/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF',
           STRATA_PROFILE='/home/chris/workspace/Strata/data/expert-profile.bin', STRATA_VANILLA='/home/chris/workspace/Strata',
           ROCM_LIB='/opt/rocm/core-10.1/lib', HSA_ENABLE_SDMA='0', HIP_VISIBLE_DEVICES='0,1')
os.environ.update(ENV)
TOTAL = RUN / 'gpu-seconds.txt'
FAULTS = ('xid', 'gfxhub', 'gcvm', 'gpu fault', 'page fault', 'amdgpu: ring', 'gpu reset')

def engine_argv(config, drop=(), drop_flags=(), extra=()):
    cfg = expand(json.load(open(config)), {"STRATA_REPO": REPO, "RUN_DIR": str(RUN)})
    a = [str(x) for x in cfg['args']]; out = []; i = 0
    while i < len(a):
        if a[i] in drop: i += 2; continue
        if a[i] in drop_flags: i += 1; continue
        out.append(a[i]); i += 1
    ls = cfg.get('layer_split')
    return [cfg['exe'], '--serve'] + out + (['--layer-split', str(ls)] if ls else []) + list(extra)

def kernel():
    return subprocess.check_output(['journalctl', '-k', '-b', '--no-pager'], text=True)

def used():
    return float(TOTAL.read_text()) if TOTAL.exists() else 0.0

def account(name, seconds, base):
    tot = used() + seconds; TOTAL.write_text(f"{tot:.1f}\n")
    now = kernel(); delta = now[len(base):] if now.startswith(base) else now
    (RUN / f'{name}.kernel-delta.txt').write_text(delta)
    bad = [l for l in delta.splitlines() if any(s in l.lower() for s in FAULTS)]
    print(f"{name}: {seconds:.1f} s, total {tot:.1f} s, fault lines {len(bad)}", flush=True)
    for l in bad: print('  ', l, flush=True)
    return not bad
