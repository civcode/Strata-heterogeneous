#!/usr/bin/env python3
"""One accounted engine run over the serve protocol, from a run config (mixed_split/configs/*.json).

    run_engine.py RUN_DIR NAME CONFIG REQUESTS [--env K=V ...] [--add ARG ...] [--drop OPT ...] [--no-check]

The config's engine and arguments (placeholders filled as serve_mixed.py does, its `env` and `lib_dirs` applied,
`layer_split` passed as --layer-split) are run through engine_drive.py with RUN_DIR/prompts.json (mkprompts.py) and
REQUESTS (engine_drive's syntax).  --add appends one argument, --drop removes an option and its value.  The launch
check (check_launch.py) runs first unless --no-check (only for a build from the commit before the working tree).
Writes RUN_DIR/NAME.json / .err / .driver.log / .kernel-delta.txt and adds the wall time to RUN_DIR/gpu-seconds.txt.
Exit status 0 only when the engine ended cleanly, under 120 s, with no new GPU fault line in the kernel log.
"""
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from mixed_split_common import expand  # noqa: E402
from check_launch import check_config  # noqa: E402

FAULTS = ('xid', 'gfxhub', 'gcvm', 'gpu fault', 'page fault', 'amdgpu: ring', 'gpu reset')
DEFAULTS = dict(STRATA_DATA='/home/chris/workspace/Strata-data', STRATA_MODEL_DIR='/home/chris/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF',
                STRATA_PROFILE='/home/chris/workspace/Strata/data/expert-profile.bin', STRATA_VANILLA='/home/chris/workspace/Strata',
                ROCM_LIB='/opt/rocm/core-10.1/lib')


def kernel():
    return subprocess.check_output(['journalctl', '-k', '-b', '--no-pager'], text=True, timeout=30)


def main():
    a = sys.argv[1:]
    if len(a) < 4:
        raise SystemExit(__doc__)
    run, name, config, reqs = a[:4]
    env_kv, add, drop, check = [], [], set(), True
    i = 4
    while i < len(a):
        if a[i] == '--env': env_kv.append(a[i + 1]); i += 2
        elif a[i] == '--add': add.append(a[i + 1]); i += 2
        elif a[i] == '--drop': drop.add(a[i + 1]); i += 2
        elif a[i] == '--no-check': check = False; i += 1
        else: raise SystemExit('unknown option ' + a[i])
    run = os.path.abspath(run)
    env = dict(DEFAULTS, **os.environ)
    names = {k: env[k] for k in DEFAULTS}
    names.update(STRATA_REPO=REPO, RUN_DIR=run)
    cfg = expand(json.load(open(config)), names)
    if check:
        check_config(cfg)
    args = [str(x) for x in cfg['args']]
    out, j = [], 0
    while j < len(args):
        if args[j] in drop:
            j += 2
            continue
        out.append(args[j])
        j += 1
    if cfg.get('layer_split'):
        out += ['--layer-split', str(cfg['layer_split'])]
    argv = [cfg['exe'], '--serve'] + out + add
    env.update({k: str(v) for k, v in cfg.get('env', {}).items()})
    env.setdefault('HSA_ENABLE_SDMA', '0')
    if cfg.get('gpu') and cfg.get('backend') == 'hip':
        env['HIP_VISIBLE_DEVICES'] = ','.join(str(g) for g in cfg['gpu'])
    if cfg.get('lib_dirs'):
        env['LD_LIBRARY_PATH'] = ':'.join(cfg['lib_dirs'] + ([env['LD_LIBRARY_PATH']] if env.get('LD_LIBRARY_PATH') else []))
    for kv in env_kv:
        k, v = kv.split('=', 1)
        env[k] = v
    own = [sys.executable, os.path.join(HERE, 'engine_drive.py'), os.path.join(run, name), os.path.join(run, 'prompts.json'), reqs]
    if '--front-exe' in argv and '--mmap-experts' not in argv:
        own.append('--shared-arena')
    base = kernel()
    t0 = time.monotonic()
    with open(os.path.join(run, name + '.driver.log'), 'w') as log:
        rc = subprocess.call(own + ['--'] + argv, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=REPO, timeout=170)
    el = time.monotonic() - t0
    tot_p = os.path.join(run, 'gpu-seconds.txt')
    tot = (float(open(tot_p).read()) if os.path.exists(tot_p) else 0.0) + el
    open(tot_p, 'w').write('%.1f\n' % tot)
    now = kernel()
    delta = now[len(base):] if now.startswith(base) else now
    open(os.path.join(run, name + '.kernel-delta.txt'), 'w').write(delta)
    bad = [x for x in delta.splitlines() if any(s in x.lower() for s in FAULTS)]
    print('%s: %.1f s, GPU total %.1f s, fault lines %d' % (name, el, tot, len(bad)), flush=True)
    for x in bad:
        print('  ', x, flush=True)
    res_p = os.path.join(run, name + '.json')
    res = json.load(open(res_p)) if os.path.exists(res_p) else {}
    for r in res.get('requests', []):
        print('  %-12s %-28s prompt %6d reused %s prefill %8.1f decode %6.1f drafts %s %s' % (
            r['name'], r.get('keys', ''), r['prompt'], r.get('reused'), r.get('prefill_tps') or 0, r.get('decode_tps') or 0,
            r.get('drafts'), r['done'][:80] if not r['done'].startswith('DONE') else ''), flush=True)
    sys.exit(0 if rc == 0 and res.get('rc') == 0 and el < 120 and not bad else 1)


if __name__ == '__main__':
    main()
