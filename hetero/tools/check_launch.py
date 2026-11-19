#!/usr/bin/env python3
"""Reject launch configurations whose executables or libraries mix builds."""
import json
import os
from pathlib import Path
import subprocess
import sys

from hetero_common import expand

REPO = Path(__file__).resolve().parents[2]
VISION = REPO / 'terminal' / 'tools' / 'vision'
ENGINE = REPO / 'engine'
# what does not go into a build: notes, reports, run logs, docs, benchmark write-ups, and the mixed split's driver
# scripts and run configs.  A commit that changes only
# these does not make a build stale (rule 7 is about the code that runs).
NOT_BUILT = [':!.ai', ':!runs*', ':!docs', ':!bench', ':!*.md',
             ':!hetero/tools/*.py', ':!hetero/tools/*.sh', ':!hetero/configs']   # drivers and run configs: not built


def git(*args):
    return subprocess.run(['git', '-C', str(REPO), *args], text=True, capture_output=True, timeout=30)


def source_changed(since, until=None):
    """True when the built sources differ between commit `since` and `until` (None: the working tree)."""
    r = git('diff', '--quiet', since, *([until] if until else []), '--', '.', *NOT_BUILT)
    if r.returncode not in (0, 1):
        fail(f'BUILD_COMMIT {since} is not a commit of this checkout')
    return r.returncode == 1


def fail(message):
    raise ValueError(message)


def check_config(cfg):
    head = git('rev-parse', 'HEAD').stdout.strip()
    if source_changed('HEAD'):
        fail('the checkout has uncommitted source changes: commit or stash them, rebuild, and stamp the builds')
    cwd = Path(cfg.get('cwd', REPO)).resolve()
    args = list(map(str, cfg.get('args', [])))
    if any(a.startswith('--front-exe=') for a in args):
        fail('use separate --front-exe and path arguments')
    fronts = [i for i, a in enumerate(args) if a == '--front-exe']
    if len(fronts) > 1 or (fronts and fronts[0] + 1 == len(args)):
        fail('invalid --front-exe')
    executables = [('host', cfg.get('exe'))]
    if fronts:
        executables.append(('front', args[fronts[0] + 1]))
    vision = cfg.get('vision') or {}
    for key in ('exe', 'cpu_exe'):
        if vision.get(key):
            executables.append(('vision.' + key, vision[key]))
    env = os.environ.copy()
    dirs = [str(p) for p in cfg.get('lib_dirs', []) if Path(p).is_dir()]
    if dirs:
        env['LD_LIBRARY_PATH'] = ':'.join(dirs + ([env['LD_LIBRARY_PATH']] if env.get('LD_LIBRARY_PATH') else []))
    builds, stamps = {}, {}
    for role, name in executables:
        if not isinstance(name, str) or not name:
            fail(f'{role}: executable missing')
        exe = Path(name)
        exe = (cwd / exe).resolve() if not exe.is_absolute() else exe.resolve()
        if not exe.is_file() or not os.access(exe, os.X_OK):
            fail(f'{role}: executable not found or not executable: {exe}')
        build = next((d for d in exe.parents if (d / 'CMakeCache.txt').is_file()), None)
        if build is None:
            fail(f'{role}: no enclosing CMake build: {exe}')
        cache = (build / 'CMakeCache.txt').read_text().splitlines()
        values = dict(line.split('=', 1) for line in cache if '=' in line and not line.startswith(('#', '//')))
        home = Path(values.get('CMAKE_HOME_DIRECTORY:INTERNAL', '/')).resolve()
        if home != ENGINE and not (role.startswith('vision.') and home == VISION):
            fail(f'{role}: build is not from this checkout: {build}')
        stamp = build / 'BUILD_COMMIT'
        if not stamp.is_file():
            fail(f'{role}: no BUILD_COMMIT in {build} (write `git rev-parse HEAD` there after a successful build)')
        built = stamp.read_text().strip()
        if built != head and source_changed(built, head):
            fail(f'{role}: stale build: the sources changed since {built[:12]} ({build})')
        stamps[role] = built
        if home == VISION:
            backend = 'vision-hip' if values.get('STRATA_VISION_HIP:BOOL') == 'ON' else 'vision-cpu'
        else:
            backend = 'hip' if values.get('STRATA_ENABLE_HIP:BOOL') == 'ON' else 'cuda' if values.get('STRATA_ENABLE_CUDA:BOOL') == 'ON' else 'host'
        builds[role] = (build, backend)
        result = subprocess.run(['ldd', str(exe)], env=env, cwd=cwd, text=True, capture_output=True, timeout=20)
        if result.returncode:
            fail(f'{role}: ldd failed: {result.stderr.strip()}')
        allowed = [build] + [Path(p).resolve() for p in ('/lib', '/lib64', '/lib32', '/libx32')]
        allowed += [p.resolve() for parent, pattern in ((Path('/opt'), 'rocm*'), (Path('/usr/local'), 'cuda*'), (Path('/usr'), 'lib*')) for p in parent.glob(pattern)]
        for line in result.stdout.splitlines():
            if 'not found' in line:
                fail(f'{role}: unresolved library: {line.strip()}')
            words = line.split('=>', 1)[-1].strip().split()
            if not words or not words[0].startswith('/'):
                continue
            lib = Path(words[0]).resolve()
            if not any(lib == root or root in lib.parents for root in allowed):
                fail(f'{role}: library outside build/system/toolkit: {lib}')
    host = builds['host']
    if host[1] != 'hip':
        fail('host is not a HIP build')
    if fronts and (builds['front'][1] != 'cuda' or builds['front'][0] == host[0]):
        fail('front is not a separate CUDA build')
    first = next(iter(stamps.values()))
    if any(c != first and source_changed(first, c) for c in stamps.values()):   # (each is current, so this is a guard)
        fail('the builds come from different sources: ' + ', '.join(f'{r} {c[:12]}' for r, c in sorted(stamps.items())))
    for role, (build, backend) in builds.items():
        if role.startswith('vision.') and build != host[0] and not backend.startswith('vision'):
            fail(f'{role}: neither the host build nor a tools/vision build of this checkout: {build}')
    if 'vision.exe' in builds and builds['vision.exe'][1] == 'vision-cpu' and cfg.get('vision', {}).get('gpu'):
        fail('vision.exe: "gpu": true needs a STRATA_VISION_HIP build')
    return builds


def main():
    if len(sys.argv) != 2:
        raise SystemExit('usage: check_launch.py CONFIG')
    try:
        with open(sys.argv[1]) as f:
            cfg = expand(json.load(f), {'STRATA_REPO': str(REPO), 'RUN_DIR': os.environ.get('RUN_DIR', str(REPO / 'runs'))})
        print('launch check OK:', check_config(cfg))
    except (ValueError, OSError, subprocess.SubprocessError) as e:
        raise SystemExit(f'launch check: {e}')


if __name__ == '__main__':
    main()
