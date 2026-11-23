#!/usr/bin/env python3
"""Handoff 0094 runner: one replay of the recorded session on a freshly started server (the Human User's maxperf
config, copied), with the safety watchers.  Numbers only leave this script.

    runner.py NAME [--limit N] [--env K=V ...] [--exe-wrap PATH] [--front-wrap PATH] [--pyspy] [--sampler]
              [--timeout-s S]

Server started like the run script (serve/server.py --engine strata --config COPY --port 8081) from this Python
process after reset_ignored_signals(); stopped with stop_cooperatively (SIGINT, drain, SIGTERM; never SIGKILL).
"""
import argparse, json, os, re, signal, subprocess, sys, threading, time, urllib.request
from pathlib import Path

REPO = Path('/home/chris/workspace/Strata-heterogeneous')
sys.path.insert(0, str(REPO / 'mixed_split/tools'))
from mixed_split_common import reset_ignored_signals, stop_cooperatively, force_sdma_off  # noqa: E402
from check_launch import check_config  # noqa: E402

REC = '/home/chris/workspace/strata-local-archive/recordings/session-20261123T170506Z.jsonl'
BOOT = 'f94c87ce-2f4b-412e-b02c-09dfdb04f155'
SAFETY = REPO / '.ai/runs/0094-safety'
PROF = Path('/home/chris/workspace/strata-local-archive/profiles/0094')
PY = str(REPO / '.venv/bin/python')
STOP_PAT = re.compile(r'gfxhub|GCVM|page fault|segfault|general protection|Bad page|Bad swap|Oops|BUG:|rcu.*stall|'
                      r'amdgpu.*(fault|reset|timeout|hang)|ring .* timeout|MES failed|Xid|NVRM', re.I)
REC_PAT = re.compile(STOP_PAT.pattern + r'|amdgpu|workqueue:.*\[amdgpu\]|restore_userptr_worker|svm_range_restore_work|MES', re.I)


def free_m():
    return subprocess.run(['free', '-m'], capture_output=True, text=True, timeout=10).stdout


def leftovers():
    r = subprocess.run(['pgrep', '-af', r'build-0093|serve/server.py|strata-vision|rocprofv3|nsys|py-spy'],
                       capture_output=True, text=True, timeout=10).stdout
    return [l for l in r.splitlines() if 'runner.py' not in l and 'pgrep' not in l]


def children(pid):
    out = []
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            try:
                st = (p / 'stat').read_text()
                if int(st.rsplit(')', 1)[1].split()[1]) == pid:
                    out.append(int(p.name))
            except (OSError, ValueError, IndexError):
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name')
    ap.add_argument('--limit', type=int, default=106)
    ap.add_argument('--env', action='append', default=[])
    ap.add_argument('--exe-wrap')
    ap.add_argument('--front-wrap')
    ap.add_argument('--pyspy', action='store_true')
    ap.add_argument('--sampler', action='store_true')
    ap.add_argument('--timeout-s', type=float, default=2400)
    ap.add_argument('--cfg', action='append', default=[], help='KEY=JSON top-level config override (a copy)')
    a = ap.parse_args()
    reset_ignored_signals()
    force_sdma_off(os.environ)
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == BOOT, 'boot id changed'
    assert not (SAFETY / 'STOP').exists(), 'STOP file present'
    left = leftovers()
    assert not left, f'leftover processes: {left}'
    d = REPO / f'.ai/runs/0094-{a.name}'
    d.mkdir(parents=True, exist_ok=False)
    prof = PROF / a.name
    prof.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((REPO / 'strata-iq3_xxs-mixed-maxperf.json').read_text())
    cfg['log'] = str(prof / 'server.log')          # the log can hold prompt text in errors: kept local
    cfg['port'] = 8081
    for kv in a.env:
        k, v = kv.split('=', 1)
        cfg['env'][k] = v
    for kv in a.cfg:
        k, v = kv.split('=', 1)
        cfg[k] = json.loads(v)
    cfg['env']['HSA_ENABLE_SDMA'] = '0'
    check_config(cfg)                               # the unwrapped config: builds, libraries, the first rule
    if a.exe_wrap:
        cfg['env']['STRATA_0094_REAL_EXE'] = cfg['exe']
        cfg['exe'] = a.exe_wrap
    if a.front_wrap:
        i = cfg['args'].index('--front-exe')
        cfg['env']['STRATA_0094_REAL_FRONT'] = cfg['args'][i + 1]
        cfg['args'][i + 1] = a.front_wrap
    if a.sampler:
        cfg['env']['LD_PRELOAD'] = str(SAFETY / 'cpusampler/libcpusampler.so')
        cfg['env']['CPUSAMPLER_DIR'] = str(prof / 'cpusampler')
        (prof / 'cpusampler').mkdir(exist_ok=True)
    cpath = d / 'config.json'
    cpath.write_text(json.dumps(cfg, indent=1))
    info = dict(name=a.name, limit=a.limit, env=a.env, cfg=a.cfg, exe_wrap=a.exe_wrap, front_wrap=a.front_wrap, pyspy=a.pyspy,
                sampler=a.sampler, boot=BOOT, free_before=free_m(), t_start=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    stop = threading.Event()
    j = subprocess.Popen(['journalctl', '-k', '-f', '-n', '0', '--no-pager'], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)

    def watcher():
        with (d / 'kernel.log').open('w') as f:
            for line in j.stdout:
                if REC_PAT.search(line):
                    f.write(line)
                    f.flush()
                if STOP_PAT.search(line):
                    info['stop_line'] = line
                    (SAFETY / 'STOP').write_text(line)
                    stop.set()
    threading.Thread(target=watcher, daemon=True).start()
    env = os.environ.copy()
    argv = [PY, str(REPO / 'serve/server.py'), '--engine', 'strata', '--config', str(cpath), '--port', '8081']
    if a.pyspy:
        argv = [str(REPO / '.venv/bin/py-spy'), 'record', '--nonblocking', '--threads', '--rate', '100', '--format', 'raw',
                '-o', str(prof / 'pyspy.raw'), '--'] + argv
    t0 = time.monotonic()
    console = (prof / 'console.log').open('w')
    p = subprocess.Popen(argv, cwd=str(REPO), stdout=console, stderr=subprocess.STDOUT, env=env)
    server_pid = p.pid
    rp = None
    try:
        for _ in range(400):
            if stop.is_set() or p.poll() is not None:
                raise RuntimeError(f'server exited {p.returncode} or stop line')
            try:
                with urllib.request.urlopen('http://127.0.0.1:8081/health', timeout=2) as h:
                    if json.load(h).get('loaded'):
                        break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1)
        else:
            raise RuntimeError('server not ready')
        info['ready_s'] = round(time.monotonic() - t0, 1)
        if a.pyspy:
            kids = children(p.pid)
            server_pid = kids[0] if kids else p.pid
        info['server_pid'] = server_pid
        if a.sampler:
            (prof / 'cpusampler' / 'START').write_text('1')
        rp = subprocess.Popen([PY, str(REPO / 'mixed_split/tools/replay_session.py'), 'run', REC, str(d / 'replay'),
                               '--url', 'http://127.0.0.1:8081', '--limit', str(a.limit)],
                              stdout=(d / 'replay.log').open('w'), stderr=subprocess.STDOUT, env=env)
        tr = time.monotonic()
        log = Path(cfg['log'])
        while rp.poll() is None:
            time.sleep(1)
            if stop.is_set():
                info['error'] = 'kernel stop line'
                break
            if p.poll() is not None:
                info['error'] = f'server exited {p.returncode}'
                break
            if time.monotonic() - tr > a.timeout_s:
                info['error'] = 'time limit'
                break
            if log.exists() and int(time.monotonic()) % 15 == 0:
                if 'non-finite logits' in log.read_text(errors='replace').lower():
                    info['error'] = 'non-finite logits'
                    (SAFETY / 'STOP').write_text('non-finite logits')
                    break
        if rp.poll() is None:
            rp.send_signal(signal.SIGINT)
            try:
                rp.wait(timeout=60)
            except subprocess.TimeoutExpired:
                rp.terminate()
                rp.wait(timeout=30)
        info['replay_rc'] = rp.returncode
        info['replay_s'] = round(time.monotonic() - tr, 1)
        if a.sampler:
            (prof / 'cpusampler' / 'STOP').write_text('1')
            time.sleep(8)
    except Exception as e:  # noqa: BLE001
        info['error'] = f'{type(e).__name__}: {e}'
    finally:
        if a.pyspy and server_pid != p.pid:
            # stop the server itself (py-spy's child) cooperatively; py-spy writes its file when its target exits
            for sig, w in ((signal.SIGINT, 120), (signal.SIGTERM, 60)):
                if not Path(f'/proc/{server_pid}').exists():
                    break
                os.kill(server_pid, sig)
                for _ in range(w):
                    if not Path(f'/proc/{server_pid}').exists() or \
                            Path(f'/proc/{server_pid}/stat').read_text().rsplit(')', 1)[1].split()[0] == 'Z':
                        break
                    time.sleep(1)
                else:
                    continue
                break
            try:
                info['server_rc'] = p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                info['server_rc'] = stop_cooperatively(p, drain_s=60, term_s=60)
        else:
            info['server_rc'] = stop_cooperatively(p, drain_s=120, term_s=60)
        info['elapsed_s'] = round(time.monotonic() - t0, 1)
        time.sleep(2)
        j.terminate()
        j.wait(timeout=10)
        console.close()
        info['free_after'] = free_m()
        for _ in range(600):                        # a profiler may still be writing its files
            if not [l for l in leftovers() if 'nsys' in l or 'rocprofv3' in l]:
                break
            time.sleep(1)
        info['leftovers'] = leftovers()
        log = Path(cfg['log'])
        if log.exists():
            txt = log.read_text(errors='replace')
            info['gpu_pause_lines'] = [l for l in txt.splitlines() if 'paused the GPU' in l]
            info['non_finite'] = txt.lower().count('non-finite logits')
        (d / 'runner.json').write_text(json.dumps(info, indent=1))
        print(json.dumps({k: v for k, v in info.items() if k not in ('free_before', 'free_after', 'gpu_pause_lines')}, indent=1))
        print('pause lines', len(info.get('gpu_pause_lines', [])))
        if info['server_rc'] is None or info['leftovers']:
            (SAFETY / 'STOP').write_text('process did not stop: ' + repr(info['leftovers']))
    return 1 if info.get('error') or info.get('replay_rc') else 0


if __name__ == '__main__':
    sys.exit(main())
