import os, sys, signal, subprocess, time, urllib.request, json, re, threading
sys.path.insert(0, os.path.abspath('mixed_split/tools'))
from mixed_split_common import reset_ignored_signals, stop_cooperatively
reset_ignored_signals()
run, config, mode, arg = sys.argv[1:5]
extra = sys.argv[5:]
os.makedirs(run, exist_ok=True)
env = dict(STRATA_DATA='/home/chris/workspace/Strata-data', STRATA_MODEL_DIR='/home/chris/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF', STRATA_PROFILE='/home/chris/workspace/Strata/data/expert-profile.bin', STRATA_VANILLA='/home/chris/workspace/Strata', ROCM_LIB='/opt/rocm/core-10.1/lib', HIP_VISIBLE_DEVICES='0,1')
env.update(os.environ, HSA_ENABLE_SDMA='0', STRATA_IQ_MT_MIN='1')
boot = open('/proc/sys/kernel/random/boot_id').read().strip()
def swap():
    d = dict(line.split(':', 1) for line in open('/proc/meminfo') if ':' in line)
    return int(d['SwapTotal'].split()[0]) - int(d['SwapFree'].split()[0])
base = swap()
j = subprocess.run(['journalctl','-k','-b','-n','1','--show-cursor','--no-pager'], capture_output=True, text=True, check=True)
cursor = j.stdout.rsplit('-- cursor: ', 1)[1].strip()
pat = re.compile(r'amdgpu|gfxhub|GCVM|MES|page fault|segfault|general protection|Bad page|Oops|rcu.*stall', re.I)
stop = []
done = threading.Event()
def monitor():
    while not done.wait(1):
        j = subprocess.run(['journalctl','-k','-b','--after-cursor='+cursor,'--no-pager','-o','short-iso'], capture_output=True, text=True)
        lines = [x for x in j.stdout.splitlines() if pat.search(x)]
        if lines: stop.append('kernel match: ' + '\n'.join(lines))
        elif boot != open('/proc/sys/kernel/random/boot_id').read().strip(): stop.append('boot changed')
        elif swap() > base + 1048576: stop.append('swap grew over 1 GiB')
        elif time.monotonic() - start > 540: stop.append('run reached 9 minutes')
        if stop:
            print('SAFETY STOP', stop[0], flush=True)
            os.kill(os.getpid(), signal.SIGINT)
            return
start = time.monotonic()
server = client = None
rc = 1
with open(os.path.join(run, 'console.log'), 'w') as log:
    threading.Thread(target=monitor, daemon=True).start()
    try:
        server = subprocess.Popen([sys.executable, 'mixed_split/tools/serve_mixed.py', '--config', config, '--port', '8096', '--run-dir', run], env=env, stdout=log, stderr=subprocess.STDOUT)
        while time.monotonic() - start < 90 and server.poll() is None:
            try:
                with urllib.request.urlopen('http://127.0.0.1:8096/health', timeout=2) as r:
                    if json.load(r).get('loaded'): break
            except Exception: pass
            time.sleep(.5)
        else: raise RuntimeError('server readiness failed: %s' % server.poll())
        args = [sys.executable, 'mixed_split/tools/agent_session.py', mode, arg, '--url', 'http://127.0.0.1:8096']
        if mode == 'run': args.append('--scaled')
        args.extend(extra)
        client = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT)
        rc = client.wait()
    finally:
        if client and client.poll() is None: stop_cooperatively(client)
        if server:
            sr = stop_cooperatively(server)
            if sr != 0: rc = 2
        done.set()
elapsed = time.monotonic()-start
print('elapsed %.1f client_rc %s server_rc %s safety %s' % (elapsed, rc, server.returncode if server else None, stop), flush=True)
with open(os.path.join(run, 'run-result.json'), 'w') as f: json.dump(dict(elapsed=elapsed, client_rc=rc, server_rc=server.returncode if server else None, safety=stop), f)
sys.exit(2 if stop else rc)
