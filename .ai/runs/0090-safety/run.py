#!/usr/bin/env python3
import datetime, json, os, re, signal, subprocess, sys, threading, time, urllib.request
sys.path.insert(0, os.path.abspath('mixed_split/tools'))
from mixed_split_common import reset_ignored_signals, stop_cooperatively
reset_ignored_signals()
run, config, mode, arg = sys.argv[1:5]
limit = int(sys.argv[5]) if len(sys.argv) > 5 else 6900
os.makedirs(run, exist_ok=True)
marker = '.ai/runs/0090-safety/STOP'
if os.path.exists(marker): raise SystemExit('a safety stop has already occurred')
env = dict(STRATA_DATA='/home/chris/workspace/Strata-data', STRATA_MODEL_DIR='/home/chris/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF', STRATA_PROFILE='/home/chris/workspace/Strata/data/expert-profile.bin', STRATA_VANILLA='/home/chris/workspace/Strata', ROCM_LIB='/opt/rocm/core-10.1/lib', HIP_VISIBLE_DEVICES='0,1')
env.update(os.environ, HSA_ENABLE_SDMA='0', STRATA_IQ_MT_MIN='1')
env.pop('HSA_USERPTR_FOR_PAGED_MEM', None)
boot = open('/proc/sys/kernel/random/boot_id').read().strip()
j = subprocess.run(['journalctl','-k','-b','-n','1','--show-cursor','--no-pager'], capture_output=True, text=True, check=True)
cursor = j.stdout.rsplit('-- cursor: ', 1)[1].strip()
safety = re.compile(r'gfxhub|GCVM|page fault|segfault|general protection|Bad page|Bad swap|Oops|BUG:|rcu.*stall|amdgpu.*(fault|reset|timeout|hang)|ring .* timeout|MES failed|Xid|NVRM', re.I)
watch = re.compile(r'gfxhub|GCVM|page fault|segfault|general protection|Bad page|Bad swap|Oops|BUG:|rcu.*stall|amdgpu|ring .* timeout|MES|Xid|NVRM|restore_userptr_worker|svm_range_restore_work', re.I)
logfile = os.path.join(run, 'server-' + ('mixed' if 'mixed.example' in config else 'hip') + '.log')
done = threading.Event()
reason = []
start = time.monotonic()
def now(): return datetime.datetime.now().astimezone().isoformat(timespec='seconds')
def latch(msg, safety_stop=True):
    if not reason:
        reason.append(msg)
        if safety_stop:
            with open(marker, 'w') as f: f.write(now()+' '+msg+'\n')
        print('STOP', msg, flush=True)
def monitor():
    global cursor
    offset = 0
    with open(os.path.join(run, 'watcher.log'), 'w', buffering=1) as out:
        out.write(now()+' watcher started\n')
        while not done.wait(.5):
            j = subprocess.run(['journalctl','-k','-b','--after-cursor='+cursor,'--show-cursor','--no-pager','-o','short-iso'], capture_output=True, text=True)
            if j.returncode:
                latch('journalctl failed'); continue
            for line in j.stdout.splitlines():
                if line.startswith('-- cursor: '): cursor = line[len('-- cursor: '):]
                elif watch.search(line):
                    out.write(now()+' KERNEL '+line+'\n')
                    if safety.search(line): latch('kernel safety line: '+line)
            try:
                with open(logfile, 'r', errors='replace') as f:
                    f.seek(offset)
                    for line in f:
                        if 'paused the GPU' in line or 'non-finite logits' in line.lower():
                            out.write(now()+' ENGINE '+line)
                            if 'non-finite logits' in line.lower(): latch('non-finite logits: '+line.strip())
                    offset = f.tell()
            except FileNotFoundError: pass
            if boot != open('/proc/sys/kernel/random/boot_id').read().strip(): latch('boot changed')
            if time.monotonic() - start > limit: latch('run time limit', False)
        out.write(now()+' watcher stopped\n')
server = client = None
rc = 1
with open(os.path.join(run, 'console.log'), 'w') as log:
    threading.Thread(target=monitor, daemon=True).start()
    try:
        server = subprocess.Popen([sys.executable,'mixed_split/tools/serve_mixed.py','--config',config,'--port','8096','--run-dir',run], env=env, stdout=log, stderr=subprocess.STDOUT)
        while time.monotonic()-start < 120 and server.poll() is None and not reason:
            try:
                with urllib.request.urlopen('http://127.0.0.1:8096/health',timeout=2) as r:
                    if json.load(r).get('loaded'): break
            except Exception: pass
            time.sleep(.5)
        else: raise RuntimeError('server readiness failed: %s %s' % (server.poll(), reason))
        args = [sys.executable,'mixed_split/tools/agent_session.py',mode,arg,'--url','http://127.0.0.1:8096']
        client = subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT)
        sent = False
        while client.poll() is None:
            if server.poll() is not None: latch('server exited unexpectedly rc=%s' % server.returncode)
            if reason and not sent:
                client.send_signal(signal.SIGUSR1 if mode == 'run' else signal.SIGINT)
                sent = True
            time.sleep(.2)
        rc = client.returncode
    finally:
        if client and client.poll() is None:
            if mode == 'run':
                client.send_signal(signal.SIGUSR1)
                client.wait()
            else: stop_cooperatively(client)
        if server:
            sr = stop_cooperatively(server)
            if sr is None:
                latch('server did not drain')
            if sr != 0: rc = 2
        done.set()
elapsed = time.monotonic()-start
result = dict(elapsed=elapsed,client_rc=rc,server_rc=server.returncode if server else None,stop=reason)
with open(os.path.join(run,'run-result.json'),'w') as f: json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
sys.exit(2 if reason else rc)
