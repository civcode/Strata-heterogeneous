#!/usr/bin/env python3
import os, signal, subprocess, sys, time, urllib.request, json
run, mode, arg = sys.argv[1:]
os.makedirs(run, exist_ok=True)
env = dict(STRATA_DATA='/home/chris/workspace/Strata-data', STRATA_MODEL_DIR='/home/chris/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF', STRATA_PROFILE='/home/chris/workspace/Strata/data/expert-profile.bin', STRATA_VANILLA='/home/chris/workspace/Strata', ROCM_LIB='/opt/rocm/core-10.1/lib', HIP_VISIBLE_DEVICES='0,1')
env.update(os.environ, HSA_ENABLE_SDMA='0', STRATA_IQ_MT_MIN='1')
def reset():
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
server = None
client = None
start = time.monotonic()
rc = 1
with open(os.path.join(run, 'console.log'), 'w') as log:
    try:
        server = subprocess.Popen([sys.executable, 'mixed_split/tools/serve_mixed.py', '--config', 'mixed_split/configs/mixed.example.json', '--port', '8096', '--run-dir', run], env=env, stdout=log, stderr=subprocess.STDOUT, preexec_fn=reset)
        while time.monotonic() - start < 90 and server.poll() is None:
            try:
                with urllib.request.urlopen('http://127.0.0.1:8096/health', timeout=2) as r:
                    if json.load(r).get('loaded'): break
            except Exception: pass
            time.sleep(.5)
        else: raise RuntimeError('server failed readiness: %s' % server.poll())
        args = [sys.executable, 'mixed_split/tools/agent_session.py', mode, arg, '--url', 'http://127.0.0.1:8096']
        if mode == 'run': args.append('--scaled')
        client = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT, preexec_fn=reset)
        rc = client.wait()
    finally:
        if client and client.poll() is None:
            client.send_signal(signal.SIGINT)
            client.wait()
        if server and server.poll() is None:
            server.send_signal(signal.SIGINT)
        if server:
            server.wait()
            if server.returncode != 0: rc = 1
print('agent %s elapsed %.1fs client/server rc %s/%s' % (mode, time.monotonic()-start, rc, server.returncode if server else None), flush=True)
sys.exit(rc)
