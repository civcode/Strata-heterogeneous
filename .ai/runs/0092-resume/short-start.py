import sys, os, time, subprocess, threading, json, urllib.request, re, pathlib, traceback
sys.path.insert(0,'mixed_split/tools')
from mixed_split_common import reset_ignored_signals, stop_cooperatively, force_sdma_off
reset_ignored_signals(); force_sdma_off(os.environ)
out=pathlib.Path('.ai/runs/0092-resume'); out.mkdir(exist_ok=True)
boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
assert boot=='ffcbbb74-121f-46fa-a930-ca08f55e881c', 'boot changed'
stop=threading.Event(); done=threading.Event(); results={}; started=time.monotonic()
pattern=re.compile(r'gfxhub|GCVM|page fault|segfault|general protection|Bad page|Bad swap|Oops|BUG:|rcu.*stall|amdgpu.*(fault|reset|timeout|hang)|ring .* timeout|MES failed|Xid|NVRM',re.I)
record=re.compile(r'gfxhub|GCVM|page fault|segfault|general protection|Bad page|Bad swap|Oops|BUG:|rcu.*stall|amdgpu|ring .* timeout|MES|Xid|NVRM|restore_userptr_worker|svm_range_restore_work',re.I)
journal=subprocess.Popen(['journalctl','-k','-f','-n','0','--no-pager'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
def watch():
 with (out/'kernel-during.log').open('w') as f:
  for line in journal.stdout:
   if record.search(line): f.write(line); f.flush()
   if pattern.search(line): results['stop_line']=line; stop.set()
threading.Thread(target=watch,daemon=True).start()
log=pathlib.Path('strata-iq3_xxs-mixed.log'); offset=log.stat().st_size if log.exists() else 0
with (out/'server-console.log').open('w') as console:
 p=subprocess.Popen(['./run-iq3_xxs-mixed.sh'],stdout=console,stderr=subprocess.STDOUT,env=os.environ.copy())
 def client():
  try:
   while time.monotonic()-started<350 and not stop.is_set():
    if p.poll() is not None: raise RuntimeError('server exited during start')
    try:
     with urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=2) as r:
      if r.status==200: break
    except Exception: time.sleep(1)
   else: raise RuntimeError('server not ready')
   results['ready_s']=time.monotonic()-started
   from serve_session import stream_req
   results['short']=stream_req(8080,'Reply with a short greeting.',64)
   if not stop.is_set():
    prompt=('The quick brown fox jumps over the lazy dog. ' * 900)+'\nSummarize the repeated sentence in one line.'
    results['long']=stream_req(8080,prompt,64)
  except Exception: results['error']=traceback.format_exc()
  finally: done.set()
 threading.Thread(target=client,daemon=True).start()
 try:
  while not done.wait(0.5):
   if p.poll() is not None: results['error']='server exited: '+str(p.returncode); break
   if time.monotonic()-started>500: results['error']='500 second limit'; break
   if log.exists():
    with log.open(errors='replace') as f:
     f.seek(offset); s=f.read()
    if 'non-finite logits' in s.lower(): results['error']='non-finite logits'; break
   if stop.is_set(): break
 finally:
  results['stop_return']=stop_cooperatively(p,drain_s=60,term_s=30)
  results['elapsed_s']=time.monotonic()-started
  journal.terminate(); journal.wait(timeout=5)
  (out/'results.json').write_text(json.dumps(results,indent=2))
  print(json.dumps(results,indent=2),flush=True)
