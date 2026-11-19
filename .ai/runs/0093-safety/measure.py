import os, sys, time, json, pathlib, subprocess, threading, re, traceback, urllib.request, statistics
sys.path.insert(0,'mixed_split/tools')
from mixed_split_common import reset_ignored_signals, stop_cooperatively, force_sdma_off, expand
from check_launch import check_config
reset_ignored_signals(); force_sdma_off(os.environ)
name=sys.argv[1]; model, mode = name.split('-',1); mode=mode.split('-')[0]
root=pathlib.Path('.ai/runs/0093-measure').resolve(); root.mkdir(parents=True,exist_ok=True); d=root/name; d.mkdir(exist_ok=False)
repo=pathlib.Path.cwd(); boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(); assert boot=='ffcbbb74-121f-46fa-a930-ca08f55e881c'
assert not pathlib.Path('.ai/runs/0093-safety/STOP').exists()
if model=='iq': cfg=json.loads(pathlib.Path('strata-iq3_xxs-mixed-maxperf.json').read_text())
else:
 cfg=json.loads(pathlib.Path('mixed_split/configs/mixed.example.json').read_text()); cfg=expand(cfg,dict(STRATA_REPO=str(repo),STRATA_DATA='/home/chris/workspace/Strata-data',STRATA_MODEL_DIR='/home/chris/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF',STRATA_PROFILE='/home/chris/workspace/Strata/data/expert-profile.bin',STRATA_VANILLA='/home/chris/workspace/Strata',ROCM_LIB='/opt/rocm/core-10.1/lib',RUN_DIR=str(d)))
if 'new' in name:
 cfg=json.loads(json.dumps(cfg).replace('build-0092-','build-0093-'))
args=cfg['args']; idx=args.index('--pipeline-windows'); args[idx+1]='0' if mode=='serial' else '2'
if mode=='vanilla':
 for opt in ('--front-exe','--front-layers'):
  i=args.index(opt); del args[i:i+2]
if 'room' in name: args.extend(['--front-arg','--vram-reserve-mib','--front-arg','450'])
if 'lowcache' in name: args.extend(['--front-arg','--vram-reserve-mib','--front-arg','956'])
if 'pool4' in name: args.extend(['--front-arg','--pool-workers','--front-arg','4'])
if 'last' in name: args.extend(['--front-arg','--host-core','--front-arg','last'])
if 'sibling' in name: args.extend(['--front-arg','--host-core','--front-arg','sibling'])
if 'tailoff' in name: cfg['env']['STRATA_ROUTE_TAIL_SKIP']='0'
cfg['env'].update(STRATA_SPLIT_TIMING='1',STRATA_DECODE_TIMING='1',STRATA_PIPELINE_AUTO='0',STRATA_MTP_STAGE0='1' if mode=='first' else '0', HSA_ENABLE_SDMA='0')
cfg['log']=str(d/'server.log'); cfg['port']=18093; cfg['open_browser']=False
check_config(cfg); (d/'config.json').write_text(json.dumps(cfg,indent=2));
stop=threading.Event(); done=threading.Event(); sample_stop=threading.Event(); results={}; samples=[]; start=time.monotonic()
pat=re.compile(r'gfxhub|GCVM|page fault|segfault|general protection|Bad page|Bad swap|Oops|BUG:|rcu.*stall|amdgpu.*(fault|reset|timeout|hang)|ring .* timeout|MES failed|Xid|NVRM',re.I)
rec=re.compile(r'gfxhub|GCVM|page fault|segfault|general protection|Bad page|Bad swap|Oops|BUG:|rcu.*stall|amdgpu|ring .* timeout|MES|Xid|NVRM|restore_userptr_worker|svm_range_restore_work',re.I)
j=subprocess.Popen(['journalctl','-k','-f','-n','0','--no-pager'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
def watcher():
 with (d/'kernel.log').open('w') as f:
  for line in j.stdout:
   if rec.search(line): f.write(line); f.flush()
   if pat.search(line): results['stop_line']=line; stop.set(); pathlib.Path('.ai/runs/0093-safety/STOP').write_text(line)
threading.Thread(target=watcher,daemon=True).start()
def sampler():
 while not sample_stop.is_set():
  try:
   nv=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10,env=os.environ)
   amd=json.loads(subprocess.check_output(['amd-smi','metric','-u','--json'],text=True,timeout=10,env=os.environ))
   samples.append({'t':time.monotonic(),'cuda':int(nv.strip().splitlines()[0]),'hip':[x['usage']['gfx_activity']['value'] for x in amd['gpu_data'][:2]]})
  except Exception as e: samples.append({'t':time.monotonic(),'error':repr(e)})
  sample_stop.wait(0.8)
threading.Thread(target=sampler,daemon=True).start()
def request(content,n):
 data={'model':'x','messages':[{'role':'user','content':content}],'temperature':0,'max_tokens':n,'reasoning_budget_tokens':0,'stream':True,'stream_options':{'include_usage':True}}
 req=urllib.request.Request('http://127.0.0.1:18093/v1/chat/completions',data=json.dumps(data).encode(),headers={'Content-Type':'application/json'})
 t0=time.monotonic(); first=last=None; usage=timings=None; response=[]
 with urllib.request.urlopen(req,timeout=180) as f:
  for line in f:
   line=line.strip()
   if not line.startswith(b'data:') or line==b'data: [DONE]': continue
   e=json.loads(line[5:]); usage=e.get('usage') or usage; timings=e.get('timings') or timings
   for c in e.get('choices',[]):
    delta=c.get('delta',{}); text=delta.get('content') or delta.get('reasoning_content')
    if text:
     now=time.monotonic(); first=first or now; last=now; response.append(text)
 subset=[s for s in samples if first and last and first<=s['t']<=last and 'hip' in s]
 busy={k:statistics.median([s[k] for s in subset]) for k in ['cuda']} if subset else {}
 if subset:
  busy['hip0']=statistics.median([s['hip'][0] for s in subset]); busy['hip1']=statistics.median([s['hip'][1] for s in subset])
 return dict(start=t0,first=first,last=last,end=time.monotonic(),usage=usage,timings=timings,busy=busy,busy_samples=len(subset),text=''.join(response))
with (d/'console.log').open('w') as console:
 p=subprocess.Popen([sys.executable,'mixed_split/tools/serve_mixed.py','--config',str(d/'config.json'),'--run-dir',str(d),'--port','18093'],stdout=console,stderr=subprocess.STDOUT,env=os.environ.copy())
 def client():
  try:
   while time.monotonic()-start<240 and not stop.is_set():
    if p.poll() is not None: raise RuntimeError('server exited '+str(p.returncode))
    try:
     with urllib.request.urlopen('http://127.0.0.1:18093/health',timeout=2) as r:
      if r.status==200: break
    except Exception: time.sleep(1)
   else: raise RuntimeError('not ready')
   results['ready_s']=time.monotonic()-start
   code='Write a complete Python implementation of a thread-safe LRU cache, including type hints, documentation, and example usage. Explain the design and important edge cases.'
   docs=pathlib.Path('docs/MULTI_GPU.md').read_text()+pathlib.Path('README.md').read_text()
   results['short']=request(code,300)
   if not stop.is_set(): results['long']=request('Summarise the following document in five detailed bullet points.\n\n'+docs,300)
  except Exception: results['error']=traceback.format_exc()
  finally: done.set()
 threading.Thread(target=client,daemon=True).start()
 try:
  while not done.wait(.25):
   if stop.is_set(): break
   if p.poll() is not None: results['error']='server exited '+str(p.returncode); break
   if time.monotonic()-start>440: results['error']='time limit'; break
   path=d/'server.log'
   if path.exists() and 'non-finite logits' in path.read_text(errors='replace').lower():
    results['error']='non-finite logits'; pathlib.Path('.ai/runs/0093-safety/STOP').write_text(results['error']); break
 finally:
  results['stop_return']=stop_cooperatively(p,drain_s=60,term_s=40); results['elapsed_s']=time.monotonic()-start
  sample_stop.set(); j.terminate(); j.wait(timeout=5)
  (d/'samples.json').write_text(json.dumps(samples,indent=1)); (d/'result.json').write_text(json.dumps(results,indent=2))
  print(json.dumps({k:v for k,v in results.items() if k not in ('short','long')},indent=2),flush=True)
  for k in ('short','long'):
   if k in results: print(k,results[k]['usage'],results[k]['timings'],results[k]['busy'],results[k]['busy_samples'],flush=True)
  if results['stop_return'] is None: pathlib.Path('.ai/runs/0093-safety/STOP').write_text('process did not stop')
 if 'error' in results or stop.is_set() or results['stop_return'] is None: sys.exit(1)
