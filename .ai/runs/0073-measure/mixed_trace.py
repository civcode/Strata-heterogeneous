import json, os, signal, subprocess, sys, time, urllib.request
from pathlib import Path
root=Path.cwd(); run=root/'.ai/runs/0073-measure'; port=8092
sys.path.insert(0,str(root/'tools')); from strata_tokenizer import Tokenizer
model='/home/chris/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF'
tk=Tokenizer.from_gguf(model+'/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf')
doc=(root/'docs/MULTI_GPU.md').read_text()+(root/'README.md').read_text()
body=tk.decode(tk.encode(doc)[:4900])
prompts=['Write a complete Python implementation of a thread-safe LRU cache, including type hints, documentation, and example usage. Explain the design and important edge cases.', 'Summarise the following document in five bullet points.\n\n'+body]
env=dict(os.environ,STRATA_DATA='/home/chris/workspace/Strata-data',STRATA_MODEL_DIR=model,STRATA_PROFILE='/home/chris/workspace/Strata/data/expert-profile.bin',STRATA_VANILLA='/home/chris/workspace/Strata',ROCM_LIB='/opt/rocm/core-10.1/lib',HSA_ENABLE_SDMA='0',STRATA_SPLIT_TIMING='1',STRATA_DECODE_TIMING='1')
base=subprocess.check_output(['journalctl','-k','-b','--no-pager'],text=True); total=508.1
for i,arm in enumerate(['trace']):
 cfg=json.load(open('hetero/configs/mixed.example.json')); cfg['exe']=cfg['exe'].replace('build-0070-', 'build-0071-'); cfg['args']=[x.replace('build-0070-', 'build-0071-') for x in cfg['args']]; config=run/f'mixed-traced-{i}.config.json'; config.write_text(json.dumps(cfg)); pfx=run/f'mixed-traced-{i}-{arm}'; start=time.monotonic(); results=[]
 with open(str(pfx)+'.out','w') as log:
  p=subprocess.Popen([sys.executable,'hetero/tools/serve_mixed.py','--config',str(config),'--port',str(port),'--run-dir',str(run)],stdout=log,stderr=subprocess.STDOUT,env=env)
  try:
   while time.monotonic()-start<70 and p.poll() is None:
    try:
     with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as r:
      if json.load(r).get('loaded'): break
    except Exception: pass
    time.sleep(.5)
   else: raise RuntimeError('not ready')
   for prompt,n in zip(prompts,[300,200]):
    data={'model':'x','messages':[{'role':'user','content':prompt}],'temperature':0,'max_tokens':n,'reasoning_budget_tokens':0}
    req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',data=json.dumps(data).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=45) as r: results.append(json.load(r))
  except Exception as e: results.append({'error':repr(e)})
  finally:
   if p.poll() is None: p.send_signal(signal.SIGINT)
   try: p.wait(timeout=90)
   except subprocess.TimeoutExpired: print('STOP still running',p.pid,flush=True); sys.exit(1)
 elapsed=time.monotonic()-start; total+=elapsed
 (Path(str(pfx)+'.json')).write_text(json.dumps({'elapsed':elapsed,'rc':p.returncode,'responses':results},indent=2)); now=subprocess.check_output(['journalctl','-k','-b','--no-pager'],text=True); delta=now[len(base):] if now.startswith(base) else now; Path(str(pfx)+'.kernel-delta.txt').write_text(delta); base=now
 print(i,arm,'rc',p.returncode,'elapsed',round(elapsed,1),'total',round(total,1),flush=True)
 if p.returncode or elapsed>=120 or len(results)!=2 or any('error' in r for r in results) or any(s in delta.lower() for s in ('xid','gfxhub','gcvm','gpu fault')) or total>780: print('STOP safety or error',flush=True); break
