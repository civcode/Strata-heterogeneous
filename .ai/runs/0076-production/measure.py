import base64, json, os, signal, subprocess, sys, time, urllib.request
from pathlib import Path
R=Path.cwd(); D=R/'.ai/runs/0076-production'; mode=sys.argv[1]; idx=sys.argv[2]; port=8096; name=f'{mode}-{idx}'
E=dict(os.environ,STRATA_DATA='/home/chris/workspace/Strata-data',STRATA_MODEL_DIR='/home/chris/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF',STRATA_PROFILE='/home/chris/workspace/Strata/data/expert-profile.bin',STRATA_VANILLA='/home/chris/workspace/Strata',ROCM_LIB='/opt/rocm/core-10.1/lib',HSA_ENABLE_SDMA='0',HIP_VISIBLE_DEVICES='0,1')
sys.path.insert(0,str(R/'tools')); from strata_tokenizer import Tokenizer
tk=Tokenizer.from_gguf(E['STRATA_MODEL_DIR']+'/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf')
prompts=json.load(open(D/'prompts.json'))
def journal(): return subprocess.check_output(['journalctl','-k','-b','--no-pager'],text=True,timeout=10)
def req(messages,n=128,**kw):
    d=dict(model='x',messages=messages,temperature=0,max_tokens=n,reasoning_budget_tokens=0,**kw)
    r=urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',data=json.dumps(d).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(r,timeout=55) as h: return json.load(h)
def snap(label):
    with open(D/f'{name}-{label}-memory.txt','w') as f:
        for cmd in (['free','-h'],['nvidia-smi'],['amd-smi','metric','-m']):
            f.write(subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=8).stdout+'\n')
base=journal(); t=time.monotonic(); out={}; p=None
cfg=json.load(open(R/f'hetero/configs/{"mixed" if mode=="fixed" else mode}.example.json'))
if mode=='fixed': cfg['layer_split']='33'
cp=D/f'{name}-config.json'; cp.write_text(json.dumps(cfg,indent=1))
with open(D/f'{name}-console.log','w') as log:
    try:
        p=subprocess.Popen([sys.executable,'hetero/tools/serve_mixed.py','--config',str(cp),'--port',str(port),'--run-dir',str(D/name)],stdout=log,stderr=subprocess.STDOUT,env=E)
        while time.monotonic()-t<65 and p.poll() is None:
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as h:
                    if json.load(h).get('loaded'): break
            except Exception: pass
            time.sleep(.5)
        else: raise RuntimeError(f'not ready; rc={p.poll()}')
        if idx=='features':
            snap('idle')
            prompt=tk.decode(prompts['long16k']); a={'role':'user','content':prompt}
            first=req([a],64); out['first']=first; snap('after16k')
            out['interference']=req([{'role':'user','content':'What is a mutex?'}],24)
            msg=first['choices'][0]['message']; out['second']=req([a,{'role':'assistant',**msg},{'role':'user','content':'Now give a one sentence summary.'}],64)
            out['cvec_off']=req([{'role':'user','content':'Define a semaphore.'}],24,experimental_speed_projection=False)
            png=base64.b64encode((R/'.ai/runs/0074-checks/shapes_swapped.png').read_bytes()).decode()
            out['vision']=req([{'role':'user','content':[{'type':'text','text':'Describe this image.'},{'type':'image_url','image_url':{'url':'data:image/png;base64,'+png}}]}],64)
        else:
            for k,n in [('code',300),('long',200),('long16k',128)]:
                # Distinct prefix avoids prompt cache reuse between measurements.
                s=f'Measurement {idx} {k}.\n'+tk.decode(prompts[k])
                out[k]=req([{'role':'user','content':s}],n)
    except Exception as e: out['error']=repr(e)
    finally:
        if p and p.poll() is None: p.send_signal(signal.SIGINT)
        if p:
            try: p.wait(timeout=80)
            except subprocess.TimeoutExpired: out['drain_error']=f'process {p.pid} still running'
            out['rc']=p.poll()
out['elapsed']=time.monotonic()-t
now=journal(); delta=now[len(base):] if now.startswith(base) else now
(D/f'{name}-kernel-delta.log').write_text(delta)
out['fault_lines']=[x for x in delta.splitlines() if any(s in x.lower() for s in ('xid','gfxhub','gcvm','gpu fault','page fault','amdgpu: ring','gpu reset'))]
(D/f'{name}-result.json').write_text(json.dumps(out,indent=1))
print(name,'elapsed',round(out['elapsed'],1),'rc',out.get('rc'),'error',out.get('error'),'faults',out['fault_lines'],flush=True)
for k in ('code','long','long16k','first','second','vision','cvec_off'):
 if k in out: print(k,out[k].get('timings'),flush=True)
if out.get('error') or out.get('fault_lines') or out.get('rc')!=0 or out['elapsed']>=120: sys.exit(1)
