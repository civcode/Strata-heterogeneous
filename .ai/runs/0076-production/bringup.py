import json, os, signal, subprocess, sys, time, urllib.request
from pathlib import Path
R=Path.cwd(); D=R/'.ai/runs/0076-production'; mode=sys.argv[1]; port=8096
E=dict(os.environ, STRATA_DATA='/home/chris/workspace/Strata-data', STRATA_MODEL_DIR='/home/chris/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF', STRATA_PROFILE='/home/chris/workspace/Strata/data/expert-profile.bin', STRATA_VANILLA='/home/chris/workspace/Strata', ROCM_LIB='/opt/rocm/core-10.1/lib', HSA_ENABLE_SDMA='0', HIP_VISIBLE_DEVICES='0,1')
def snapshot(name):
    with open(D/f'{mode}-{name}-memory.txt','w') as f:
        for cmd in (['free','-h'],['nvidia-smi'],['amd-smi','metric','-m']):
            try: f.write(subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=8).stdout+'\n')
            except Exception as e: f.write(repr(e)+'\n')
def request(msg,n):
    data={'model':'x','messages':[{'role':'user','content':msg}],'temperature':0,'max_tokens':n,'reasoning_budget_tokens':0}
    req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',data=json.dumps(data).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=40) as h: return json.load(h)
t=time.monotonic(); result={}; p=None
with open(D/f'{mode}-console.log','w') as log:
    try:
        p=subprocess.Popen([sys.executable,'hetero/tools/serve_mixed.py','--config',f'hetero/configs/{mode}.example.json','--port',str(port),'--run-dir',str(D/mode)],stdout=log,stderr=subprocess.STDOUT,env=E)
        while time.monotonic()-t<65 and p.poll() is None:
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as h:
                    if json.load(h).get('loaded'): break
            except Exception: pass
            time.sleep(.5)
        else: raise RuntimeError(f'not ready; rc={p.poll()}')
        snapshot('idle')
        result['request']=request('Explain briefly what a mutex is.',32)
        sys.path.insert(0,str(R/'tools')); from strata_tokenizer import Tokenizer
        tk=Tokenizer.from_gguf(E['STRATA_MODEL_DIR']+'/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf')
        prompts=json.load(open(D/'prompts.json'))
        result['long16k']=request(tk.decode(prompts['long16k']),32)
        snapshot('after16k')
    except Exception as e: result['error']=repr(e)
    finally:
        if p and p.poll() is None: p.send_signal(signal.SIGINT)
        if p:
            try: p.wait(timeout=80)
            except subprocess.TimeoutExpired: result['drain_error']=f'process {p.pid} still running'; print(result['drain_error'],flush=True)
            result['rc']=p.poll()
result['elapsed']=time.monotonic()-t
(D/f'{mode}-result.json').write_text(json.dumps(result,indent=1))
print(json.dumps({'mode':mode,'error':result.get('error'),'rc':result.get('rc'),'elapsed':result['elapsed']}),flush=True)
