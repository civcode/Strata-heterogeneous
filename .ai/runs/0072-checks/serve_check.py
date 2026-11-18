import base64, json, os, signal, subprocess, sys, time, urllib.request
from pathlib import Path
root=Path.cwd(); run=root/'.ai/runs/0072-checks'; port=8092
sys.path.insert(0,str(root/'hetero/tools'))
from serve_requests import post
log=open(run/'server.out','w'); start=time.monotonic()
p=subprocess.Popen([sys.executable,'hetero/tools/serve_mixed.py','--config','hetero/configs/hiponly.example.json','--port',str(port),'--run-dir',str(run)],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,HSA_ENABLE_SDMA='0'))
try:
 while time.monotonic()-start<65 and p.poll() is None:
  try:
   with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as r:
    if json.load(r).get('loaded'): break
  except Exception: pass
  time.sleep(.5)
 else: raise RuntimeError('server not ready')
 def req(messages,n=80):
  data={'model':'x','messages':messages,'temperature':0,'max_tokens':n,'reasoning_budget_tokens':0}
  r=urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',data=json.dumps(data).encode(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(r,timeout=45) as h: return json.load(h)
 q={'role':'user','content':'Explain in two sentences how a refrigerator keeps food cold.'}
 a=req([q]); print('turn1',json.dumps(a),flush=True)
 b=req([q,a['choices'][0]['message'],{'role':'user','content':'Why does it hum?'}]); print('turn2',json.dumps(b),flush=True)
 img=base64.b64encode((root/'tools/qwen35/pictures/shapes.png').read_bytes()).decode()
 c=req([{'role':'user','content':[{'type':'text','text':'Describe this image briefly.'},{'type':'image_url','image_url':{'url':'data:image/png;base64,'+img}}]}]); print('image',json.dumps(c),flush=True)
 doc=(root/'docs/MULTI_GPU.md').read_text()+(root/'README.md').read_text()
 # Use the same decoded 4900-token body as mkprompts.
 sys.path.insert(0,str(root/'tools')); from strata_tokenizer import Tokenizer
 tk=Tokenizer.from_gguf(os.environ['STRATA_MODEL_DIR']+'/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf')
 body=tk.decode(tk.encode(doc)[:4900]); d=req([{'role':'user','content':'Summarise the following document in five bullet points.\n\n'+body}],200); print('long',json.dumps(d),flush=True)
finally:
 if p.poll() is None: p.send_signal(signal.SIGINT)
 try: p.wait(timeout=100)
 except subprocess.TimeoutExpired: print('server still draining pid',p.pid,flush=True)
 print('server exit',p.returncode,'elapsed',round(time.monotonic()-start,2),flush=True)
