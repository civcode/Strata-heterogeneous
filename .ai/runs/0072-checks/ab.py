import json, os, subprocess, sys, time
from pathlib import Path
root=Path.cwd(); archive=Path('/home/chris/workspace/Strata-hardened-0072'); out=root/'.ai/runs/0072-checks'; py=str(root/'.venv/bin/python'); prompts=str(root/'runs69/prompts.json'); model='/home/chris/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Q2_0/'
args=['--serve','--pack','/home/chris/workspace/Strata-data/packs/q2_0','--native',model+'Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf','--ple-gguf',model+'Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf','--expert-profile','/home/chris/workspace/Strata/data/expert-profile.bin','--expert-cache','auto','--prefill','auto','--max-context','196608','--kv','int8','--vram-reserve-mib','700','--spec','4','--spec-min-p','0.5','--mtp','/home/chris/workspace/Strata-data/mtp/rt','--layer-split','33']
env=dict(os.environ,HIP_VISIBLE_DEVICES='0,1',HSA_ENABLE_SDMA='0',LD_LIBRARY_PATH='/opt/rocm/core-10.1/lib')
base=subprocess.check_output(['journalctl','-k','-b','--no-pager'],text=True); total=34.09
for i,kind in enumerate('ABABAB',1):
 if total+120>1500: print('budget stop',flush=True); break
 tree=root if kind=='A' else archive; hb='build-0071-hip' if kind=='A' else 'build-0072-hip'; cb='build-0071-cuda' if kind=='A' else 'build-0072-cuda'; prefix=out/f'{i}-{kind}'
 cmd=[py,str(tree/'hetero/tools/engine_drive.py'),str(prefix),prompts,'long:200','--shared-arena','--',str(tree/hb/'strata')]+args+['--front-exe',str(tree/cb/'strata'),'--front-layers','15']
 t=time.monotonic()
 with open(str(prefix)+'.driver.log','w') as f: r=subprocess.run(cmd,env=env,cwd=tree,stdout=f,stderr=subprocess.STDOUT)
 elapsed=time.monotonic()-t; total+=elapsed; print(kind,i,'exit',r.returncode,'elapsed',round(elapsed,2),'total',round(total,2),flush=True)
 now=subprocess.check_output(['journalctl','-k','-b','--no-pager'],text=True); delta=now[len(base):] if now.startswith(base) else now
 open(str(prefix)+'.kernel-delta.log','w').write(delta)
 if any(x in delta.lower() for x in ['xid','gfxhub','gcvm','gpu fault']): print('NEW GPU FAULT STOP',flush=True); break
 base=now
 if r.returncode or elapsed>=120: print('run failed or exceeded limit; stop',flush=True); break
 d=json.load(open(str(prefix)+'.json')); print(json.dumps({k:v for k,v in d['requests'][0].items() if k!='tokens'}),flush=True)
