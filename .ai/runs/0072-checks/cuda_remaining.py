import json, os, re, subprocess, time
from pathlib import Path
root=Path.cwd(); out=root/'.ai/runs/0072-checks'; old=root/'.ai/runs/0071-tests'
alln=[m.group(1) for l in open(old/'cuda-list.txt') if (m:=re.search(r'Test\s+#\d+:\s+(\S+)',l))]
done=set()
for f in ['cuda-first.log','cuda-prefill.log']:
 for l in open(old/f):
  if m:=re.search(r'Test\s+#\d+:\s+(\S+)',l): done.add(m.group(1))
left=[n for n in alln if n not in done and n!='ple_parity']; total=243.84; results=[]
env=dict(os.environ,HIP_VISIBLE_DEVICES='0,1',HSA_ENABLE_SDMA='0',LD_LIBRARY_PATH='/opt/rocm/core-10.1/lib')
base=subprocess.check_output(['journalctl','-k','-b','--no-pager'],text=True)
for name in left:
 if total+120>1500: print('budget limit',flush=True); break
 start=time.monotonic(); r=subprocess.run(['ctest','--test-dir',str(root/'build-0071-cuda'),'-R','^'+re.escape(name)+'$','--output-on-failure'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env); elapsed=time.monotonic()-start; total+=elapsed
 result={'name':name,'rc':r.returncode,'seconds':round(elapsed,2),'skip':'(Skipped)' in r.stdout,'output':r.stdout}; results.append(result); json.dump(results,open(out/'cuda-results.json','w'),indent=1)
 print(name,'rc',r.returncode,'skip',result['skip'],'seconds',round(elapsed,2),'budget',round(total,2),flush=True)
 now=subprocess.check_output(['journalctl','-k','-b','--no-pager'],text=True); delta=now[len(base):] if now.startswith(base) else now; base=now
 if any(x in delta.lower() for x in ['xid','gfxhub','gcvm','gpu fault']): print('NEW GPU FAULT STOP',flush=True); break
 if elapsed>=120: print('run exceeded limit STOP',flush=True); break
json.dump([n for n in left if n not in {r['name'] for r in results}],open(out/'cuda-not-run.json','w'),indent=1)
