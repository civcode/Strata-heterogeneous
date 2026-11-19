import sys,os,time,json,subprocess,pathlib,threading,re
sys.path.insert(0,'mixed_split/tools')
from mixed_split_common import reset_ignored_signals,stop_cooperatively,force_sdma_off
from check_launch import check_config
reset_ignored_signals(); force_sdma_off(os.environ)
name=sys.argv[1]; mode=name.split('-')[0]; root=pathlib.Path('.ai/runs/0093-exact').resolve(); cfg=json.loads(pathlib.Path('.ai/runs/0093-measure/iq-'+('serial-new' if mode=='serial' else 'pipe-final-new')+'/config.json').read_text())
if 'matched' in name or 'tailonly' in name: cfg['env']['STRATA_ROUTE_TAIL_SKIP']='0'
if ('matched' in name or 'equalcache' in name) and mode=='serial':
 cfg['args'] += ['--front-arg','--vram-reserve-mib','--front-arg','956']
check_config(cfg)
assert pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()=='ffcbbb74-121f-46fa-a930-ca08f55e881c'; assert not pathlib.Path('.ai/runs/0093-safety/STOP').exists()
env=os.environ.copy(); env.update(cfg['env']); env.update(STRATA_IQ_MT_MIN='1',HIP_VISIBLE_DEVICES=','.join(map(str,cfg['gpu'])),LD_LIBRARY_PATH=':'.join(cfg['lib_dirs'])); force_sdma_off(env)
argv=[cfg['exe'],'--serve']+cfg['args']+['--layer-split',cfg['layer_split']]
cmd=[sys.executable,'mixed_split/tools/engine_drive.py',str(root/name),str(root/'prompts.json'),'code:300','--']+argv
stop=threading.Event(); pat=re.compile(r'gfxhub|GCVM|page fault|segfault|general protection|Bad page|Bad swap|Oops|BUG:|rcu.*stall|amdgpu.*(fault|reset|timeout|hang)|ring .* timeout|MES failed|Xid|NVRM',re.I)
j=subprocess.Popen(['journalctl','-k','-f','-n','0','--no-pager'],stdout=subprocess.PIPE,text=True)
def watch():
 for l in j.stdout:
  if pat.search(l): (root/(name+'-stop.txt')).write_text(l); pathlib.Path('.ai/runs/0093-safety/STOP').write_text(l); stop.set()
threading.Thread(target=watch,daemon=True).start(); t=time.monotonic()
with (root/(name+'.driver.log')).open('w') as f:
 p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
 while p.poll() is None:
  if stop.is_set() or time.monotonic()-t>150:
   stop_cooperatively(p,60,40); break
  log=root/(name+'.err')
  if log.exists() and 'non-finite logits' in log.read_text(errors='replace').lower():
   pathlib.Path('.ai/runs/0093-safety/STOP').write_text('non-finite logits'); stop_cooperatively(p,60,40); break
  time.sleep(.2)
 rc=p.poll(); j.terminate(); j.wait(timeout=5)
 print(mode,'rc',rc,'elapsed',time.monotonic()-t)
 if rc!=0 or stop.is_set(): sys.exit(1)
