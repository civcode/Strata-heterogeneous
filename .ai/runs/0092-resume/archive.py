import os, pathlib, subprocess, shutil, hashlib
base=pathlib.Path('/home/chris/workspace'); out=base/'strata-local-archive'
for name in ['Strata-0084-archive','Strata-hardened-0072','Strata-mixed-native']:
 p=base/name; dst=out/name; dst.mkdir(parents=True,exist_ok=True)
 files=set()
 for flags in [[],['--ignored']]:
  b=subprocess.check_output(['git','-C',str(p),'ls-files','-z','--others','--exclude-standard',*flags])
  files.update(os.fsdecode(x) for x in b.split(b'\0') if x)
 copied=[]
 for f in sorted(files):
  q=pathlib.PurePosixPath(f)
  if any(x.startswith('build') or x in ('.venv','__pycache__') for x in q.parts[:-1]) or f.endswith('.o'): continue
  src=p/f; target=dst/f; target.parent.mkdir(parents=True,exist_ok=True)
  if target.exists() or target.is_symlink(): raise RuntimeError(f'already exists: {target}')
  if src.is_symlink(): target.symlink_to(os.readlink(src))
  elif src.is_file():
   shutil.copy2(src,target)
   def digest(x):
    h=hashlib.sha256()
    with open(x,'rb') as fh:
     for chunk in iter(lambda:fh.read(1024*1024),b''): h.update(chunk)
    return h.digest()
   if digest(src)!=digest(target): raise RuntimeError(f'copy mismatch: {src}')
  else: raise RuntimeError(f'not a file: {src}')
  copied.append(f)
 pathlib.Path(f'.ai/runs/0092-resume/{name}.copied').write_text('\n'.join(copied)+'\n')
 print(name, len(copied),flush=True)
