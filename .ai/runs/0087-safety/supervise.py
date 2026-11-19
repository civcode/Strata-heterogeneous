#!/usr/bin/env python3
import os, signal, subprocess, sys, time, re
pattern = re.compile(r'amdgpu|gfxhub|GCVM|MES|page fault|segfault|general protection|Bad page|Oops|rcu.*stall', re.I)
boot = open('/proc/sys/kernel/random/boot_id').read().strip()
def swap():
    d = dict(line.split(':', 1) for line in open('/proc/meminfo') if ':' in line)
    return int(d['SwapTotal'].split()[0]) - int(d['SwapFree'].split()[0])
base = swap()
j = subprocess.run(['journalctl','-k','-b','-n','1','--show-cursor','--no-pager'], capture_output=True, text=True, check=True)
cursor = j.stdout.rsplit('-- cursor: ', 1)[1].strip()
def reset():
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
t = time.monotonic()
p = subprocess.Popen(sys.argv[1:], preexec_fn=reset)
stop = None
while p.poll() is None:
    time.sleep(1)
    j = subprocess.run(['journalctl','-k','-b','--after-cursor='+cursor,'--no-pager','-o','short-iso'], capture_output=True, text=True)
    lines = [x for x in j.stdout.splitlines() if pattern.search(x)]
    if lines: stop = 'kernel match: ' + '\n'.join(lines)
    elif open('/proc/sys/kernel/random/boot_id').read().strip() != boot: stop = 'boot changed'
    elif swap() > base + 1048576: stop = 'swap grew over 1 GiB'
    elif time.monotonic() - t > 570: stop = 'run time limit'
    if stop:
        print('SAFETY STOP:', stop, flush=True)
        p.send_signal(signal.SIGINT)
        break
rc = p.wait()
print('supervisor elapsed %.1fs rc %d stop %s' % (time.monotonic()-t, rc, stop), flush=True)
sys.exit(2 if stop else (rc if rc >= 0 else 1))
