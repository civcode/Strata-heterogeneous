# vision.py NAME CONFIG: the vision checks of handoff 0074 through serve/server.py (serve_mixed.py), accounted
import base64, json, signal, subprocess, sys, time, urllib.request
from common import *
name, config = sys.argv[1:3]
port = 8093
pics = {k: base64.b64encode((ROOT / f'tools/qwen35/pictures/{k}.png').read_bytes()).decode() for k in ('shapes', 'code4217')}
# the same size as shapes.png (the same <|image_pad|> tokens), red and blue swapped
pics['shapes_swapped'] = base64.b64encode((RUN / 'shapes_swapped.png').read_bytes()).decode()
def img(k, text):
    return {'role': 'user', 'content': [{'type': 'text', 'text': text}, {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + pics[k]}}]}
def text(r):
    m = r['choices'][0]['message']
    return ((m.get('content') or '') + ' [reasoning] ' + (m.get('reasoning_content') or '')).replace('\n', ' ')
def req(messages, n=100):
    data = {'model': 'x', 'messages': messages, 'temperature': 0, 'max_tokens': n, 'reasoning_budget_tokens': 0}
    r = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=json.dumps(data).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(r, timeout=60) as h: return json.load(h)
base = kernel(); t0 = time.monotonic(); out = []
log = open(RUN / f'{name}.out', 'w')
p = subprocess.Popen([sys.executable, 'hetero/tools/serve_mixed.py', '--config', config, '--port', str(port), '--run-dir', str(RUN / name)],
                     stdout=log, stderr=subprocess.STDOUT, env=ENV)
try:
    while time.monotonic() - t0 < 80 and p.poll() is None:
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as r:
                if json.load(r).get('loaded'): break
        except Exception: pass
        time.sleep(.5)
    else: raise RuntimeError('server not ready')
    q = 'Describe this image briefly, including any text.'
    steps = [('image shapes', [img('shapes', q)]),
             ('image code4217 (same text)', [img('code4217', q)]),
             ('image shapes_swapped (same size, same text)', [img('shapes_swapped', q)]),
             ('text after image', [{'role': 'user', 'content': 'Explain in two sentences how a refrigerator keeps food cold.'}]),
             ('image shapes again (same text)', [img('shapes', q)])]
    t1 = [img('code4217', 'What number is written in the image? Answer briefly.')]
    for label, msgs in steps:
        r = req(msgs); out.append({'step': label, 'resp': r}); print(label, '->', text(r)[:400], r.get('usage'), flush=True)
    r1 = req(t1, 80); out.append({'step': 'turn 1 (code4217)', 'resp': r1}); print('turn1 ->', text(r1)[:400], r1.get('usage'), flush=True)
    t2 = t1 + [{'role': 'assistant', 'content': r1['choices'][0]['message'].get('content') or ''}, {'role': 'user', 'content': 'What colour is the circle in the same image?'}]
    r2 = req(t2, 80); out.append({'step': 'turn 2 (code4217)', 'resp': r2}); print('turn2 ->', text(r2)[:400], r2.get('usage'), flush=True)
except Exception as e:
    out.append({'error': repr(e)}); print('ERROR', repr(e), flush=True)
finally:
    if p.poll() is None: p.send_signal(signal.SIGINT)
    try: p.wait(timeout=100)
    except subprocess.TimeoutExpired: print('STOP: server still running', p.pid, flush=True); sys.exit(1)
(RUN / f'{name}.json').write_text(json.dumps({'rc': p.returncode, 'steps': out}, indent=1))
ok = account(name, time.monotonic() - t0, base)
print('rc', p.returncode, flush=True)
