# perf.py: the long-prompt request through mixed.example.json and hiponly.example.json, untraced, alternating, one
# server start per sample (the 0073 method: a code request of 300 tokens, then the long one of 200), accounted
import json, signal, subprocess, sys, time, urllib.request
from common import *
sys.path.insert(0, str(ROOT / 'tools')); from strata_tokenizer import Tokenizer
tk = Tokenizer.from_gguf(ENV['STRATA_MODEL_DIR'] + '/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf')
doc = (ROOT / 'docs/MULTI_GPU.md').read_text() + (ROOT / 'README.md').read_text()
body = tk.decode(tk.encode(doc)[:4900])
prompts = [('code', 'Write a complete Python implementation of a thread-safe LRU cache, including type hints, documentation, and example usage. Explain the design and important edge cases.', 300),
           ('long', 'Summarise the following document in five bullet points.\n\n' + body, 200)]
port = 8094
for i, arm in enumerate(['mixed', 'hiponly'] * 3):
    name = f'perf-{i}-{arm}'; base = kernel(); t0 = time.monotonic(); res = []
    with open(RUN / f'{name}.out', 'w') as log:
        p = subprocess.Popen([sys.executable, 'hetero/tools/serve_mixed.py', '--config', f'hetero/configs/{arm}.example.json', '--port', str(port),
                              '--run-dir', str(RUN / name)], stdout=log, stderr=subprocess.STDOUT, env=ENV)
        try:
            while time.monotonic() - t0 < 80 and p.poll() is None:
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as r:
                        if json.load(r).get('loaded'): break
                except Exception: pass
                time.sleep(.5)
            else: raise RuntimeError('not ready')
            for label, text, n in prompts:
                data = {'model': 'x', 'messages': [{'role': 'user', 'content': text}], 'temperature': 0, 'max_tokens': n, 'reasoning_budget_tokens': 0}
                r = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=json.dumps(data).encode(), headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(r, timeout=60) as h: d = json.load(h)
                res.append({'label': label, 'usage': d.get('usage'), 'timings': d.get('timings')})
        except Exception as e:
            res.append({'error': repr(e)})
        finally:
            if p.poll() is None: p.send_signal(signal.SIGINT)
            try: p.wait(timeout=100)
            except subprocess.TimeoutExpired: print('STOP: server still running', p.pid, flush=True); sys.exit(1)
    el = time.monotonic() - t0
    (RUN / f'{name}.json').write_text(json.dumps({'rc': p.returncode, 'elapsed': el, 'results': res}, indent=1))
    ok = account(name, el, base)
    for x in res:
        t = x.get('timings') or {}
        print(name, x.get('label'), x.get('error', ''), 'prompt_n', t.get('prompt_n'), 'prefill', t.get('prompt_per_second'), 'decode', t.get('predicted_per_second'), flush=True)
    if not ok or p.returncode or el >= 120 or any('error' in x for x in res) or used() > 1500:
        print('STOP', flush=True); break
