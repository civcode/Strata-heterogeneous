#!/usr/bin/env python3
"""One accounted server session: start serve/server.py through serve_mixed.py, run a suite, stop it cooperatively.

    serve_session.py RUN_DIR NAME CONFIG SUITE [--prompts PROMPTS_JSON] [--port 8096] [--tag TEXT]

SUITE:
  checks    server_checks.py (stream, non-stream, sampled, a cancelled request and the next one, two turns, readers)
  features  a two-turn chat over the 16K prompt with another request between the turns (the first conversation is
            parked and restored), a request with cvec=0 (experimental_speed_projection false) and a PNG vision request
  measure   the code, long and long16k prompts (300 / 200 / 128 tokens), each with "Measurement TAG NAME." in front, so
            no request reuses another's prefix (report 0076)
  pack-block  handoff 0082: code, long and long16k, each once warm-up then twice measured, greedy and uncached
  cancel    report 0079: a streamed long32k request whose client leaves while the host's first stage reads the chunk
            at --cancel-at (default 0; seen in the server log's STRATA_TRACE lines, so run it with STRATA_TRACE=1),
            --cancel-after seconds (default 2.0) after that chunk's trace line; then the same conversation with one
            more turn, then an unrelated request.  The server log's lines for each go into the result.
  concurrent  handoff 0081: --clients (2) clients at once, each sending the code (300 tokens) and then the long prompt
            (200 tokens), --rounds (3) times, streamed; aggregate decode tok/s (tokens / the wall time with at least one
            request decoding), per-request decode tok/s and time to first token
  warm      report 0079: the code prompt three times (16 tokens each), each with "Warm TAG i." in front, so none reuses
            another's prefix; the second and third are a warm server's short-prompt first token
PROMPTS_JSON: mkprompts.py's token prompts (default RUN_DIR/prompts.json).  The server gets SIGINT at the end and its
drain is awaited (never killed).  Writes RUN_DIR/NAME-result.json, -console.log, -kernel-delta.log, the server's run
directory RUN_DIR/NAME/, and adds the session's wall time to RUN_DIR/gpu-seconds.txt.  Exit status 0 only when every
request succeeded, the server ended with status 0 within 120 s and no GPU fault line appeared in the kernel log.
"""
import argparse, base64, json, os, signal, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, 'tools'))
FAULTS = ('xid', 'gfxhub', 'gcvm', 'gpu fault', 'page fault', 'amdgpu: ring', 'gpu reset')
DEFAULTS = dict(STRATA_DATA='/home/chris/workspace/Strata-data', STRATA_MODEL_DIR='/home/chris/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF',
                STRATA_PROFILE='/home/chris/workspace/Strata/data/expert-profile.bin', STRATA_VANILLA='/home/chris/workspace/Strata',
                ROCM_LIB='/opt/rocm/core-10.1/lib', HSA_ENABLE_SDMA='0', HIP_VISIBLE_DEVICES='0,1')
PNG = os.path.join(REPO, '.ai/runs/0074-checks/shapes_swapped.png')


def journal():
    return subprocess.check_output(['journalctl', '-k', '-b', '--no-pager'], text=True, timeout=30)


def stream_req(port, content, n):
    """One streamed greedy request; client-side times of the first and the last content chunk."""
    d = dict(model='x', messages=[{'role': 'user', 'content': content}], temperature=0, max_tokens=n,
             reasoning_budget_tokens=0, stream=True, stream_options={'include_usage': True})
    r = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=json.dumps(d).encode(),
                               headers={'Content-Type': 'application/json'})
    t0 = time.monotonic()
    first = lastc = None
    usage = timings = None
    with urllib.request.urlopen(r, timeout=110) as h:
        for line in h:
            line = line.strip()
            if not line.startswith(b'data:') or line == b'data: [DONE]':
                continue
            ev = json.loads(line[5:])
            usage = ev.get('usage') or usage
            timings = ev.get('timings') or timings
            for c in ev.get('choices', []):
                dl = c.get('delta', {})
                if dl.get('content') or dl.get('reasoning_content'):
                    now = time.monotonic()
                    first = now if first is None else first
                    lastc = now
    ntok = (usage or {}).get('completion_tokens') or (timings or {}).get('predicted_n') or 0
    return dict(t0=t0, t_first=first, t_last=lastc, usage=usage, timings=timings, n=ntok)


def concurrent_load(port, tk, prompts, tag, clients, rounds):
    """handoff 0081: `clients` threads at once, each sending the code prompt (300 tokens) and then the long prompt
    (200 tokens), `rounds` times; every request has its own prefix so none reuses another's.  Aggregate decode tok/s =
    generated tokens / the wall time during which at least one request was between its first and last token."""
    import threading
    res, lock = [], threading.Lock()

    def client(c):
        for r in range(rounds):
            for k, n in (('code', 300), ('long', 200)):
                try:
                    x = stream_req(port, f'Load {tag} client {c} round {r} {k}.\n' + tk.decode(prompts[k]), n)
                except Exception as e:  # noqa: BLE001
                    x = dict(error=repr(e))
                x.update(client=c, round=r, prompt=k)
                with lock:
                    res.append(x)
    t0 = time.monotonic()
    th = [threading.Thread(target=client, args=(c,)) for c in range(clients)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    wall = time.monotonic() - t0
    ok = [x for x in res if 'error' not in x and x.get('t_first') is not None]
    iv = sorted((x['t_first'], x['t_last']) for x in ok)
    busy, cur = 0.0, None
    for s, e in iv:
        if cur is None or s > cur[1]:
            busy += (cur[1] - cur[0]) if cur else 0.0
            cur = [s, e]
        else:
            cur[1] = max(cur[1], e)
    busy += (cur[1] - cur[0]) if cur else 0.0
    gen = sum(x['n'] for x in ok)
    for x in res:
        if 'error' not in x:
            x['ttft_s'] = x['t_first'] - x['t0'] if x.get('t_first') else None
            x['client_decode_tps'] = (x['n'] - 1) / (x['t_last'] - x['t_first']) \
                if x.get('t_first') and x['t_last'] > x['t_first'] else None
            for k in ('t0', 't_first', 't_last'):
                x[k] = None if x.get(k) is None else x[k] - t0
    return dict(wall_s=wall, decode_busy_s=busy, generated=gen, aggregate_decode_tps=gen / busy if busy else None,
                aggregate_output_tps=gen / wall, requests=res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run'); ap.add_argument('name'); ap.add_argument('config'); ap.add_argument('suite')
    ap.add_argument('--prompts'); ap.add_argument('--port', type=int, default=8096); ap.add_argument('--tag', default='0')
    ap.add_argument('--clients', type=int, default=2); ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--cancel-at', type=int, default=0); ap.add_argument('--cancel-after', type=float, default=2.0)
    a = ap.parse_args()
    run = os.path.abspath(a.run)
    os.makedirs(run, exist_ok=True)
    env = dict(DEFAULTS, **os.environ)
    from strata_tokenizer import Tokenizer
    tk = Tokenizer.from_gguf(env['STRATA_MODEL_DIR'] + '/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf')
    prompts = json.load(open(a.prompts or os.path.join(run, 'prompts.json')))
    port = a.port

    def req(messages, n=128, **kw):
        d = dict(model='x', messages=messages, temperature=0, max_tokens=n, reasoning_budget_tokens=0, **kw)
        r = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=json.dumps(d).encode(),
                                   headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(r, timeout=55) as h:
            return json.load(h)

    base = journal()
    t = time.monotonic()
    out, p = {}, None
    with open(os.path.join(run, a.name + '-console.log'), 'w') as log:
        try:
            p = subprocess.Popen([sys.executable, os.path.join(HERE, 'serve_mixed.py'), '--config', a.config, '--port', str(port),
                                  '--run-dir', os.path.join(run, a.name)], stdout=log, stderr=subprocess.STDOUT, env=env, cwd=REPO)
            while time.monotonic() - t < 65 and p.poll() is None:
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as h:
                        if json.load(h).get('loaded'):
                            break
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(0.5)
            else:
                raise RuntimeError(f'not ready; rc={p.poll()}')
            out['ready_s'] = time.monotonic() - t
            if a.suite == 'checks':
                cfg = json.load(open(os.path.join(run, a.name, os.path.basename(a.config).replace('.example', ''))))
                rc = subprocess.call([sys.executable, os.path.join(HERE, 'server_checks.py'), str(port), cfg['model_name'], cfg['log'],
                                      os.path.join(run, a.name + '-checks.json')], stdout=log, stderr=subprocess.STDOUT, timeout=100)
                chk = json.load(open(os.path.join(run, a.name + '-checks.json')))
                out['checks_rc'] = rc
                out['checks_errors'] = [k for k, v in chk.items() if isinstance(v, dict) and ('error' in v or v.get('status', 200) != 200)]
            if a.suite == 'features':
                first_msg = {'role': 'user', 'content': tk.decode(prompts['long16k'])}
                out['first'] = req([first_msg], 64)
                out['interference'] = req([{'role': 'user', 'content': 'What is a mutex?'}], 24)
                msg = out['first']['choices'][0]['message']
                out['second'] = req([first_msg, {'role': 'assistant', **msg}, {'role': 'user', 'content': 'Now give a one sentence summary.'}], 64)
                out['cvec_off'] = req([{'role': 'user', 'content': 'Define a semaphore.'}], 24, experimental_speed_projection=False)
                png = base64.b64encode(open(PNG, 'rb').read()).decode()
                out['vision'] = req([{'role': 'user', 'content': [{'type': 'text', 'text': 'Describe this image.'},
                                                                   {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + png}}]}], 64)
            elif a.suite == 'cancel':
                import http.client, re
                cfg = json.load(open(os.path.join(run, a.name, os.path.basename(a.config).replace('.example', ''))))
                logp = cfg['log']
                at = os.path.getsize(logp)

                def log_since(pos):
                    with open(logp, errors='replace') as f:
                        f.seek(pos)
                        return f.read()
                user = {'role': 'user', 'content': tk.decode(prompts['long32k'])}
                body = json.dumps(dict(model='x', messages=[user], temperature=0, max_tokens=32, reasoning_budget_tokens=0, stream=True))
                c = http.client.HTTPConnection('127.0.0.1', port, timeout=55)
                c.request('POST', '/v1/chat/completions', body=body, headers={'Content-Type': 'application/json'})
                # the host's first stage starts the chunk at --cancel-at (trace line), then the client leaves
                mark = re.compile(r'strata trace: prompt chunk %d of (\d+) stage \[15,' % a.cancel_at)
                t0 = time.monotonic()
                while not mark.search(log_since(at)):
                    if time.monotonic() - t0 > 40:
                        raise RuntimeError('no trace line of the chunk to cancel at (STRATA_TRACE=1?)')
                    time.sleep(0.05)
                time.sleep(a.cancel_after)
                c.sock.shutdown(2)
                c.close()
                out['cancel_sent_s'] = time.monotonic() - t0
                t1 = time.monotonic()
                while 'cancelled' not in log_since(at) and time.monotonic() - t1 < 30:
                    time.sleep(0.2)
                time.sleep(1.0)
                out['cancel_log'] = [x for x in log_since(at).splitlines() if 'prompt chunk' in x or 'strata serve:' in x or 'ERR' in x]
                at = os.path.getsize(logp)
                for k, msgs in (('continue', [user, {'role': 'assistant', 'content': 'Here is a summary.'},
                                              {'role': 'user', 'content': 'Answer in one sentence instead.'}]),
                                ('unrelated', [{'role': 'user', 'content': 'What is a mutex?'}])):
                    try:
                        out[k] = req(msgs, 32)
                    except Exception as e:  # noqa: BLE001
                        out[k] = {'error': repr(e), 'body': getattr(e, 'read', lambda: b'')().decode(errors='replace')}
                    time.sleep(0.5)
                    out[k + '_log'] = [x for x in log_since(at).splitlines() if 'strata serve:' in x or 'ERR' in x or 'front' in x]
                    at = os.path.getsize(logp)
                if any('error' in out[k] for k in ('continue', 'unrelated')):
                    out['error'] = 'a request after the cancel failed'
            elif a.suite == 'warm':
                for i in range(3):
                    out['warm%d' % i] = req([{'role': 'user', 'content': f'Warm {a.tag} {i}.\n' + tk.decode(prompts['code'])}], 16)
            elif a.suite == 'concurrent':
                out['concurrent'] = concurrent_load(port, tk, prompts, a.tag, a.clients, a.rounds)
                if any('error' in r for r in out['concurrent']['requests']):
                    out['error'] = 'a concurrent request failed'
            elif a.suite == 'pack-block':
                # Identical inputs across packs and blocks, distinct prefixes within a block to avoid cache reuse.
                for k, n in [('code', 300), ('long', 200), ('long16k', 128)]:
                    out[k] = []
                    for i in range(3):
                        out[k].append(req([{'role': 'user', 'content': f'Pack comparison {i} {k}.\n' + tk.decode(prompts[k])}], n))
            elif a.suite == 'measure':
                for k, n in [('code', 300), ('long', 200), ('long16k', 128)]:
                    out[k] = req([{'role': 'user', 'content': f'Measurement {a.tag} {k}.\n' + tk.decode(prompts[k])}], n)
        except Exception as e:  # noqa: BLE001
            out['error'] = repr(e)
        finally:
            if p and p.poll() is None:
                p.send_signal(signal.SIGINT)
            if p:
                try:
                    p.wait(timeout=80)
                except subprocess.TimeoutExpired:
                    out['drain_error'] = f'process {p.pid} still running'
                out['rc'] = p.poll()
    out['elapsed'] = time.monotonic() - t
    tot_p = os.path.join(run, 'gpu-seconds.txt')
    tot = (float(open(tot_p).read()) if os.path.exists(tot_p) else 0.0) + out['elapsed']
    open(tot_p, 'w').write('%.1f\n' % tot)
    now = journal()
    delta = now[len(base):] if now.startswith(base) else now
    open(os.path.join(run, a.name + '-kernel-delta.log'), 'w').write(delta)
    out['fault_lines'] = [x for x in delta.splitlines() if any(s in x.lower() for s in FAULTS)]
    json.dump(out, open(os.path.join(run, a.name + '-result.json'), 'w'), indent=1)
    print(a.name, 'elapsed %.1f (GPU total %.1f)' % (out['elapsed'], tot), 'rc', out.get('rc'), 'error', out.get('error'),
          'checks', out.get('checks_rc'), out.get('checks_errors'), 'faults', out['fault_lines'], flush=True)
    if 'concurrent' in out:
        c = out['concurrent']
        print('  concurrent wall %.1f s, decode busy %.1f s, %d tokens, aggregate decode %.1f tok/s, output %.1f tok/s'
              % (c['wall_s'], c['decode_busy_s'], c['generated'], c['aggregate_decode_tps'] or 0, c['aggregate_output_tps']), flush=True)
    for k in ('warm0', 'warm1', 'warm2', 'code', 'long', 'long16k', 'first', 'second', 'interference', 'vision', 'cvec_off'):
        if k in out:
            values = out[k] if isinstance(out[k], list) else [out[k]]
            for value in values:
                print('  %-12s %s' % (k, value.get('timings')), flush=True)
    bad = out.get('error') or out.get('fault_lines') or out.get('rc') != 0 or out['elapsed'] >= 120 or out.get('checks_errors') or out.get('checks_rc')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
