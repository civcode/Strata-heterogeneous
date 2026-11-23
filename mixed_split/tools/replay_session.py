#!/usr/bin/env python3
"""Replay a recorded agent session (record_proxy.py) against a freshly started Strata server, and summarise runs.

    replay_session.py run RECORDING OUT_DIR [--url http://127.0.0.1:8081] [--sampling greedy|recorded]
        [--no-pauses | --pause-scale X] [--limit N] [--sample-s 30]
    replay_session.py summary OUT_DIR [OUT_DIR ...] [--markdown FILE]

`run` sends the recording's chat-completion requests in order, each with its recorded body, so every replay reads the
same prompts at the same context lengths (the history inside each request already holds the replies the agent got
when it was recorded).  Changes to each request:
  - `--sampling greedy` (the default, for profiling): temperature 0, top_p 1, top_k / min_p / typical_p / seed
    removed, seed 0, so the tokens are the same in every run;  `recorded` keeps the agent's settings;
  - `max_tokens` (and `max_completion_tokens`) = the tokens the request generated when recorded, so each request
    decodes about as much as it did then;
  - `stream: true` with usage, so time to first token and Strata's timings come back.
Pauses: by default the replay waits between requests as long as the agent did (its tool runs; Strata's expert cache
adapts then); `--no-pauses` drops them, `--pause-scale` scales them.
Start the server fresh for each run (the replay begins with an empty cache, as the session did); stop it with Ctrl-C /
SIGINT afterwards.  Output in OUT_DIR: requests.jsonl (one line per request: prompt and cached tokens, generated
tokens, time to first token, prefill and decode tok/s, drafts, chunk counts per phase, whether prompt_tokens matched
the recording), samples.jsonl (memory and VRAM, agent_session.py's sampler), summary.json.

`summary` prints, per run directory: the totals in the Human User's format (requests, generated_tokens,
decode_seconds, engine_decode_tps, draft_acceptance_pct), a phase table (thinking / answering / writing tool calls:
tokens counted as stream chunks, time as the gaps before each chunk), and the median over the runs given.
"""
import argparse, json, os, statistics, sys, threading, time, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from agent_session import sampler  # noqa: E402  (the memory/VRAM sampler)

SAMPLING_DROP = ('top_k', 'min_p', 'typical_p', 'repetition_penalty', 'presence_penalty', 'frequency_penalty',
                 'logit_bias', 'n')


def load(recording):
    recs = [json.loads(l) for l in open(os.path.expanduser(recording)) if l.strip()]
    return [r for r in recs if not r.get('header') and r.get('method') == 'POST'
            and r.get('path', '').rstrip('/').endswith('/chat/completions') and r.get('request')]


def prepare(r: dict, sampling: str) -> bytes:
    body = json.loads(r['request'])
    gen = (r.get('usage') or {}).get('completion_tokens')
    if gen:
        body['max_tokens'] = int(gen)
        if 'max_completion_tokens' in body:
            body['max_completion_tokens'] = int(gen)
    if sampling == 'greedy':
        for k in SAMPLING_DROP:
            body.pop(k, None)
        body.update(temperature=0, top_p=1, seed=0)
    body['stream'] = True
    body['stream_options'] = {'include_usage': True}
    return json.dumps(body, ensure_ascii=False).encode()


def send(url, path, raw, api_key=None):
    """One streamed request: per-chunk times and kinds, usage, timings."""
    h = {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key or "dummy"}'}
    req = urllib.request.Request(url.rstrip('/') + path, data=raw, headers=h)
    t0 = time.monotonic()
    res = dict(usage=None, timings=None, finish=None, error=None, chunks=[])
    try:
        with urllib.request.urlopen(req, timeout=7200) as resp:
            for line in resp:
                line = line.strip()
                if not line.startswith(b'data:') or line == b'data: [DONE]':
                    continue
                ev = json.loads(line[5:])
                if ev.get('error'):
                    res['error'] = ev['error']
                res['usage'] = ev.get('usage') or res['usage']
                res['timings'] = ev.get('timings') or res['timings']
                for c in ev.get('choices') or []:
                    d = c.get('delta') or {}
                    kind = 'reasoning' if d.get('reasoning_content') or d.get('reasoning') else \
                        'tool_calls' if d.get('tool_calls') else 'content' if d.get('content') else None
                    if kind:
                        res['chunks'].append((round(time.monotonic() - t0, 4), kind))
                    res['finish'] = c.get('finish_reason') or res['finish']
    except Exception as e:  # noqa: BLE001 - recorded; the run goes on, the summary counts it
        res['error'] = f'{type(e).__name__}: {e}'
    res['wall_s'] = round(time.monotonic() - t0, 3)
    return res


def phases(chunks):
    """Tokens (chunks) and seconds per phase; a chunk's time is the gap since the previous chunk (the first one's
    gap is the time to first token, which is prefill, so it is left out)."""
    out = {k: [0, 0.0] for k in ('reasoning', 'content', 'tool_calls')}
    for i, (t, kind) in enumerate(chunks):
        out[kind][0] += 1
        if i:
            out[kind][1] += t - chunks[i - 1][0]
    return {k: {'chunks': v[0], 'seconds': round(v[1], 4)} for k, v in out.items()}


def cmd_run(a):
    recs = load(a.recording)[: a.limit or None]
    if not recs:
        raise SystemExit('replay_session: no chat-completion requests in ' + a.recording)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    th = None
    if a.sample_s > 0:
        th = threading.Thread(target=sampler, args=(str(out / 'samples.jsonl'), a.sample_s, a.rss_match, stop), daemon=True)
        th.start()
    t_run = time.time()
    errors = 0
    with open(out / 'requests.jsonl', 'w') as h:
        for i, r in enumerate(recs):
            if i and not a.no_pauses:
                gap = (r.get('t_start') or 0) - (recs[i - 1].get('t_end') or 0)
                if gap > 0:
                    time.sleep(gap * a.pause_scale)
            res = send(a.url, r['path'], prepare(r, a.sampling), os.environ.get(a.api_key_env) if a.api_key_env else None)
            u, t = res['usage'] or {}, res['timings'] or {}
            ru = r.get('usage') or {}
            line = dict(n=i + 1, seq=r.get('seq'), prompt_tokens=u.get('prompt_tokens'),
                        prompt_tokens_recorded=ru.get('prompt_tokens'),
                        prompt_matches=u.get('prompt_tokens') == ru.get('prompt_tokens') if ru.get('prompt_tokens') else None,
                        cached_tokens=(u.get('prompt_tokens_details') or {}).get('cached_tokens'),
                        completion_tokens=u.get('completion_tokens'), completion_recorded=ru.get('completion_tokens'),
                        ttft_s=res['chunks'][0][0] if res['chunks'] else None, wall_s=res['wall_s'],
                        prompt_tok_s=t.get('prompt_per_second'), prompt_ms=t.get('prompt_ms'), prompt_n=t.get('prompt_n'),
                        decode_tok_s=t.get('predicted_per_second'), predicted_ms=t.get('predicted_ms'),
                        predicted_n=t.get('predicted_n'), draft_n=t.get('draft_n'), draft_accepted=t.get('draft_n_accepted'),
                        phases=phases(res['chunks']), finish=res['finish'], error=res['error'])
            errors += bool(res['error'])
            h.write(json.dumps(line) + '\n')
            h.flush()
            print(f"#{i + 1:3d}/{len(recs)} prompt {line['prompt_tokens']} (rec {line['prompt_tokens_recorded']}) "
                  f"cached {line['cached_tokens']} gen {line['completion_tokens']} ttft {line['ttft_s']} "
                  f"dec {line['decode_tok_s']}" + (f" ERROR {res['error']}" if res['error'] else ''), flush=True)
    stop.set()
    if th:
        th.join(timeout=30)
    s = summarise(out)
    s.update(recording=os.path.expanduser(a.recording), sampling=a.sampling, pauses=not a.no_pauses,
             pause_scale=a.pause_scale, wall_s=round(time.time() - t_run, 1), errors=errors)
    json.dump(s, open(out / 'summary.json', 'w'), indent=1)
    print(json.dumps({k: s[k] for k in ('requests', 'generated_tokens', 'decode_seconds', 'engine_decode_tps',
                                         'draft_acceptance_pct', 'errors')}, indent=1))
    return 0 if errors == 0 else 1


def summarise(d: Path) -> dict:
    rows = [json.loads(l) for l in open(d / 'requests.jsonl') if l.strip()]
    gen = sum(r['predicted_n'] or r['completion_tokens'] or 0 for r in rows)
    dec_s = sum((r['predicted_ms'] or 0) for r in rows) / 1000
    dn, da = sum(r['draft_n'] or 0 for r in rows), sum(r['draft_accepted'] or 0 for r in rows)
    ph = {k: {'chunks': sum(r['phases'][k]['chunks'] for r in rows), 'seconds': sum(r['phases'][k]['seconds'] for r in rows)}
          for k in ('reasoning', 'content', 'tool_calls')}
    for v in ph.values():
        v['tok_s'] = round(v['chunks'] / v['seconds'], 2) if v['seconds'] else None
    return dict(requests=len(rows), generated_tokens=gen, decode_seconds=round(dec_s, 3),
                engine_decode_tps=round(gen / dec_s, 2) if dec_s else None,
                draft_acceptance_pct=round(100 * da / dn, 2) if dn else None,
                prompt_mismatches=sum(1 for r in rows if r['prompt_matches'] is False),
                ttft_median_s=round(statistics.median([r['ttft_s'] for r in rows if r['ttft_s'] is not None]), 3)
                if any(r['ttft_s'] is not None for r in rows) else None,
                phases=ph)


def cmd_summary(a):
    runs = [(Path(d).name, summarise(Path(d))) for d in a.dirs]
    names = {'reasoning': 'thinking', 'content': 'answering', 'tool_calls': 'writing tool calls'}
    lines = ['| run | requests | generated | decode s | engine decode tok/s | draft acceptance % | ttft median s | '
             'thinking tok/s | answering tok/s | tool calls tok/s | prompt mismatches |', '|' + '---|' * 11]
    for n, s in runs:
        p = s['phases']
        lines.append(f"| {n} | {s['requests']} | {s['generated_tokens']} | {s['decode_seconds']} | {s['engine_decode_tps']} | "
                     f"{s['draft_acceptance_pct']} | {s['ttft_median_s']} | {p['reasoning']['tok_s']} | {p['content']['tok_s']} | "
                     f"{p['tool_calls']['tok_s']} | {s['prompt_mismatches']} |")
    if len(runs) > 1:
        def m(f):
            xs = [f(s) for _, s in runs if f(s) is not None]
            return round(statistics.median(xs), 2) if xs else None
        lines.append(f"| **median** | | | | {m(lambda s: s['engine_decode_tps'])} | {m(lambda s: s['draft_acceptance_pct'])} | "
                     f"{m(lambda s: s['ttft_median_s'])} | {m(lambda s: s['phases']['reasoning']['tok_s'])} | "
                     f"{m(lambda s: s['phases']['content']['tok_s'])} | {m(lambda s: s['phases']['tool_calls']['tok_s'])} | |")
    text = '\n'.join(lines) + '\n\nPhases: tokens counted as stream chunks; ' + ', '.join(f'{k} = {v}' for k, v in names.items()) + '.\n'
    if a.markdown:
        open(a.markdown, 'w').write(text)
    print(text)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('recording')
    r.add_argument('out')
    r.add_argument('--url', default='http://127.0.0.1:8081')
    r.add_argument('--sampling', choices=('greedy', 'recorded'), default='greedy')
    r.add_argument('--no-pauses', action='store_true')
    r.add_argument('--pause-scale', type=float, default=1.0)
    r.add_argument('--limit', type=int, default=0)
    r.add_argument('--sample-s', type=float, default=30)
    r.add_argument('--rss-match', default=r'strata|server\.py|front')
    r.add_argument('--api-key-env', default=None)
    s = sub.add_parser('summary')
    s.add_argument('dirs', nargs='+')
    s.add_argument('--markdown')
    a = ap.parse_args()
    return cmd_run(a) if a.cmd == 'run' else cmd_summary(a)


if __name__ == '__main__':
    sys.exit(main())
