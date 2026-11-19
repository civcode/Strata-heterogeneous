#!/usr/bin/env python3
"""A synthetic pi coding-agent session against a running Strata server (`.ai/RC.md`, the release-candidate workload).

    agent_session.py run OUT_DIR [--url http://127.0.0.1:8096] [--scaled] [--seed 85] [model flags ...]
    agent_session.py replay PROBE_JSON [--url ...]       one saved probe against a (fresh) server; exit 0 when equal
    agent_session.py stub [--port 8099]                  a fake server for bookkeeping checks (no GPU, no model)
    agent_session.py selftest                            two scaled sessions against the stub: same requests, pruning,
                                                         compactions, request cap, probes and replay

The driver only talks to the server's OpenAI endpoint; start and stop the server as for serve_session.py (through
serve_mixed.py / check_launch.py).  Every length is planned in characters from a fixed seed (`--chars-per-token`
converts the model's token targets), and every decision that depends on the server uses `usage.prompt_tokens` only,
which is the same in every configuration of one model.  So two configurations get byte-identical request sequences;
`requests.jsonl` holds each request's SHA-256 to prove it.

What goes into the history after a request is a scripted assistant message, never the model's reply: the reply is
streamed, timed and dropped.  Probes (criterion 2 of `.ai/RC.md`) are a fixed question at max_tokens 64, greedy; the
probe file keeps the request and the reply's text (content, reasoning and tool-call arguments) and completion_tokens,
and `replay` compares those.  The 0.2.15 server's stream carries no token ids; equal text and an equal token count is
the check.

Output in OUT_DIR: requests.jsonl (one line per request), samples.jsonl (every --sample-s: VRAM per card, available
RAM, swap, resident memory of processes matching --rss-match), probes/probe-*.json, session.json (totals; written at
the end, also after an error).  Exit status 0 only when every request succeeded.
"""
import argparse, hashlib, http.server, json, os, random, re, signal, subprocess, sys, threading, time, urllib.request

stop_after_request = False


def request_stop(signum, frame):
    global stop_after_request
    stop_after_request = True

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
CORPUS_DIRS = ('src', 'include', 'serve', 'tools', 'docs')
CORPUS_EXT = ('.cpp', '.hpp', '.h', '.cu', '.py', '.md')
PROBE_Q = ('Probe: in one short paragraph, name the file the most recent tool output came from and say what the code '
           'in it does.')
TOOL_NAMES = ('read', 'bash', 'edit', 'write', 'grep', 'find', 'ls')


# ----------------------------------------------------------------------------------------------------------------------
# the workload


def add_model_flags(p):
    g = p.add_argument_group('workload model (.ai/RC.md); token counts are converted with --chars-per-token')
    g.add_argument('--seed', type=int, default=85)
    g.add_argument('--scaled', action='store_true', help='context 32768, one compaction (smoke test)')
    g.add_argument('--context', type=int, default=262144)
    g.add_argument('--compact-at', type=float, default=0.92, help='fraction of --context')
    g.add_argument('--request-cap', type=float, default=248000 / 262144, help='fraction of --context a request may reach')
    g.add_argument('--compactions', type=int, default=2, help='the session ends one request after this many')
    g.add_argument('--chars-per-token', type=float, default=3.3)
    g.add_argument('--prefix-tokens', type=int, default=10000, help='system prompt with tool definitions')
    g.add_argument('--tool-tokens', type=int, default=128, help='generated tokens of a tool-call request')
    g.add_argument('--answer-tokens', type=int, default=512, help='generated tokens of every --answer-every request')
    g.add_argument('--answer-every', type=int, default=5)
    g.add_argument('--summary-gen-tokens', type=int, default=2048, help='generated tokens of the compaction request')
    g.add_argument('--summary-tokens', type=int, default=8000, help='scripted summary the next cycle starts from')
    g.add_argument('--keep-turns', type=int, default=2, help='turns kept after a compaction')
    g.add_argument('--out-dist', default='0.50:500-2000,0.35:2000-8000,0.15:8000-20000',
                   help='tool output tokens: share:low-high,...')
    g.add_argument('--prune-every', type=int, default=8)
    g.add_argument('--prune-keep', type=int, default=6, help='most recent turns never pruned')
    g.add_argument('--prune-min', type=int, default=1000, help='only outputs longer than this many tokens are pruned')
    g.add_argument('--stub-tokens', type=int, default=150)
    g.add_argument('--call-tokens', type=int, default=60, help='scripted assistant text before a tool call')
    g.add_argument('--answer-script-tokens', type=int, default=400, help='scripted text of an answer turn')
    g.add_argument('--probe-deep', type=float, default=200000 / 262144, help='probe once prompt_tokens reach this fraction')
    g.add_argument('--probe-tokens', type=int, default=64)
    g.add_argument('--reasoning-budget', type=int, default=0,
                   help='thinking token limit; 0 disables thinking via the chat template')
    g.add_argument('--max-requests', type=int, default=0, help='stop after this many requests (0: no limit)')
    g.add_argument('--max-cycle-min', type=float, default=90, help='stop after a cycle exceeding this many minutes (0: no limit)')


def finish_flags(a):
    if a.scaled:
        a.context, a.compactions = 32768, min(a.compactions, 1)
        # a 32K context cannot hold 20K outputs or an 8K summary: scale the token sizes with the context
        f = 32768 / 262144
        a.prefix_tokens = max(1000, int(a.prefix_tokens * f * 2))
        a.summary_tokens = max(500, int(a.summary_tokens * f))
        a.summary_gen_tokens = min(a.summary_gen_tokens, 512)
        a.out_dist = ','.join(f'{s}:{max(100, int(int(lo) * f * 2))}-{max(200, int(int(hi) * f * 2))}'
                              for s, lo, hi in (re.match(r'([\d.]+):(\d+)-(\d+)', x).groups()
                                                for x in a.out_dist.split(',')))
    a.dist = [(float(s), int(lo), int(hi)) for s, lo, hi in
              (re.match(r'([\d.]+):(\d+)-(\d+)$', x.strip()).groups() for x in a.out_dist.split(','))]
    return a


class Corpus:
    """Real text from this checkout: tool outputs look like the files and searches an agent reads."""

    def __init__(self):
        files = []
        for d in CORPUS_DIRS:
            for root, dirs, names in os.walk(os.path.join(REPO, d)):
                dirs.sort()
                for n in sorted(names):
                    if n.endswith(CORPUS_EXT):
                        files.append(os.path.join(root, n))
        self.files = [f for f in files if 1000 < os.path.getsize(f) < 4_000_000]
        if not self.files:
            raise SystemExit('agent_session: no corpus files under ' + REPO)
        self.cache = {}

    def text(self, i):
        f = self.files[i % len(self.files)]
        if f not in self.cache:
            with open(f, encoding='utf-8', errors='replace') as h:
                self.cache[f] = h.read()
        return os.path.relpath(f, REPO), self.cache[f]

    def take(self, rng, chars):
        """`chars` characters from consecutive files, starting at a seeded file and line."""
        i = rng.randrange(len(self.files))
        path, t = self.text(i)
        start = t.rfind('\n', 0, rng.randrange(max(1, len(t) - min(len(t) - 1, chars)))) + 1
        out, first = [t[start:start + chars]], path
        while sum(map(len, out)) < chars:
            i += 1
            path, t = self.text(i)
            out.append(f'\n--- {path}\n' + t[:chars - sum(map(len, out))])
        return first, start, ''.join(out)[:chars]


def filler(corpus, rng, chars, head):
    return head + corpus.take(rng, max(0, chars - len(head)))[2]


class Session:
    def __init__(self, a, send):
        self.a, self.send = a, send
        self.rng = random.Random(a.seed)
        self.corpus = Corpus()
        self.cpt = a.chars_per_token
        self.ch = lambda tokens: int(tokens * self.cpt)
        self.trigger = int(a.compact_at * a.context)
        self.cap = int(a.request_cap * a.context)
        self.deep = int(a.probe_deep * a.context)
        prng = random.Random(a.seed + 1)            # the fixed prefix is the same in every cycle
        self.tools = [{'type': 'function', 'function': {
            'name': n, 'description': filler(self.corpus, prng, self.ch(a.prefix_tokens // 20),
                                             f'The {n} tool. Reference notes follow.\n'),
            'parameters': {'type': 'object', 'properties': {
                'path': {'type': 'string', 'description': 'A path in the repository.'},
                'command': {'type': 'string', 'description': 'A command or a pattern.'}}, 'required': []}}}
            for n in TOOL_NAMES]
        tools_chars = len(json.dumps(self.tools))
        self.system = {'role': 'system', 'content': filler(
            self.corpus, prng, max(2000, self.ch(a.prefix_tokens) - tools_chars),
            'You are a coding agent working in a C++/CUDA/HIP inference engine repository. Use the tools. Project '
            'guidelines follow.\n')}
        self.n_req = self.n_call = 0
        self.turn = 0                    # turns of the current cycle
        self.tool_msgs = []              # (turn, index in messages) of the current cycle's tool outputs
        self.probed = set()
        self.after_prune = self.after_compaction = False
        self.totals = dict(requests=0, prompt_tokens=0, cached_tokens=0, completion_tokens=0, prunes=0,
                           pruned_outputs=0, compactions=0, cut_outputs=0, errors=0, probes=0)

    # -- one request
    def request(self, messages, max_tokens, kind):
        body = dict(model='strata', messages=messages, tools=self.tools, temperature=0, seed=0, max_tokens=max_tokens,
                    reasoning_budget_tokens=self.a.reasoning_budget, stream=True,
                    stream_options={'include_usage': True})
        if self.a.reasoning_budget == 0:
            # A zero budget means *unlimited* in the server, not thinking off.
            body['chat_template_kwargs'] = {'enable_thinking': False}
        raw = json.dumps(body, ensure_ascii=False, sort_keys=True).encode()
        self.n_req += 1
        r = self.send(raw)
        r.update(n=self.n_req, kind=kind, cycle=self.totals['compactions'], turn=self.turn, max_tokens=max_tokens,
                 sha256=hashlib.sha256(raw).hexdigest(), request_bytes=len(raw), messages=len(messages))
        u = r.get('usage') or {}
        for k in ('prompt_tokens', 'completion_tokens'):
            self.totals[k] += u.get(k) or 0
        self.totals['cached_tokens'] += (u.get('prompt_tokens_details') or {}).get('cached_tokens') or 0
        self.totals['requests'] += 1
        if r.get('error'):
            self.totals['errors'] += 1
        return r, raw

    # -- the history
    def call_msg(self, rng, text_tokens, name=None):
        self.n_call += 1
        name = name or TOOL_NAMES[rng.randrange(len(TOOL_NAMES))]
        path = self.corpus.take(rng, 10)[0]
        m = {'role': 'assistant', 'content': filler(self.corpus, rng, self.ch(text_tokens), 'Next I check this. '),
             'tool_calls': [{'id': f'call_{self.n_call}', 'type': 'function',
                             'function': {'name': name, 'arguments': json.dumps({'path': path})}}]}
        return m, name

    def tool_out(self, rng, est_tokens):
        u = rng.random()
        acc = 0.0
        for share, lo, hi in self.a.dist:
            acc += share
            if u <= acc:
                break
        tokens = rng.randint(lo, hi)
        room = self.cap - est_tokens - self.a.answer_tokens - 200
        if tokens > room:
            tokens = max(50, room)
            self.totals['cut_outputs'] += 1
        path, start, text = self.corpus.take(rng, self.ch(tokens))
        return f'$ read {path} (from character {start})\n' + text

    def prune(self, messages):
        """The extension: outputs older than the --prune-keep most recent turns and longer than --prune-min become
        stubs.  Pruned outputs stay as they are."""
        n = 0
        for t, i in self.tool_msgs:
            m = messages[i]
            if t < self.turn - self.a.prune_keep and not m.get('_pruned') and len(m['content']) > self.ch(self.a.prune_min):
                note = '\n[output pruned by the context extension]'
                m['content'] = m['content'][:max(0, self.ch(self.a.stub_tokens) - len(note))] + note
                m['_pruned'] = True
                n += 1
        self.totals['prunes'] += 1
        self.totals['pruned_outputs'] += n
        return n

    def probe(self, messages, why, out_dir, log):
        if why in self.probed:
            return
        self.probed.add(why)
        pm = messages + [{'role': 'user', 'content': PROBE_Q}]
        r, raw = self.request(clean(pm), self.a.probe_tokens, 'probe-' + why)
        log(r)
        self.totals['probes'] += 1
        os.makedirs(os.path.join(out_dir, 'probes'), exist_ok=True)
        with open(os.path.join(out_dir, 'probes', f'probe-{why}.json'), 'w') as h:
            json.dump({'why': why, 'request': json.loads(raw), 'reply': reply_of(r)}, h, ensure_ascii=False)

    def run(self, out_dir, log):
        a, rng = self.a, self.rng
        messages = [self.system, {'role': 'user', 'content': filler(
            self.corpus, rng, self.ch(300), 'Task: find why the second stage is slow on long prompts and fix it. '
            'Context from the issue follows.\n')}]
        end_after = None
        cycle_started = time.monotonic()
        while True:
            if stop_after_request:
                self.totals['stopped_after_request'] = True
                return
            answer = self.turn % a.answer_every == a.answer_every - 1
            r, _ = self.request(clean(messages), a.answer_tokens if answer else a.tool_tokens,
                                'answer' if answer else 'tool')
            log(r)
            if stop_after_request:
                self.totals['stopped_after_request'] = True
                return
            if r.get('error') or (a.max_requests and self.n_req >= a.max_requests):
                return
            pt = (r.get('usage') or {}).get('prompt_tokens') or 0
            if self.after_prune and self.totals['compactions'] == 0 and self.turn >= 2 * a.prune_every:
                self.probe(messages, 'pruned', out_dir, log)
            if stop_after_request:
                self.totals['stopped_after_request'] = True
                return
            if self.after_compaction:
                self.probe(messages, 'compacted', out_dir, log)
            if stop_after_request:
                self.totals['stopped_after_request'] = True
                return
            if pt >= self.deep:
                self.probe(messages, 'deep', out_dir, log)
            if stop_after_request:
                self.totals['stopped_after_request'] = True
                return
            self.after_prune = self.after_compaction = False
            if end_after is not None:
                return
            if pt >= self.trigger:
                messages = self.compact(messages, log)
                if messages is None:
                    return
                if a.max_cycle_min and time.monotonic() - cycle_started > a.max_cycle_min * 60:
                    self.totals['cycle_limit_stop'] = True
                    return
                cycle_started = time.monotonic()
                if self.totals['compactions'] >= a.compactions:
                    end_after = True
                continue
            est = pt
            if answer:
                messages.append({'role': 'assistant', 'content': filler(self.corpus, rng, self.ch(a.answer_script_tokens),
                                                                        'Here is what I found so far. ')})
                messages.append({'role': 'user', 'content': 'Good. Continue with the next file.'})
                est += a.answer_script_tokens + 20
            m, _ = self.call_msg(rng, a.call_tokens)
            messages.append(m)
            out = self.tool_out(rng, est + a.call_tokens + 40)
            messages.append({'role': 'tool', 'tool_call_id': m['tool_calls'][0]['id'], 'content': out})
            self.tool_msgs.append((self.turn, len(messages) - 1))
            self.turn += 1
            if self.turn % a.prune_every == 0 and self.prune(messages):
                self.after_prune = True

    def compact(self, messages, log):
        a, rng = self.a, self.rng
        r, _ = self.request(clean(messages + [{'role': 'user', 'content':
                            'Summarise this session so far for a fresh context: the task, the files read, the '
                            'findings and the next steps.'}]), a.summary_gen_tokens, 'compaction')
        log(r)
        if r.get('error'):
            return None
        self.totals['compactions'] += 1
        # the last --keep-turns turns: from the assistant message of the oldest kept tool output to the end
        kept = []
        if self.tool_msgs:
            first = self.tool_msgs[-min(a.keep_turns, len(self.tool_msgs))][1] - 1
            kept = [dict(m) for m in messages[first:]]
        new = [self.system, {'role': 'user', 'content': filler(
            self.corpus, rng, self.ch(a.summary_tokens),
            'Summary of the session so far (compacted). Continue the task from here.\n')}] + kept
        self.tool_msgs = [(0, i) for i, m in enumerate(new) if m.get('role') == 'tool']
        self.turn = 0
        self.after_compaction = True
        return new


def clean(messages):
    return [{k: v for k, v in m.items() if not k.startswith('_')} for m in messages]


def reply_of(r):
    return {k: r.get(k) for k in ('content', 'reasoning', 'tool_args', 'finish')} | \
        {'completion_tokens': (r.get('usage') or {}).get('completion_tokens')}


# ----------------------------------------------------------------------------------------------------------------------
# the client


def make_sender(url, api_key=None, timeout=3600):
    def send(raw):
        h = {'Content-Type': 'application/json'}
        if api_key:
            h['Authorization'] = 'Bearer ' + api_key
        req = urllib.request.Request(url.rstrip('/') + '/v1/chat/completions', data=raw, headers=h)
        t0 = time.monotonic()
        res = dict(t_start=time.time(), content='', reasoning='', tool_args='', finish=None, usage=None, timings=None,
                   lp_content=[], lp_reasoning=[])
        first = last = None
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
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
                        for k, field in (('content', 'lp_content'), ('reasoning_content', 'lp_reasoning')):
                            res[field] += ((c.get('logprobs') or {}).get(k)) or []
                        got = False
                        for k, field in (('content', 'content'), ('reasoning_content', 'reasoning')):
                            if d.get(k):
                                res[field] += d[k]
                                got = True
                        for tc in d.get('tool_calls') or []:
                            res['tool_args'] += (tc.get('function') or {}).get('name') or ''
                            res['tool_args'] += (tc.get('function') or {}).get('arguments') or ''
                            got = True
                        if got:
                            now = time.monotonic()
                            first = now if first is None else first
                            last = now
                        res['finish'] = c.get('finish_reason') or res['finish']
        except Exception as e:  # noqa: BLE001 - every failure is recorded and ends the session
            res['error'] = f'{type(e).__name__}: {e}'
        end = time.monotonic()
        res.update(wall_s=round(end - t0, 3), ttft_s=round(first - t0, 3) if first else None,
                   client_decode_s=round(last - first, 3) if first and last else None)
        return res
    return send


def row(r):
    """The requests.jsonl line: what the summary needs, not the text."""
    u, t = r.get('usage') or {}, r.get('timings') or {}
    return dict(n=r['n'], kind=r['kind'], cycle=r['cycle'], turn=r['turn'], sha256=r['sha256'],
                request_bytes=r['request_bytes'], messages=r['messages'], max_tokens=r['max_tokens'],
                prompt_tokens=u.get('prompt_tokens'), cached_tokens=(u.get('prompt_tokens_details') or {}).get('cached_tokens'),
                completion_tokens=u.get('completion_tokens'), prompt_n=t.get('prompt_n'), cache_n=t.get('cache_n'),
                prompt_ms=t.get('prompt_ms'), prompt_tok_s=t.get('prompt_per_second'), predicted_ms=t.get('predicted_ms'),
                decode_tok_s=t.get('predicted_per_second'), draft_n=t.get('draft_n'), draft_accepted=t.get('draft_n_accepted'),
                ttft_s=r.get('ttft_s'), wall_s=r.get('wall_s'), finish=r.get('finish'), error=r.get('error'),
                t_start=r.get('t_start'))


# ----------------------------------------------------------------------------------------------------------------------
# the sampler


def sample(match):
    s = dict(t=time.time())
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=20).stdout
        s['cuda_vram_mib'] = {l.split(',')[0].strip(): int(l.split(',')[1]) for l in out.strip().splitlines() if ',' in l}
    except Exception:  # noqa: BLE001
        pass
    try:
        out = subprocess.run(['rocm-smi', '--showmeminfo', 'vram', '--json'], capture_output=True, text=True,
                             timeout=20).stdout
        j = json.loads(out)
        s['hip_vram_mib'] = {k: int(v.get('VRAM Total Used Memory (B)', 0)) >> 20 for k, v in j.items()
                             if isinstance(v, dict)}
    except Exception:  # noqa: BLE001
        try:  # ROCm 10 on Oberon supplies amd-smi instead of rocm-smi.
            out = subprocess.run(['amd-smi', 'metric', '-m', '--json'], capture_output=True, text=True,
                                 timeout=20, check=True).stdout
            j = json.loads(out)
            s['hip_vram_mib'] = {str(g['gpu']): g['mem_usage']['used_vram']['value'] for g in j['gpu_data']}
        except Exception:  # noqa: BLE001
            pass
    try:
        mi = dict(l.split(':', 1) for l in open('/proc/meminfo'))
        kb = lambda k: int(mi[k].split()[0])
        s['mem_available_mib'] = kb('MemAvailable') >> 10
        s['swap_used_mib'] = (kb('SwapTotal') - kb('SwapFree')) >> 10
    except Exception:  # noqa: BLE001
        pass
    if match:
        rss = {}
        for pid in os.listdir('/proc'):
            if not pid.isdigit():
                continue
            try:
                cmd = open(f'/proc/{pid}/cmdline', 'rb').read().replace(b'\0', b' ').decode(errors='replace')
                if re.search(match, cmd) and 'agent_session' not in cmd:
                    st = dict(l.split(':', 1) for l in open(f'/proc/{pid}/status'))
                    rss[pid] = {'rss_mib': int(st['VmRSS'].split()[0]) >> 10, 'cmd': cmd[:120]}
            except Exception:  # noqa: BLE001
                pass
        s['rss'] = rss
    return s


def sampler(path, every, match, stop):
    with open(path, 'a') as h:
        while not stop.is_set():
            h.write(json.dumps(sample(match)) + '\n')
            h.flush()
            stop.wait(every)


# ----------------------------------------------------------------------------------------------------------------------
# commands


def cmd_run(a, send=None, quiet=False):
    finish_flags(a)
    os.makedirs(a.out, exist_ok=True)
    send = send or make_sender(a.url, os.environ.get(a.api_key_env) if a.api_key_env else None)
    stop = threading.Event()
    th = None
    if a.sample_s > 0:
        th = threading.Thread(target=sampler, args=(os.path.join(a.out, 'samples.jsonl'), a.sample_s, a.rss_match, stop),
                              daemon=True)
        th.start()
    global stop_after_request
    stop_after_request = False
    previous_signal = signal.signal(signal.SIGUSR1, request_stop) if hasattr(signal, 'SIGUSR1') else None
    s = Session(a, send)
    t0 = time.time()
    with open(os.path.join(a.out, 'requests.jsonl'), 'w') as h:
        def log(r):
            line = row(r)
            h.write(json.dumps(line) + '\n')
            h.flush()
            if not quiet:
                print(f"#{line['n']:4d} c{line['cycle']} t{line['turn']:3d} {line['kind']:14s} prompt {line['prompt_tokens']} "
                      f"cached {line['cached_tokens']} gen {line['completion_tokens']} ttft {line['ttft_s']} "
                      f"pp {line['prompt_tok_s']} dec {line['decode_tok_s']}" + (f" ERROR {line['error']}" if line['error'] else ''),
                      flush=True)
        try:
            s.run(a.out, log)
        finally:
            stop.set()
            if previous_signal is not None:
                signal.signal(signal.SIGUSR1, previous_signal)
            if th:
                th.join(timeout=30)
            flags = {k: v for k, v in vars(a).items() if k not in ('func', 'dist')}
            with open(os.path.join(a.out, 'session.json'), 'w') as f:
                json.dump(dict(totals=s.totals, wall_s=round(time.time() - t0, 1), flags=flags), f, indent=1)
    return 0 if s.totals['errors'] == 0 and not s.totals.get('stopped_after_request') else 1


NEAR_TIE_NATS = 1.0    # a live token this close to the fresh server's choice (log-probability) is a near-tie


def near_tie(live: str, fresh: str, entries: list, max_gap: float = NEAR_TIE_NATS) -> dict:
    """Where `live` first leaves `fresh` (one field of the reply), is the live reply's next token one the fresh server
    nearly chose?  `entries`: the fresh server's logprobs entries for that field (OpenAI form, each with
    top_logprobs).  Only the first difference is judged: after it the two replies are conditioned on different text."""
    if live == fresh:
        return {'verdict': 'equal'}
    at = next((i for i, (x, y) in enumerate(zip(live, fresh)) if x != y), min(len(live), len(fresh)))
    spelled = ''.join(e.get('token') or '' for e in entries)
    if not entries or not fresh.startswith(spelled[:at + 1]) or len(spelled) <= at and len(fresh) > at:
        return {'char': at, 'verdict': 'unjudged',
                'why': 'the fresh logprobs do not spell the fresh text up to the difference (missing or incomplete entries)'}
    pos = 0
    for k, e in enumerate(entries):
        tok = e.get('token') or ''
        if pos + len(tok) > at or k == len(entries) - 1:
            rest = live[pos:]
            alts = [x for x in e.get('top_logprobs') or [] if x.get('token') and x['token'] != tok]
            hit = max((x for x in alts if rest.startswith(x['token'])), key=lambda x: x['logprob'], default=None)
            base = dict(char=at, token_index=k, fresh_token=tok, fresh_logprob=e.get('logprob'))
            if hit is None:
                return {**base, 'verdict': 'DIFFERENT', 'why': 'the live token is not in the fresh top logprobs'}
            gap = float(e.get('logprob', 0.0)) - float(hit['logprob'])
            return {**base, 'live_token': hit['token'], 'live_logprob': hit['logprob'], 'gap_nats': round(gap, 3),
                    'verdict': 'near-tie' if gap <= max_gap else 'DIFFERENT'}
        pos += len(tok)
    return {'char': at, 'verdict': 'unjudged', 'why': 'no logprobs from the fresh server'}


def judge(live: dict, fresh: dict, max_gap: float = NEAR_TIE_NATS) -> dict:
    """The probe verdict: equal, near-tie (the first difference is a near-tie in the field that differs first: reasoning,
    then content), or DIFFERENT.  Tool-call arguments come last and are compared only when the texts are equal."""
    for field in ('reasoning', 'content'):
        a, b = live.get(field) or '', fresh.get(field) or ''
        if a != b:
            return {'field': field, **near_tie(a, b, fresh.get('lp_' + field) or [], max_gap)}
    if (live.get('tool_args') or '') != (fresh.get('tool_args') or ''):
        return {'field': 'tool_args', 'verdict': 'DIFFERENT', 'why': 'equal text, different tool-call arguments'}
    if live.get('completion_tokens') != fresh.get('completion_tokens'):
        return {'field': 'completion_tokens', 'verdict': 'DIFFERENT', 'why': 'equal text, different token counts'}
    return {'verdict': 'equal'}


def cmd_replay(a):
    """One saved probe on a (freshly started) server, with top logprobs, judged against the live reply (judge).  Exit 0
    for equal or near-tie.  A non-finite-logits line in either server's log is a failure whatever this says: check the
    logs (.ai/RC.md, criterion 2)."""
    p = json.load(open(a.probe))
    req = dict(p['request'], logprobs=True, top_logprobs=5)
    r = make_sender(a.url, os.environ.get(a.api_key_env) if a.api_key_env else None)(
        json.dumps(req, ensure_ascii=False, sort_keys=True).encode())
    got, want = reply_of(r), p['reply']
    got_lp = {**got, 'lp_reasoning': r.get('lp_reasoning') or [], 'lp_content': r.get('lp_content') or []}
    v = {'verdict': 'DIFFERENT', 'why': f"replay error: {r['error']}"} if r.get('error') else judge(want, got_lp, a.max_gap)
    same = v['verdict'] == 'equal'
    res = dict(probe=a.probe, why=p['why'], equal=same, verdict=v, error=r.get('error'), live=want, fresh=got,
               fresh_logprobs=bool(got_lp['lp_reasoning'] or got_lp['lp_content']),
               prompt_tokens=(r.get('usage') or {}).get('prompt_tokens'), wall_s=r.get('wall_s'))
    out = a.probe[:-5] + '-replay.json'
    json.dump(res, open(out, 'w'), indent=1, ensure_ascii=False)
    extra = f" (gap {v['gap_nats']} nats at token {v['token_index']})" if 'gap_nats' in v else (f" ({v['why']})" if 'why' in v else '')
    print(f"{v['verdict'].upper() if v['verdict'] == 'equal' else v['verdict']} {p['why']}{extra}: {out}")
    return 0 if v['verdict'] in ('equal', 'near-tie') else 1


class StubHandler(http.server.BaseHTTPRequestHandler):
    """A fake server: 3.3 characters per token, the longest common prefix with the previous request as the cache,
    a deterministic reply of min(max_tokens, 24) tokens.  For the driver's bookkeeping only."""
    prev = ''
    lock = threading.Lock()

    def log_message(self, *x):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        text = json.dumps(body['messages'], sort_keys=True) + json.dumps(body.get('tools'), sort_keys=True)
        with StubHandler.lock:
            common = os.path.commonprefix([text, StubHandler.prev])
            StubHandler.prev = text
        pt, cached = int(len(text) / 3.3), int(len(common) / 3.3)
        n = min(body.get('max_tokens') or 16, 24)
        words = hashlib.sha256(text.encode()).hexdigest()
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()
        for i in range(n):
            ev = {'choices': [{'index': 0, 'delta': {'content': words[i % 64]}, 'finish_reason': None}]}
            self.wfile.write(b'data: ' + json.dumps(ev).encode() + b'\n\n')
        last = {'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'length'}],
                'usage': {'prompt_tokens': pt, 'completion_tokens': n, 'total_tokens': pt + n,
                          'prompt_tokens_details': {'cached_tokens': cached}},
                'timings': {'cache_n': cached, 'prompt_n': pt - cached, 'prompt_ms': 1.0, 'prompt_per_second': 1000.0,
                            'predicted_n': n, 'predicted_ms': 1.0, 'predicted_per_second': 100.0}}
        self.wfile.write(b'data: ' + json.dumps(last).encode() + b'\n\ndata: [DONE]\n\n')


def stub_server(port):
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', port), StubHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def cmd_selftest(a):
    import tempfile
    srv = stub_server(0)
    url = f'http://127.0.0.1:{srv.server_address[1]}'
    p = argparse.ArgumentParser()
    add_run_flags(p)
    tmp = tempfile.mkdtemp(prefix='agent_session_selftest_')
    shas, totals = [], []
    for k in range(2):
        StubHandler.prev = ''
        ra = p.parse_args(['run', os.path.join(tmp, f's{k}'), '--url', url, '--scaled', '--sample-s', '0'])
        assert cmd_run(ra, quiet=True) == 0, 'a stub session failed'
        rows = [json.loads(l) for l in open(os.path.join(tmp, f's{k}', 'requests.jsonl'))]
        shas.append([r['sha256'] for r in rows])
        totals.append(json.load(open(os.path.join(tmp, f's{k}', 'session.json')))['totals'])
    rows = [json.loads(l) for l in open(os.path.join(tmp, 's0', 'requests.jsonl'))]
    t = totals[0]
    fa = finish_flags(p.parse_args(['run', 'x', '--scaled']))
    checks = {
        'identical request sequences for one seed': shas[0] == shas[1],
        'at least one pruning step removed outputs': t['pruned_outputs'] > 0,
        f'{fa.compactions} compaction(s)': t['compactions'] == fa.compactions,
        'ends one request after the last compaction': [r['kind'] for r in rows if not r['kind'].startswith('probe')][-2:]
                                                      in (['compaction', 'tool'], ['compaction', 'answer']),
        'no request over the cap': max(r['prompt_tokens'] for r in rows) <= int(fa.request_cap * fa.context) + 64,
        'a compaction only at or above the trigger': all(rows[i - 1]['prompt_tokens'] >= int(fa.compact_at * fa.context)
                                                       for i, r in enumerate(rows) if r['kind'] == 'compaction'),
        'probes pruned, compacted and deep saved': all(os.path.exists(os.path.join(tmp, 's0', 'probes', f'probe-{w}.json'))
                                                       for w in ('pruned', 'compacted', 'deep')),
        'every turn after a pruning step reuses less than the one before it at least once':
            any(r['cached_tokens'] < r['prompt_tokens'] * 0.9 for r in rows if r['kind'] in ('tool', 'answer')),
    }
    for budget in (0, 20):
        captured = []
        args = finish_flags(p.parse_args(['run', 'x', '--scaled', '--reasoning-budget', str(budget)]))
        session = Session(args, lambda raw: captured.append(json.loads(raw)) or {})
        session.request([{'role': 'user', 'content': 'hello'}], 16, 'tool')
        body = captured[0]
        checks[f'reasoning budget {budget} request'] = (body['reasoning_budget_tokens'] == budget and
            (body.get('chat_template_kwargs') == {'enable_thinking': False} if budget == 0
             else 'chat_template_kwargs' not in body))
    for w in ('pruned', 'compacted', 'deep'):
        if not os.path.exists(os.path.join(tmp, 's0', 'probes', f'probe-{w}.json')):
            continue
        StubHandler.prev = ''
        ra = argparse.Namespace(probe=os.path.join(tmp, 's0', 'probes', f'probe-{w}.json'), url=url, api_key_env=None, max_gap=NEAR_TIE_NATS)
        checks[f'replay {w} equal on the stub'] = cmd_replay(ra) == 0
    # the near-tie judgement on made-up logprobs: the fresh reply says "The user asks", the live one "The user's"
    ent = [{'token': 'The', 'logprob': -0.1, 'top_logprobs': []}, {'token': ' user', 'logprob': -0.2, 'top_logprobs': []},
           {'token': ' asks', 'logprob': -0.6, 'top_logprobs': [{'token': ' asks', 'logprob': -0.6},
                                                               {'token': "'s", 'logprob': -0.9}]}]
    checks['near-tie: a close second choice'] = judge({'reasoning': "The user's query"},
                                                      {'reasoning': 'The user asks', 'lp_reasoning': ent})['verdict'] == 'near-tie'
    checks['DIFFERENT: a far second choice'] = judge({'reasoning': "The user's query"},
                                                     {'reasoning': 'The user asks', 'lp_reasoning': ent}, 0.2)['verdict'] == 'DIFFERENT'
    checks['DIFFERENT: not among the alternatives'] = judge({'reasoning': 'The user wants'},
                                                            {'reasoning': 'The user asks', 'lp_reasoning': ent})['verdict'] == 'DIFFERENT'
    checks['equal texts, different tool arguments'] = judge({'reasoning': 'a', 'tool_args': 'x'},
                                                            {'reasoning': 'a', 'tool_args': 'y'})['verdict'] == 'DIFFERENT'
    checks['unjudged: an entry missing before the difference'] = judge(
        {'reasoning': "The user's query"}, {'reasoning': 'The user asks', 'lp_reasoning': [ent[0], ent[2]]})['verdict'] == 'unjudged'
    for k, v in checks.items():
        print(('ok    ' if v else 'FAIL  ') + k)
    print(f"requests {t['requests']}, prunes {t['prunes']} ({t['pruned_outputs']} outputs), compactions {t['compactions']}, "
          f"cut outputs {t['cut_outputs']}, max prompt {max(r['prompt_tokens'] for r in rows)}; files in {tmp}")
    srv.shutdown()
    return 0 if all(checks.values()) else 1


def add_run_flags(p):
    p.add_argument('cmd')
    p.add_argument('out')
    p.add_argument('--url', default='http://127.0.0.1:8096')
    p.add_argument('--api-key-env', default=None, help='the name of an environment variable holding the API key')
    p.add_argument('--sample-s', type=float, default=30)
    p.add_argument('--rss-match', default=r'strata|server\.py|front', help='regex for processes whose RSS is sampled')
    add_model_flags(p)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'run':
        p = argparse.ArgumentParser()
        add_run_flags(p)
        return cmd_run(p.parse_args())
    if cmd == 'replay':
        p = argparse.ArgumentParser()
        p.add_argument('cmd')
        p.add_argument('probe')
        p.add_argument('--url', default='http://127.0.0.1:8096')
        p.add_argument('--api-key-env', default=None)
        p.add_argument('--max-gap', type=float, default=NEAR_TIE_NATS,
                       help='nats: a first difference this close to the fresh choice is a near-tie')
        return cmd_replay(p.parse_args())
    if cmd == 'stub':
        p = argparse.ArgumentParser()
        p.add_argument('cmd')
        p.add_argument('--port', type=int, default=8099)
        srv = stub_server(p.parse_args().port)
        print(f'stub on {srv.server_address}', flush=True)
        threading.Event().wait()
    if cmd == 'selftest':
        return cmd_selftest(None)
    print(__doc__)
    return 2


if __name__ == '__main__':
    sys.exit(main())
