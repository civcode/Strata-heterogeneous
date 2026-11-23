#!/usr/bin/env python3
"""Record a real agent session: a pass-through proxy between the agent (pi) and Strata that saves every request.

    python3 mixed_split/tools/record_proxy.py [--listen 8091] [--upstream http://127.0.0.1:8081]
        [--out ~/workspace/strata-local-archive/recordings/]

Point the agent at the proxy (pi: "baseUrl": "http://127.0.0.1:8091/v1") for one session, then stop the proxy with
Ctrl-C and point the agent back at Strata.  Every method, path, header and body goes through unchanged, and a streamed
reply reaches the agent as it arrives; the agent sees no difference except the port.  When the agent stops a reply
(cancel), the proxy closes the upstream connection too, so Strata sees the cancel.

The recording: `<out>/session-<UTC time>.jsonl`, one JSON line per request, written as soon as the request ends:
  seq, t_start / t_first_byte / t_end (epoch seconds), method, path, status, request (the body exactly as sent, as a
  string), and from the reply: usage, timings (Strata's), finish_reason, the number of stream chunks per kind
  (reasoning, content, tool calls) and their text, `client_cancelled`.  The first line is a header (upstream, start).
The `Authorization` header is never written.  The recording holds the session's prompts, tool output and code: keep
it outside the repository (the default folder is), and put only numbers into reports.
"""
import argparse, datetime, http.client, http.server, json, os, sys, threading, time, urllib.parse
from pathlib import Path

HOP = {'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization', 'te', 'trailers', 'transfer-encoding',
       'upgrade', 'host', 'content-length', 'accept-encoding'}


class Recorder:
    def __init__(self, path: Path, upstream: str):
        self.path, self.lock, self.seq = path, threading.Lock(), 0
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as h:
            h.write(json.dumps({'header': True, 'upstream': upstream, 'started': time.time(),
                                'tool': 'mixed_split/tools/record_proxy.py'}) + '\n')

    def next_seq(self):
        with self.lock:
            self.seq += 1
            return self.seq

    def write(self, rec: dict):
        with self.lock, open(self.path, 'a') as h:
            h.write(json.dumps(rec, ensure_ascii=False) + '\n')


class StreamTap:
    """Reads the SSE lines of a streamed chat completion as they pass: usage, timings, chunk counts and text per kind."""

    def __init__(self):
        self.buf = b''
        self.out = {'usage': None, 'timings': None, 'finish_reason': None,
                    'chunks': {'reasoning': 0, 'content': 0, 'tool_calls': 0},
                    'text': {'reasoning': '', 'content': '', 'tool_calls': ''}}

    def feed(self, data: bytes):
        self.buf += data
        while b'\n' in self.buf:
            line, self.buf = self.buf.split(b'\n', 1)
            line = line.strip()
            if not line.startswith(b'data:') or line == b'data: [DONE]':
                continue
            try:
                ev = json.loads(line[5:])
            except ValueError:
                continue
            self.event(ev)

    def event(self, ev: dict):
        o = self.out
        o['usage'] = ev.get('usage') or o['usage']
        o['timings'] = ev.get('timings') or o['timings']
        for c in ev.get('choices') or []:
            d = c.get('delta') or c.get('message') or {}
            if d.get('reasoning_content') or d.get('reasoning'):
                o['chunks']['reasoning'] += 1
                o['text']['reasoning'] += d.get('reasoning_content') or d.get('reasoning') or ''
            if d.get('content'):
                o['chunks']['content'] += 1
                o['text']['content'] += d['content']
            for tc in d.get('tool_calls') or []:
                o['chunks']['tool_calls'] += 1
                f = tc.get('function') or {}
                o['text']['tool_calls'] += (f.get('name') or '') + (f.get('arguments') or '')
            o['finish_reason'] = c.get('finish_reason') or o['finish_reason']


def make_handler(upstream: str, rec: Recorder):
    up = urllib.parse.urlsplit(upstream)

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.0'          # one request per connection; a stream ends when the connection closes

        def log_message(self, *a):
            pass

        def do_GET(self):
            self.forward()

        def do_POST(self):
            self.forward()

        def do_DELETE(self):
            self.forward()

        def forward(self):
            seq, t0 = rec.next_seq(), time.time()
            n = int(self.headers.get('Content-Length') or 0)
            body = self.rfile.read(n) if n else b''
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
            r = {'seq': seq, 't_start': t0, 'method': self.command, 'path': self.path,
                 'request': body.decode('utf-8', errors='replace') if body else '', 'client_cancelled': False}
            tap, raw = StreamTap(), b''
            conn = http.client.HTTPConnection(up.hostname, up.port or 80, timeout=7200)
            try:
                conn.request(self.command, self.path, body=body or None, headers=headers)
                resp = conn.getresponse()
                r['status'] = resp.status
                self.send_response(resp.status)
                streaming = 'text/event-stream' in (resp.getheader('Content-Type') or '')
                for k, v in resp.getheaders():
                    if k.lower() not in HOP:
                        self.send_header(k, v)
                if not streaming and resp.getheader('Content-Length'):
                    self.send_header('Content-Length', resp.getheader('Content-Length'))
                self.end_headers()
                while True:
                    chunk = resp.read1(65536) if streaming else resp.read(65536)
                    if not chunk:
                        break
                    r.setdefault('t_first_byte', time.time())
                    if streaming:
                        tap.feed(chunk)
                    elif len(raw) < 64 << 20:
                        raw += chunk
                    try:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        r['client_cancelled'] = True      # the agent stopped the reply: close upstream, Strata cancels
                        break
                if not streaming and raw:
                    try:
                        tap.event(json.loads(raw))
                    except ValueError:
                        pass
            except Exception as e:  # noqa: BLE001 - recorded, and the agent gets the error
                r['error'] = f'{type(e).__name__}: {e}'
                try:
                    self.send_error(502, 'record_proxy: upstream failed')
                except Exception:  # noqa: BLE001
                    pass
            finally:
                conn.close()
                r['t_end'] = time.time()
                r.update(tap.out)
                rec.write(r)

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--listen', type=int, default=8091)
    ap.add_argument('--upstream', default='http://127.0.0.1:8081')
    ap.add_argument('--out', default='~/workspace/strata-local-archive/recordings/')
    a = ap.parse_args()
    out = Path(os.path.expanduser(a.out))
    path = out / f"session-{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    rec = Recorder(path, a.upstream)
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', a.listen), make_handler(a.upstream, rec))
    print(f'record_proxy: 127.0.0.1:{a.listen} -> {a.upstream}; recording to {path}  (Ctrl-C to stop)', flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    srv.server_close()
    print(f'record_proxy: {rec.seq} requests recorded in {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
