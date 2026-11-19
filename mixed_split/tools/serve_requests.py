#!/usr/bin/env python3
"""Start serve/server.py with a run config, send a list of chat requests one after another, stop it with one SIGINT.

    .venv/bin/python mixed_split/tools/serve_requests.py --config strata-q2_0.json --port 8081 \
        --request short:200 --request short:200 --request long:50 [--stream-first]

A request is PROMPT:MAX_TOKENS; PROMPT is `short`, `long` (about 5000 tokens) or `mid` (about 2500).  Prints one JSON
line per request (usage and timings as the server reports them) and the wall time the server process was alive.
Standard library only.  The server is never killed: SIGINT, then wait.
"""
import argparse, json, os, signal, subprocess, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mixed_split_common import force_sdma_off, reset_ignored_signals, stop_cooperatively  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def prompt(kind):
    if kind == "short":
        return "Explain in a few sentences why the sky is blue during the day and red at sunset."
    n = 195 if kind == "long" else 97
    body = " ".join(f"Item {i}: the measured value of sample {i} was {i * 37 % 1000} units at station {i % 17}." for i in range(n))
    return body + " Summarize the list above in one sentence."


def post(port, payload, stream):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=110) as r:
        if not stream:
            d = json.loads(r.read())
            return {"usage": d.get("usage"), "timings": d.get("timings"), "finish": d["choices"][0].get("finish_reason"),
                    "text": (d["choices"][0]["message"].get("content") or "")[:80], "wall_s": round(time.time() - t0, 3)}
        chunks, first, text, last = 0, None, "", {}
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            d = json.loads(line[5:])
            chunks += 1
            if first is None:
                first = time.time() - t0
            last = d
            for c in d.get("choices", []):
                delta = c.get("delta", {})
                text += delta.get("content") or delta.get("reasoning_content") or ""
        return {"stream_chunks": chunks, "first_chunk_s": round(first or -1, 3), "usage": last.get("usage"),
                "timings": last.get("timings"), "text": text[:80], "wall_s": round(time.time() - t0, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--request", action="append", default=[])
    ap.add_argument("--stream-first", action="store_true", help="the first request streams")
    ap.add_argument("--ready-timeout", type=float, default=100.0)
    ap.add_argument("--temperature", type=float, default=0.0)
    a = ap.parse_args()
    reset_ignored_signals()                              # the server must answer SIGINT (mixed_split_common)
    env = force_sdma_off(dict(os.environ))
    t_start = time.time()
    srv = subprocess.Popen([sys.executable, os.path.join(ROOT, "serve", "server.py"), "--engine", "strata", "--config",
                            a.config, "--port", str(a.port)], cwd=ROOT, env=env)
    rc = 1
    try:
        ready = False
        while time.time() - t_start < a.ready_timeout and srv.poll() is None:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=2) as r:
                    if r.status == 200 and json.loads(r.read()).get("loaded") is True:
                        ready = True
                        break
            except Exception:
                pass
            time.sleep(0.5)
        print(json.dumps({"ready": ready, "after_s": round(time.time() - t_start, 1)}), flush=True)
        if ready:
            rc = 0
            for i, spec in enumerate(a.request):
                kind, _, n = spec.partition(":")
                payload = {"model": "x", "messages": [{"role": "user", "content": prompt(kind)}], "temperature": a.temperature,
                           "max_tokens": int(n or 50), "stream": bool(a.stream_first and i == 0)}
                try:
                    print(json.dumps({"request": spec, **post(a.port, payload, payload["stream"])}), flush=True)
                except Exception as e:
                    print(json.dumps({"request": spec, "error": repr(e)}), flush=True)
                    rc = 1
    finally:
        if stop_cooperatively(srv) is None:              # SIGINT and drain, then SIGTERM; never SIGKILL
            print(json.dumps({"server": "still running after SIGINT and SIGTERM; left alone, no further GPU job",
                              "pid": srv.pid}), flush=True)
            rc = 1
        print(json.dumps({"server_exit": srv.returncode, "alive_s": round(time.time() - t_start, 1)}), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
