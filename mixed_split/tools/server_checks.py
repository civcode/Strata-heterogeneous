#!/usr/bin/env python3
"""The HTTP checks of report 0069 against a running serve/server.py:  server_checks.py PORT MODEL LOG [OUT_JSON]

health and the model list; one streaming, one non-streaming and one sampled request; a streaming request cancelled
after 8 chunks, then a normal one; a two-turn chat (the reused tokens of turn 2); the same streamed request read fast
and read slowly (50 ms per chunk), with the engine's decode rate for each from LOG (the server's log, where the
engine's "prompt ... generated ..." lines go).  Standard library only; the server is not started or stopped here.
"""
import http.client, json, os, re, socket, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mixed_split_common import mem_sample  # noqa: E402

port, model, log = int(sys.argv[1]), sys.argv[2], sys.argv[3]
out_path = sys.argv[4] if len(sys.argv) > 4 else None
log_at = os.path.getsize(log) if os.path.exists(log) else 0
LINE = re.compile(r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), (\d+) generated in (\d+) ms \(([\d.]+) tok/s\)")
results = {"mem_idle": mem_sample()}


def engine_lines():
    """The engine's request lines the log got since the last call."""
    global log_at
    time.sleep(0.5)
    with open(log, errors="replace") as f:
        f.seek(log_at)
        text = f.read()
        log_at = f.tell()
    out = []
    for m in LINE.finditer(text):
        n, reused, read, pms, ptps, gen, gms, gtps = m.groups()
        out.append({"prompt": int(n), "reused": int(reused), "prefill_tps": float(ptps), "generated": int(gen),
                    "decode_tps": float(gtps), "prefill_ms": int(pms)})
    return out


def get(path):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    c.request("GET", path)
    r = c.getresponse()
    body = r.read()
    c.close()
    return r.status, body


def post(body):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    c.request("POST", "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return c, c.getresponse()


def msgs(t):
    return [{"role": "user", "content": t}]


base = {"model": model, "temperature": 0, "max_tokens": 120, "reasoning_budget_tokens": 64}


def nonstream(body):
    t0 = time.monotonic()
    c, r = post(body)
    d = json.loads(r.read())
    c.close()
    m = d["choices"][0]["message"]
    return {"status": r.status, "s": round(time.monotonic() - t0, 2), "content": (m.get("content") or "")[:120],
            "usage": d.get("usage")}


def stream(body, pause=0.0, stop_after=None):
    t0 = time.monotonic()
    c, r = post({**body, "stream": True})
    n, text, first = 0, "", None
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data:"):
            continue
        p = line[5:].strip()
        if p == "[DONE]":
            break
        n += 1
        if first is None:
            first = time.monotonic() - t0
        for ch in json.loads(p).get("choices", []):
            text += ch.get("delta", {}).get("content") or ""
        if stop_after is not None and n >= stop_after:
            try:
                c.sock.shutdown(socket.SHUT_RDWR)
            except Exception:  # noqa: BLE001
                pass
            break
        if pause:
            time.sleep(pause)
    r.close()
    c.close()
    return {"status": r.status, "chunks": n, "first_chunk_s": round(first or -1, 2), "s": round(time.monotonic() - t0, 2),
            "content": text[:120]}


def step(name, fn):
    try:
        r = fn()
    except Exception as e:  # noqa: BLE001
        r = {"error": repr(e)}
    r["engine"] = engine_lines()
    results[name] = r
    print(name, json.dumps(r), flush=True)


step("health", lambda: {"health": get("/health")[0], "models": get("/v1/models")[0],
                        "model_ids": [m.get("id") for m in json.loads(get("/v1/models")[1]).get("data", [])]})
step("stream", lambda: stream({**base, "messages": msgs("Explain in three sentences why the sky is blue.")}))
step("nonstream", lambda: nonstream({**base, "messages": msgs("Write a haiku about a lighthouse in winter.")}))
step("sampled", lambda: nonstream({**base, "temperature": 0.8, "top_p": 0.95,
                                   "messages": msgs("Give me a name for a small sailing boat, and say why.")}))
step("cancel", lambda: stream({**base, "max_tokens": 400, "messages": msgs("Write a long story about a dragon.")}, stop_after=8))
time.sleep(2.0)
step("after_cancel", lambda: nonstream({**base, "messages": msgs("Name three primary colors.")}))
q1 = "Explain in three sentences how a refrigerator keeps food cold."
t1 = nonstream({**base, "messages": msgs(q1)})
t1["engine"] = engine_lines()
results["turn1"] = t1
print("turn1", json.dumps(t1), flush=True)
step("turn2", lambda: nonstream({**base, "messages": msgs(q1) + [{"role": "assistant", "content": t1["content"]}] +
                                 msgs("And why does it hum at night? Two sentences.")}))
reader = {**base, "max_tokens": 200, "messages": msgs("Describe a walk through an old harbour town at dawn, in detail.")}
step("fast_reader", lambda: stream(reader))
step("slow_reader", lambda: stream(reader, pause=0.05))
step("fast_reader2", lambda: stream(reader))
step("slow_reader2", lambda: stream(reader, pause=0.05))
results["mem_end"] = mem_sample()
if out_path:
    json.dump(results, open(out_path, "w"), indent=1)
