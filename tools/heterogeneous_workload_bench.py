#!/usr/bin/env python3
"""Sequential benchmark of the existing loopback OpenAI-compatible service.

No GPU runtime, model launcher, external tool execution, or new inference path.
JSONL evidence includes every attempt. Client timestamps are not GPU timing.
"""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import statistics
import time
from urllib.parse import urlsplit

SCHEMA = "strata-heterogeneous-workloads/v1"

def need(ok, msg):
    if not ok:
        raise ValueError(msg)

def endpoint(url):
    p = urlsplit(url)
    need(p.scheme == "http" and p.hostname in ("127.0.0.1", "::1", "localhost")
         and not p.username and not p.password and p.path in ("", "/")
         and not p.query and not p.fragment and p.port is not None,
         "explicit loopback http URL with port required")
    return p.hostname, p.port

def request(url, body, key="", timeout=600, cancel_after_first=False):
    host, port = endpoint(url)
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    start = time.monotonic()
    first = visible = None
    content, reasoning = [], []
    usage = timings = finish = None
    calls = {}
    done = False
    r = None
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    try:
        conn.request("POST", "/v1/chat/completions", json.dumps(body).encode(), headers)
        r = conn.getresponse()
        if r.status != 200:
            return {"status": "HTTP_ERROR", "http_status": r.status, "elapsed_s": time.monotonic()-start}
        if not body.get("stream"):
            raw = r.read(16*1024*1024+1)
            need(len(raw) <= 16*1024*1024, "response exceeds cap")
            j = json.loads(raw)
            need(not j.get("error") and len(j["choices"]) == 1, "bad completion")
            choice = j["choices"][0]
            message = choice["message"]
            need(isinstance(message, dict) and (message.get("content") is None or isinstance(message.get("content"), str)),
                 "invalid message")
            finish, usage, timings = choice["finish_reason"], j.get("usage"), j.get("timings")
            need(finish in ("stop", "length", "tool_calls"), "missing terminal finish")
            return {"status":"COMPLETE", "elapsed_s":time.monotonic()-start, "ttft_s":None,
                    "first_visible_s":None, "finish_reason":finish, "message":message,
                    "usage":usage, "server_timings":timings}
        count = 0
        while True:
            raw = r.readline(2*1024*1024+1)
            if not raw:
                break
            count += len(raw)
            need(time.monotonic()-start <= timeout, "overall streaming deadline exceeded")
            need(len(raw) <= 2*1024*1024 and count <= 32*1024*1024, "stream exceeds cap")
            if not raw.startswith(b"data:"):
                continue
            payload = raw[5:].strip()
            if payload == b"[DONE]":
                done = True
                break
            j = json.loads(payload)
            need(not j.get("error"), "SSE error")
            if j.get("usage") is not None: usage = j["usage"]
            if j.get("timings") is not None: timings = j["timings"]
            for ch in j.get("choices", []):
                need(ch.get("index", 0) == 0, "multiple completion choices")
                delta = ch.get("delta") or {}
                now = time.monotonic()-start
                text = delta.get("content") or ""
                thought = delta.get("reasoning_content") or ""
                fragments = delta.get("tool_calls") or []
                if text or thought or fragments:
                    if first is None: first = now
                if text:
                    if visible is None: visible = now
                    content.append(text)
                    if cancel_after_first:
                        return {"status":"CLIENT_CANCELLED", "elapsed_s":now, "ttft_s":first,
                                "first_visible_s":visible}
                if thought: reasoning.append(thought)
                for f in fragments:
                    ix = f["index"]
                    need(type(ix) is int and 0 <= ix < 8, "invalid tool index")
                    record = calls.setdefault(ix, {"id":"", "type":"function",
                        "function":{"name":"", "arguments":""}})
                    record["id"] += f.get("id") or ""
                    for k in ("name", "arguments"):
                        record["function"][k] += (f.get("function") or {}).get(k) or ""
                if ch.get("finish_reason") is not None:
                    need(finish is None, "duplicate terminal finish")
                    finish = ch["finish_reason"]
        need(done and finish in ("stop","length","tool_calls"), "incomplete SSE termination")
        message = {"role":"assistant", "content":"".join(content)}
        if reasoning: message["reasoning_content"] = "".join(reasoning)
        if calls: message["tool_calls"] = [calls[i] for i in sorted(calls)]
        return {"status":"COMPLETE", "elapsed_s":time.monotonic()-start, "ttft_s":first,
                "first_visible_s":visible, "finish_reason":finish, "message":message,
                "usage":usage, "server_timings":timings}
    except (OSError, http.client.HTTPException, ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
        return {"status":"ERROR", "error_type":type(exc).__name__, "elapsed_s":time.monotonic()-start,
                "ttft_s":first, "first_visible_s":visible}
    finally:
        if r is not None:
            r.close()
        conn.close()

def score(result, expected):
    if result["status"] != "COMPLETE":
        return "FAIL"
    if expected is None:
        return "UNSCORED"
    if result["finish_reason"] != "stop":
        return "FAIL"
    t = result["message"].get("content") or ""
    if expected["kind"] == "exact":
        return "PASS" if t.strip() == expected["value"] else "FAIL"
    if expected["kind"] == "json":
        try:
            return "PASS" if json.loads(t) == expected["value"] else "FAIL"
        except ValueError:
            return "FAIL"
    raise ValueError("unknown scorer")

def validate_manifest(doc):
    need(doc.get("schema")==SCHEMA and isinstance(doc.get("cases"),list) and doc["cases"],
         "invalid workload manifest")
    seen=set()
    for c in doc["cases"]:
        need(isinstance(c.get("id"),str) and c["id"] and c["id"] not in seen, "duplicate case")
        seen.add(c["id"])
        need(isinstance(c.get("messages"),list) and c["messages"] and
             type(c.get("max_tokens")) is int and 1<=c["max_tokens"]<=4096, "case input")
        need(type(c.get("stream",True)) is bool and type(c.get("cancel",False)) is bool and
             (not c.get("cancel") or c.get("stream",True)), "case flags")
        e=c.get("expected")
        need(e is None or (isinstance(e,dict) and e.get("kind") in ("exact","json") and "value" in e),
             "case scorer")
        if "fixture_tool" in c:
            f=c["fixture_tool"]
            need(isinstance(c.get("tools"),list) and isinstance(f,dict) and
                 set(f)=={"name","arguments","result"} and isinstance(f["name"],str) and
                 isinstance(f["arguments"],dict) and isinstance(f["result"],str) and not c.get("cancel"),
                 "invalid fixed tool fixture")

def run_case(base, model, case, key, timeout, sender=request):
    body={"model":model,"messages":case["messages"],"max_tokens":case["max_tokens"],
          "temperature":0,"top_p":1,"top_k":1,"stream":case.get("stream",True),
          "reasoning_effort":"none","chat_template_kwargs":{"enable_thinking":False}}
    if "tools" in case: body["tools"]=case["tools"]
    result=sender(base,body,key,timeout,case.get("cancel",False))
    stages=[result]
    if case.get("cancel"):
        # Socket close is not proof of a native cancellation ACK; a separate recovery case follows.
        return {"stages":stages,"quality":"UNSCORED" if result["status"]=="CLIENT_CANCELLED" else "FAIL"}
    if "fixture_tool" in case:
        f=case["fixture_tool"]
        try:
            calls=result["message"]["tool_calls"]
            need(result["status"]=="COMPLETE" and result["finish_reason"]=="tool_calls" and len(calls)==1,
                 "tool call missing")
            call=calls[0]
            need(call["type"]=="function" and call["id"] and call["function"]["name"]==f["name"] and
                 json.loads(call["function"]["arguments"])==f["arguments"], "unexpected tool request")
        except (KeyError, TypeError, ValueError, IndexError):
            return {"stages":stages,"quality":"FAIL"}
        # Only the predeclared immutable fixture result; never execute model-generated code or read arbitrary files.
        body={**body,"messages":[*body["messages"],result["message"],
              {"role":"tool","tool_call_id":call["id"],"content":f["result"]}]}
        result=sender(base,body,key,timeout)
        stages.append(result)
    return {"stages":stages,"quality":score(result,case.get("expected"))}

def aggregate(records):
    groups={}
    for row in records:
        groups.setdefault(row["id"],[]).append(row)
    out={}
    for name, rows in groups.items():
        good=[s for r in rows for s in r["stages"] if s["status"]=="COMPLETE"]
        def median(k):
            values=[s[k] for s in good if s.get(k) is not None]
            return statistics.median(values) if values else None
        rates=[]
        for s in good:
            n=(s.get("usage") or {}).get("completion_tokens")
            if type(n) is int and n>=0 and s["elapsed_s"]>0:
                rates.append(n/s["elapsed_s"])
        out[name]={"attempts":len(rows),"quality_pass":sum(r["quality"]=="PASS" for r in rows),
                   "quality_fail":sum(r["quality"]=="FAIL" for r in rows),
                   "unscored":sum(r["quality"]=="UNSCORED" for r in rows),
                   "completed_http_stages":len(good),
                   "finish_reasons":{k:sum(s.get("finish_reason")==k for s in good)
                                     for k in ("stop","length","tool_calls")},
                   "median_http_elapsed_s":median("elapsed_s"),"median_client_ttft_s":median("ttft_s"),
                   "median_end_to_end_completion_tokens_s":statistics.median(rates) if rates else None}
    return out

def freeze_replay(output, max_tokens):
    from agent_turns_bench import turns, TOOLS
    cases=[{"id":tag,"messages":messages,"tools":TOOLS,"max_tokens":max_tokens}
           for tag,messages in turns()]
    with output.open("x",encoding="utf-8") as f:
        json.dump({"schema":SCHEMA,"description":"repository agent turn replay, manual quality review",
                   "cases":cases},f,indent=2)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeze-agent-replay",type=Path)
    p.add_argument("--max-tokens",type=int,default=1024)
    p.add_argument("--url")
    p.add_argument("--model")
    p.add_argument("--manifest",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--label",default="experimental")
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--timeout",type=float,default=600)
    a=p.parse_args()
    if a.freeze_agent_replay:
        need(1<=a.max_tokens<=4096,"max-tokens")
        freeze_replay(a.freeze_agent_replay,a.max_tokens)
        return 0
    need(a.url and a.model and a.manifest and a.output and 1<=a.repeats<=20 and
         0<a.timeout<=7200,"missing or invalid benchmark options")
    endpoint(a.url)
    raw=a.manifest.read_bytes()
    doc=json.loads(raw); validate_manifest(doc)
    key=os.environ.get("STRATA_KEY","")
    rows=[]
    # Record each row durably before starting the next request; never overwrite a run.
    with a.output.open("x",encoding="utf-8") as f:
        def emit(row):
            f.write(json.dumps(row,ensure_ascii=True,allow_nan=False)+"\n");f.flush()
        emit({"kind":"header","schema":SCHEMA,"label":a.label,"model":a.model,
              "manifest_sha256":hashlib.sha256(raw).hexdigest(),"repeats":a.repeats})
        for repeat in range(a.repeats):
            for case in doc["cases"]:
                r={"kind":"case","id":case["id"],"repeat":repeat,
                   **run_case(a.url,a.model,case,key,a.timeout)}
                emit(r);rows.append(r)
        summary=aggregate(rows)
        emit({"kind":"summary","cases":summary,
              "production_admission":"NOT_QUALIFIED","scope":"actual API observations; unscored cases require review"})
    print(json.dumps(summary,indent=2))
    return 1 if any(r["quality"]=="FAIL" for r in rows) else 0

if __name__=="__main__":
    raise SystemExit(main())
