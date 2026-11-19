#!/usr/bin/env python3
"""Drive one engine over the serve protocol: start it, send GEN requests one after another, end it by closing its input.

    engine_drive.py OUT_PREFIX PROMPTS_JSON REQUESTS [--shared-arena] [--env K=V ...] -- ENGINE ARGV...

REQUESTS: 'name:max_new[:key=v+key=v],...' (the optional keys go on the GEN line as the server sends them, e.g.
cvec=0, pipe=2000 or ckpt=0).  PROMPTS_JSON (mkprompts.py) maps a name to token ids, or to a list of
parts: a list of ids, {"prompt": NAME} (that request's prompt as sent) or {"out": NAME} (the tokens that request
generated), so a second turn can be built from the first one's answer.
--shared-arena: a memfd for vanilla's --shared-expert-arena is made here and appended to the engine's arguments (a
mixed split passes it on to its front stage).
Writes OUT_PREFIX.err (the engine's stderr) and OUT_PREFIX.json (every request's tokens, DONE fields, the reused
tokens, memory samples at READY and at the end).  The engine is never signalled: its input is closed and it drains.
"""
import json, os, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mixed_split_common import force_sdma_off, mem_sample, shared_arena  # noqa: E402


def main():
    if "--" not in sys.argv:
        raise SystemExit(__doc__)
    own, argv = sys.argv[1:sys.argv.index("--")], sys.argv[sys.argv.index("--") + 1:]
    out_prefix, pj, reqs = own[0], own[1], own[2]
    env = dict(os.environ)
    i = 3
    arena = None
    while i < len(own):
        if own[i] == "--shared-arena":
            arena = shared_arena()
            argv += ["--shared-expert-arena", arena]
        elif own[i] == "--env":
            k, v = own[i + 1].split("=", 1)
            env[k] = v
            i += 1
        i += 1
    force_sdma_off(env)                                  # .ai/PROJECT.md, the first rule
    prompts = json.load(open(pj))
    sent, made = {}, {}

    def resolve(name):
        p = prompts[name]
        if p and isinstance(p[0], int):
            return list(p)
        ids = []
        for part in p:
            if isinstance(part, dict):
                ids += sent[part["prompt"]] if "prompt" in part else made[part["out"]]
            else:
                ids += part
        return ids

    err = open(out_prefix + ".err", "w")
    t0 = time.monotonic()
    p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1, env=env)
    res = {"argv": argv, "arena": arena, "requests": []}

    def line():
        s = p.stdout.readline()
        if not s:
            raise SystemExit("engine stdout ended (rc=%s)" % p.poll())
        return s.rstrip("\n")

    try:
        while True:
            s = line()
            if s.startswith("READY"):
                break
            if s.startswith("ERR"):
                raise SystemExit(s)
        res["ready_s"] = time.monotonic() - t0
        time.sleep(3.0)   # idle
        res["mem_ready"] = mem_sample()
        print("READY after %.1f s; idle: %s" % (res["ready_s"], json.dumps(res["mem_ready"])), flush=True)
        for spec in reqs.split(","):
            name, mx, *kv = spec.split(":")
            keys = "".join(" " + k for k in kv[0].split("+")) if kv and kv[0] else ""
            ids = resolve(name)
            sent[name] = ids
            ts = time.monotonic()
            p.stdin.write("GEN %d%s %s\n" % (int(mx), keys, ",".join(map(str, ids))))
            p.stdin.flush()
            toks, first, lps = [], None, []
            while True:
                s = line()
                if s.startswith("LP "):
                    lps.append(s)
                elif s.startswith("T "):
                    toks.append(int(s.split()[1]))
                    if first is None:
                        first = time.monotonic() - ts
                elif s.startswith("DONE") or s.startswith("ERR"):
                    done = s
                    break
            made[name] = toks
            f = done.split()
            r = {"name": name, "keys": keys.strip(), "prompt": len(ids), "tokens": toks, "lp": lps, "done": done, "wall_s": time.monotonic() - ts,
                 "first_token_s": first}
            if f[0] == "DONE":
                gen, n, pms, dms, reused = int(f[1]), int(f[2]), float(f[3]), float(f[4]), int(f[8])
                r.update(reused=reused, prefill_tps=(n - reused) / (pms / 1000) if pms > 0 else None,
                         decode_tps=gen / (dms / 1000) if dms > 0 else None, drafts="%s/%s" % (f[6], f[7]), finish=f[5],
                         prefill_ms=pms)
            res["requests"].append(r)
            print(json.dumps({k: v for k, v in r.items() if k not in ("tokens", "lp")}), "first ids", toks[:12], flush=True)
        res["mem_end"] = mem_sample()
    finally:
        try:
            p.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            rc = p.wait(timeout=120)
        except subprocess.TimeoutExpired:
            rc = "still running after 120 s"
        res["rc"] = rc
        res["total_s"] = time.monotonic() - t0
        json.dump(res, open(out_prefix + ".json", "w"))
        print("engine rc", rc, "total %.1f s" % res["total_s"], flush=True)


if __name__ == "__main__":
    main()
