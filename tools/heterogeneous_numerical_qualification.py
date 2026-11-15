#!/usr/bin/env python3
"""Aggregate already validated measurements; no production admission or threshold changes."""
import argparse
import json
import math
import statistics
from pathlib import Path

def need(ok,msg):
    if not ok: raise ValueError(msg)

def sample(xs):
    need(isinstance(xs,list) and len(xs)>=3 and all(type(x) in (int,float) and math.isfinite(x) and x>0 for x in xs),
         "need at least three positive timing samples")
    return {"median":statistics.median(xs),"min":min(xs),"max":max(xs)}

def summarize(d):
    need(d.get("schema")=="strata-numerical-qualification/v1","schema")
    cases=d["cases"]; need(isinstance(cases,list) and bool(cases),"no cases")
    seen=set(); out=[]
    for c in cases:
        key=(c["plan"],c["weight_contract"],c["activation_mode"],c["transport"],c["cut"],c["chunk"])
        need(key not in seen,"duplicate case"); seen.add(key)
        need(c["interop"] in ("PASS","FAIL","NOT_RUN"),"interop")
        m=c.get("quality")
        if m is None:
            need(c["interop"]=="NOT_RUN","missing numerical evidence")
            quality="NOT_RUN"
        else:
            kl=m["max_kl_ref_to_hetero"]
            need(type(kl) in (int,float) and math.isfinite(kl) and kl>=0 and
                 all(type(m[k]) is int and m[k]>=0 for k in ("prediction_disagreements","top1_disagreements")),"invalid quality")
            quality="PASS" if kl<=.001 and m["prediction_disagreements"]==m["top1_disagreements"]==0 else "FAIL"
        timings=c.get("matched_latency_ns")
        latency=None if timings is None else {"baseline":sample(timings["baseline"]),"candidate":sample(timings["candidate"])}
        if latency is not None:
            latency["ratio"]=latency["candidate"]["median"]/latency["baseline"]["median"]
        out.append({"case":key,"quality":quality,"interop":c["interop"],"latency":latency,
                    "peak_memory_bytes":c.get("peak_memory_bytes")})
    return {"status":"DIAGNOSTIC_ONLY","cases":out,"kl_threshold":.001,
            "production_admission":"NOT_QUALIFIED","coverage":"REQUIRES_REVIEW"}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--manifest",type=Path,required=True);a=p.parse_args()
    try: print(json.dumps(summarize(json.loads(a.manifest.read_text())),indent=2,allow_nan=False));return 0
    except (ValueError,KeyError,TypeError,OSError,OverflowError) as e: p.exit(2,f"INVALID: {e}\n")
if __name__=="__main__": raise SystemExit(main())
