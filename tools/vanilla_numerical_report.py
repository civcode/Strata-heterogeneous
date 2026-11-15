#!/usr/bin/env python3
"""Summarize validated analyzer JSON without altering evidence or admission."""
import argparse
import json
import math
from pathlib import Path
from vanilla_numerical_oracle import hc_cost

PLANS={"forward":((0,12,"cuda"),(12,30,"hip0"),(30,48,"hip1")),
       "reverse":((0,18,"hip0"),(18,36,"hip1"),(36,48,"cuda"))}
def number(x):
    if type(x) not in (int,float) or not math.isfinite(x) or x<0:
        raise ValueError("invalid metric")
    return x

def summarize(trace,topology):
    if trace.get("status")!="DIAGNOSTIC_ONLY" or trace.get("baseline_exact") is not True:
        raise ValueError("requires validated same-mode trace")
    cut,chunk=trace["case"]
    if (cut,chunk) not in ((2,1),(2,2),(6,3),(6,6)) or trace["transport"] not in ("shared","socket"):
        raise ValueError("case identity")
    plan=PLANS[topology]
    seen=set(); stages=[]
    for lo,hi,owner in plan:
        rows=[r for r in trace["layers"] if lo<=r["layer"]<hi]
        swaps=0; fragile=0; max_error=0; first=None
        for r in sorted(rows,key=lambda r:(r["layer"],r["position"])):
            key=(r["layer"],r["position"])
            if key in seen or not 0<=key[1]<=cut:
                raise ValueError("duplicate/invalid layer row")
            seen.add(key)
            router=r["router"]
            count=router["actual_changed_experts"]
            if type(count) is not int or not 0<=count<=10:
                raise ValueError("route count")
            swaps+=count
            if count and first is None: first=list(key)
            margin=number(router["actual_serial"]["rank10_to_11_gap"])
            err=number(router["observed_delta"]["max_abs"])
            fragile+=margin<=2*err
            max_error=max(max_error,err)
        stages.append({"owner":owner,"range":[lo,hi],"changed_expert_slots":swaps,
                       "first_route_change":first,"rows_without_margin_certificate":fragile,
                       "max_router_logit_delta":max_error})
    if seen!={(l,p) for l in range(48) for p in range(cut+1)} or len(trace["layers"])!=len(seen):
        raise ValueError("incomplete trace")
    return {"case":[cut,chunk],"transport":trace["transport"],"topology":topology,
            "quality":trace["quality"],"stages":stages,"cost_per_chunk":hc_cost(tokens=chunk),
            "confidence":"diagnostic metrics only; no causal or production admission claim"}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace",type=Path,required=True)
    p.add_argument("--topology",choices=PLANS,required=True,
                   help="must match independently validated source capture plan")
    a=p.parse_args()
    try:
        doc=json.loads(a.trace.read_text())
        if a.topology=="reverse" and doc.get("plan")!="hip:0:0-18,hip:1:18-36,cuda:0:36-48":
            raise ValueError("reverse trace must explicitly identify its plan")
        if a.topology=="forward" and doc.get("plan","cuda:0:0-12,hip:0:12-30,hip:1:30-48")!="cuda:0:0-12,hip:0:12-30,hip:1:30-48":
            raise ValueError("forward plan mismatch")
        print(json.dumps(summarize(doc,a.topology),indent=2,allow_nan=False))
        return 0
    except (ValueError,KeyError,TypeError,OSError,OverflowError) as e:
        p.exit(2,f"INVALID: {e}\n")
if __name__=="__main__":
    raise SystemExit(main())
