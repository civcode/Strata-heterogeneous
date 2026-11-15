#!/usr/bin/env python3
"""Read-only teacher-forced corpus comparison of completed placement captures.

Each input item explicitly identifies a document; rows predict the NEXT token.
Never labels this diagnostic a production admission. No token sampling.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import numpy as np
from heterogeneous_compare_parity import read_run

def need(ok, message):
    if not ok:
        raise ValueError(message)

def metrics(a,b,target):
    a,b=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    need(a.ndim==1 and a.shape==b.shape and a.size>1 and np.isfinite(a).all() and np.isfinite(b).all()
         and type(target) is int and 0<=target<a.size,"invalid logits/target")
    x=a-max(a); y=b-max(b)
    x-=np.log(np.exp(x).sum()); y-=np.log(np.exp(y).sum())
    return {"nll_ref":float(-x[target]),"nll_candidate":float(-y[target]),
            "kl":max(0.,float(np.dot(np.exp(x),x-y))),
            "top1_equal":bool(np.argmax(a)==np.argmax(b))}

def corpus(doc, loader=read_run):
    need(doc.get("schema")=="strata-quality-corpus/v1" and isinstance(doc.get("documents"),list)
         and bool(doc["documents"]),"missing corpus")
    documents=[]; ids=set(); source=None; plans=None; contracts=None
    for d in doc["documents"]:
        name=d["id"]; tokens=d["tokens"]
        need(isinstance(name,str) and name and name not in ids and isinstance(tokens,list)
             and len(tokens)>=2 and all(type(x) is int and x>=0 for x in tokens),"invalid document")
        ids.add(name)
        paths=[Path(d[k]) for k in ("reference","candidate")]
        need(all(p.is_absolute() for p in paths) and paths[0].resolve()!=paths[1].resolve(),"distinct absolute captures required")
        marks=[json.loads(Path(str(p)+".complete.json").read_text()) for p in paths]
        modes=tuple(m.get("hc_weight_contract") for m in marks)
        need(all(m is None or m in ("bf16","q8") for m in modes),"invalid weight contract")
        runs=[loader(p,placement=True) for p in paths]
        need(len(runs[0])==len(runs[1])==len(tokens)-1,"missing teacher-forced rows")
        rows=[]
        for pos,((ma,pa),(mb,pb)) in enumerate(zip(*runs)):
            keys=("model_fingerprint","pack_index_sha256","gguf_shards","profile_path")
            sa=tuple(ma[k] for k in keys); sb=tuple(mb[k] for k in keys)
            if source is None: source=sa
            if plans is None: plans=(ma["plan"],mb["plan"]); contracts=modes
            need(sa==sb==source and (ma["plan"],mb["plan"])==plans and modes==contracts,"mixed corpus identity")
            need(ma["pos0"]==mb["pos0"]==pos and ma["token_ids"]==mb["token_ids"]==[tokens[pos]]
                 and pa["rows"]==pb["rows"]==1,"wrong teacher-forced position")
            rows.append(metrics(pa["logits"],pb["logits"],tokens[pos+1]))
        documents.append({"id":name,"tokens_sha256":hashlib.sha256(json.dumps(tokens,separators=(",",":")).encode()).hexdigest(),
                          "rows":rows})
    rows=[r for d in documents for r in d["rows"]]
    avg=lambda k:statistics.fmean(r[k] for r in rows)
    a,b=avg("nll_ref"),avg("nll_candidate")
    return {"status":"DIAGNOSTIC_ONLY","documents":len(documents),"scored_tokens":len(rows),
            "plans":plans,"weight_contracts":contracts,"mean_nll_reference":a,"mean_nll_candidate":b,
            "perplexity_reference":math.exp(a),"perplexity_candidate":math.exp(b),
            "mean_kl":avg("kl"),"max_kl":max(r["kl"] for r in rows),"top1_agreement":avg("top1_equal"),
            "per_document":documents,"production_admission":"NOT_QUALIFIED"}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True,type=Path)
    a=p.parse_args()
    try:
        print(json.dumps(corpus(json.loads(a.manifest.read_text())),indent=2,allow_nan=False))
        return 0
    except (ValueError,KeyError,TypeError,OSError,OverflowError) as e:
        p.exit(2,f"INVALID: {e}\n")
if __name__=="__main__": raise SystemExit(main())
