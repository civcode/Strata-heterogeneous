#!/usr/bin/env python3
"""Actual bounded greedy generation, independent serial and fresh MTP gates."""
import argparse
import json
from pathlib import Path
import re
import sys

from heterogeneous_compare_parity import read_run, compare
from heterogeneous_mtp_reuse import MODE, accepted_rows, read_offer
from heterogeneous_mtp_three_round import trace_events
from heterogeneous_mtp_acceptance import trace_history, differences
from heterogeneous_mtp_front import compare as compare_front
from heterogeneous_mtp_qualification import TRANSPORT, identity, exact

GEN_MODE = re.compile(r'^HETERO_MTP_GENERATION_MODE prompt=(\d+) max_new=(\d+)$', re.M)
OFFER = re.compile(r'^HETERO_MTP_GENERATION_OFFER cell=(\d+) anchor=(\d+) tokens=([\d,]+) submitted=(\d+)$', re.M)
OUTPUT = re.compile(r'^HETERO_GENERATION (\{.*\})$', re.M)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def ids_valid(ids, vocab):
    return type(ids) is list and all(type(t) is int and 0 <= t < vocab for t in ids)


def reconstruct(marker, run):
    """Derive output from actual committed rows, including last un-fed token."""
    vocab = run[0][1]['vocab']
    prompt, stops, max_new = (marker[k] for k in ('prompt_ids','stop_ids','max_new'))
    mtp = marker['mtp_generation']
    require(type(mtp) is bool and ids_valid(prompt,vocab) and 1 <= len(prompt) <= 7 and
            ids_valid(stops,vocab) and 1 <= len(stops) <= 64 and
            type(max_new) is int and 1 <= max_new <= 12 and len(prompt)+max_new-1 <= 12,
            'invalid bounded generation marker')
    feed_end = len(prompt)+max_new-1
    require(len(run) >= len(prompt), 'missing actual prompt commits')
    source = identity(run)
    require(all(identity([row]) == source for row in run), 'source/plan identity changed within generation')
    prefix, output, points = prompt.copy(), [], []
    for i,(m,p) in enumerate(run[:len(prompt)]):
        require(m['rows']==m['n_keep']==1 and m['pos0']==i and m['token_ids']==[prompt[i]], 'incorrect prompt commit')
    anchor = run[len(prompt)-1][1]['predictions'][0]
    for m,p in run[len(prompt):]:
        ids, preds = m['token_ids'],p['predictions']
        require(m['pos0']==len(prefix) < feed_end and ids[0]==anchor and
                1 <= len(ids)==len(preds)==m['rows'] <= (5 if mtp else 1) and
                len(prefix)+len(ids)<=feed_end and not any(t in stops for t in ids),
                'non-authoritative or out-of-budget generation window')
        keep=1
        while keep<len(ids) and ids[keep]==preds[keep-1]:
            keep+=1
        require(m['n_keep']==keep, 'generation commit differs from greedy prefix')
        points.append(dict(prefix=prefix.copy(),cell=len(prefix)-1,anchor=anchor,submitted=ids[1:],
                           accepted_drafts=keep-1,rejected_drafts=len(ids)-keep))
        prefix.extend(ids[:keep]);output.extend(ids[:keep]);anchor=preds[keep-1]
    if anchor in stops:
        reason, stop_token='stop',anchor
    else:
        require(len(prefix)==feed_end, 'generation stopped before length/stop')
        reason,stop_token='length',-1
        output.append(anchor)
    require(marker['token_ids']==output and marker['finish_reason']==reason and
            marker['stop_token']==stop_token and marker['next']==anchor, 'published output/reason differs from actual target commits')
    final=marker['final_offer']
    require(ids_valid(final,vocab) and (2 <= len(final) <= 5 and final[0]==anchor if mtp else not final),
            'invalid unsubmitted terminal proposal')
    if mtp:
        points.append(dict(prefix=prefix.copy(),cell=len(prefix)-1,anchor=anchor,submitted=final[1:]))
    return dict(prompt=prompt,stops=stops,max_new=max_new,prefix=prefix,output=output,reason=reason,
                stop_token=stop_token,anchor=anchor,points=points,vocab=vocab)


def load_generation(prefix, log, mtp, traced=False):
    marker=json.loads(Path(str(prefix)+'.complete.json').read_text())
    require(marker.get('mtp_generation') is mtp, 'incorrect generation mode')
    run=read_run(prefix,allow_partial=mtp,generation_capture=True)
    info=reconstruct(marker,run)
    modes=TRANSPORT.findall(log)
    require(len(modes)==1,'missing or ambiguous transport')
    outputs=OUTPUT.findall(log)
    require(len(outputs)==1 and json.loads(outputs[0])==dict(prompt_tokens=len(info['prompt']),
            finish_reason=info['reason'],token_ids=info['output']), 'log output differs from completed capture')
    if mtp:
        t='1' if traced else '0'
        require(MODE.findall(log)==[(t,t,'0','0')] and GEN_MODE.findall(log)==[(str(len(info['prompt'])),str(info['max_new']))],
                'wrong MTP generation log mode')
        offers=OFFER.findall(log)
        require(len(offers)==len(info['points']), 'wrong native offer count')
        for i,(point,(cell,anchor,raw,submitted)) in enumerate(zip(info['points'],offers)):
            require(raw.endswith(','),'malformed native candidate list')
            candidates=[int(t) for t in raw[:-1].split(',')]
            require((int(cell),int(anchor))==(point['cell'],point['anchor']) and
                    ids_valid(candidates,info['vocab']) and 1<=len(candidates)<=4,'invalid native offer identity')
            cap=len(candidates)
            if i<len(offers)-1:
                cap=max(0,min(cap,len(info['prompt'])+info['max_new']-2-len(point['prefix'])))
                for j,token in enumerate(candidates[:cap]):
                    if token in info['stops']:
                        cap=j;break
            require(int(submitted)==cap and point['submitted']==candidates[:cap], 'native offer clipping differs from actual submitted window')
            point['candidates']=candidates
    else:
        require(not MODE.findall(log) and not GEN_MODE.findall(log) and not OFFER.findall(log), 'serial generation log contains native MTP')
    return run,info,modes[0]


def analyze(serial_prefix, serial_log, reused_prefix, reused_log, fresh_pairs, traced=False):
    serial,s,transport=load_generation(serial_prefix,serial_log,False)
    reused,r,mode=load_generation(reused_prefix,reused_log,True,traced)
    require(transport==mode and (s['prompt'],s['stops'],s['max_new'])==(r['prompt'],r['stops'],r['max_new']), 'different generation inputs or transport')
    require(identity(serial)==identity(reused),'different generation model/plan/source')
    require(len(fresh_pairs)==len(r['points']) and len({serial_prefix.resolve(),reused_prefix.resolve(),*(p.resolve() for p,_ in fresh_pairs)})==len(fresh_pairs)+2,
            'independent fresh capture required at every actual native offer')
    require(s['prefix']==r['prefix'] and (s['output'],s['reason'],s['stop_token'],s['anchor'])==
            (r['output'],r['reason'],r['stop_token'],r['anchor']),'actual serial and native generated output/state prefix differ')
    accepted=accepted_rows(reused)
    serial_parity=compare(accepted_rows(serial),accepted,0.001,prefill=True)
    history,drafts=None,None
    if traced:
        events,drafts=trace_events(reused_prefix)
        require(len(drafts)==len(r['points']), 'missing native front checkpoint')
        history=trace_history(events,r['prefix'],r['anchor'],tuple(p['cell'] for p in r['points']))
    gates=[]
    for i,((p,log),point) in enumerate(zip(fresh_pairs,r['points'])):
        fresh=read_run(p,mtp_fresh_replay=True)
        require(identity(fresh)==identity(reused) and all(identity([row])==identity(reused) for row in fresh) and
                [m['token_ids'] for m,_ in fresh]==[[t] for t in point['prefix']] and
                fresh[-1][1]['predictions'][0]==point['anchor'], 'wrong independent fresh generation prefix/source/anchor')
        require(TRANSPORT.findall(log)==[transport] and not GEN_MODE.findall(log),'wrong fresh transport/mode')
        offer=read_offer(log,0,r['vocab'],traced=traced)
        parity=compare(fresh,accepted[:len(fresh)],0.001,prefill=True)
        passed=exact(parity) and offer==dict(cell=point['cell'],anchor=point['anchor'],tokens=point['candidates'])
        gate=dict(point,target_parity=parity,fresh_offer=offer,committed_kv='UNMEASURED')
        if traced:
            events,fronts=trace_events(p)
            require(len(fronts)==1,'wrong independent fresh trace history')
            fresh_history=trace_history(events,point['prefix'],point['anchor'],(point['cell'],))
            diff=differences(fresh_history,history[:len(fresh_history)])
            front=compare_front(fronts[0],drafts[i])
            passed=passed and not diff and front['status']=='IDENTICAL' and front['committed_kv_identical'] and front['external_residual_identical']
            gate.update(differences=diff,front=front,committed_kv='DIFFERENT' if any(d['field'] in ('k','v') for d in diff) else 'IDENTICAL')
        gate['status']='PASS' if passed else 'FAIL';gates.append(gate)
    return dict(status='PASS' if exact(serial_parity) and all(g['status']=='PASS' for g in gates) else 'FAIL',
                scope='BOUNDED_ACTUAL_GREEDY_ONLY',transport=transport,traced=traced,output=r['output'],
                finish_reason=r['reason'],stop_token=r['stop_token'],serial_parity=serial_parity,checkpoints=gates,
                accepted_drafts=sum(p['accepted_drafts'] for p in r['points'][:-1]),
                rejected_drafts=sum(p['rejected_drafts'] for p in r['points'][:-1]),
                clipped_drafts=sum(len(p['candidates'])-len(p['submitted']) for p in r['points'][:-1]),
                production_admission='NOT QUALIFIED',resident_mtp='NOT QUALIFIED',physical_rollback='NOT MEASURED')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    derive=sub.add_parser('prefixes');gate=sub.add_parser('compare')
    for command in (derive,gate):
        command.add_argument('--reused-prefix',type=Path,required=True)
        command.add_argument('--reused-log',type=Path,required=True)
        command.add_argument('--traced',action='store_true')
    gate.add_argument('--serial-prefix',type=Path,required=True)
    gate.add_argument('--serial-log',type=Path,required=True)
    gate.add_argument('--fresh',nargs=2,metavar=('PREFIX','LOG'),action='append',required=True,type=Path)
    args=parser.parse_args()
    try:
        if args.action=='prefixes':
            _,info,_=load_generation(args.reused_prefix,args.reused_log.read_text(),True,args.traced)
            result=dict(status='PASS',checkpoints=info['points'],output=info['output'],finish_reason=info['reason'])
        else:
            result=analyze(args.serial_prefix,args.serial_log.read_text(),args.reused_prefix,args.reused_log.read_text(),
                           [(p,l.read_text()) for p,l in args.fresh],args.traced)
        print(json.dumps(result,indent=2,allow_nan=False))
        return 0 if result['status']=='PASS' else 1
    except (OSError,ValueError,TypeError,KeyError,AttributeError,IndexError,OverflowError) as exc:
        print(f'INVALID: {exc}',file=sys.stderr);return 2


if __name__=='__main__':
    sys.exit(main())
