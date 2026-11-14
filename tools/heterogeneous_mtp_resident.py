#!/usr/bin/env python3
"""Compare a clean bounded resident MTP session with fresh actual generation.

No worker launch, GPU APIs or claim of physical suffix erasure.
"""
import argparse
import json
from pathlib import Path
import re
import sys

from heterogeneous_compare_parity import read_run, compare
from heterogeneous_mtp_generation import load_generation, reconstruct, OFFER, OUTPUT
from heterogeneous_mtp_reuse import MODE
from heterogeneous_mtp_generation import GEN_MODE
from heterogeneous_mtp_qualification import TRANSPORT, identity, exact
from heterogeneous_mtp_three_round import trace_events
from heterogeneous_mtp_front import compare as compare_front

RESET = re.compile(r'^HETERO_MTP_RESIDENT_RESET request=(\d+)$', re.M)
START = re.compile(r'^HETERO_MTP_RESIDENT_REQUEST request=(\d+)$', re.M)
PROGRESS = re.compile(r'^HETERO_PROGRESS (\d+) (\d+) (\d+)$', re.M)
CLOSED = re.compile(r'^HETERO_CLOSED$', re.M)


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def marker(prefix, suffix='.complete.json'):
    return json.loads(Path(str(prefix)+suffix).read_text())


def session_segments(log, count):
    """Associate request logs by reset/start/output boundaries, not copied logs."""
    resets, starts, outputs = (list(rx.finditer(log)) for rx in (RESET, START, OUTPUT))
    require([int(m[1]) for m in resets] == list(range(1,count+1)) and
            [int(m[1]) for m in starts] == list(range(1,count+1)) and
            len(outputs)==count and len(CLOSED.findall(log))==1, 'missing/duplicate session lifecycle records')
    end=list(CLOSED.finditer(log))[0].start()
    parts=[]
    last=-1
    for i,(reset,start,output) in enumerate(zip(resets,starts,outputs),1):
        require(last < reset.start() < start.start() < output.start() < end,
                'incorrect reset/request/output ordering')
        published=json.loads(output[1])
        require(published.get('request_id')==i,'incorrect published request ID')
        parts.append((log[start.end():output.start()],published))
        last=output.end()
    # A proposal or progress record outside a request is not qualified.
    require(sum(len(OFFER.findall(part)) for part,_ in parts)==len(OFFER.findall(log)) and
            sum(len(PROGRESS.findall(part)) for part,_ in parts)==len(PROGRESS.findall(log)),
            'offer or progress outside request boundaries')
    return parts


def raw_offers(part):
    out=[]
    for cell,anchor,raw,submitted in OFFER.findall(part):
        require(raw.endswith(','), 'malformed native candidates')
        ids=[int(t) for t in raw[:-1].split(',')]
        require(1<=len(ids)<=4 and int(submitted)<=len(ids),'invalid native offer length')
        out.append((int(cell),int(anchor),ids,int(submitted)))
    return out


def trace_match(prefix, ref_prefix, cancelled, fed):
    # Same evaluated inputs and window shapes imply the same native operation
    # history. Cancellation never drafts the final retained residual.
    actual, drafts=trace_events(prefix)
    reference, ref_drafts=trace_events(ref_prefix)
    if cancelled:
        ends=[i+1 for i,(m,_) in enumerate(reference)
              if m['phase'].endswith('_after') and m['upto']==fed-1]
        require(len(ends)==1,'missing reference committed cancellation cut')
        reference=reference[:ends[0]]
        ref_drafts=ref_drafts[:sum(m['phase']=='draft_after' for m,_ in reference)]
    require(len(actual)==len(reference) and len(drafts)==len(ref_drafts),
            'extra/missing native trace operation after reset or cancellation')
    diff=[]
    for (a,ad),(b,bd) in zip(actual,reference):
        require(a==b,'native trace event identities differ')
        for field in ad:
            if ad[field]!=bd[field]:
                diff.append(dict(event=a['event'],cell=a['cell'],field=field))
    fronts=[compare_front(a,b) for a,b in zip(drafts,ref_drafts)]
    return dict(status='PASS' if not diff and all(f['status']=='IDENTICAL' and
                f['committed_kv_identical'] and f['external_residual_identical'] for f in fronts) else 'FAIL',
                differences=diff, fronts=fronts, compared_events=len(actual))


def analyze(prefix, log, refs):
    session=marker(prefix,'.resident.json')
    count=session['requests']; traced=session['traced']; transport=session['transport']
    require(session['schema']=='strata-mtp-resident/v1' and type(count) is int and count in (3,7) and
            type(traced) is bool and transport in ('shared','socket') and TRANSPORT.findall(log)==[transport],
            'expected completed three-request A/B/A or seven-request cancellation session')
    p,b=session['prompt_rows'],session['max_new']
    require(type(p) is int and type(b) is int and 1<=p<=7 and 1<=b<=12 and p+b-1<=12,
            'invalid fixed resident shape')
    t='1' if traced else '0'
    require(MODE.findall(log)==[(t,t,'0','0')] and GEN_MODE.findall(log)==[(str(p),str(b))],
            'expected one native owner with fixed generation policy')
    require(len(refs)==2 and len({q.resolve() for q,_ in refs}|{prefix.resolve()})==3,
            'two distinct independent nonresident reference captures required')
    references={}
    for path,ref_log in refs:
        run,info,mode=load_generation(path,ref_log,True,traced)
        key=tuple(info['prompt'])
        require(len(key)==p and info['max_new']==b and mode==transport and key not in references,
                'wrong reference prompt/budget/transport or duplicate prompt')
        require(identity(run)[0:2]==(session['plan'],session['model_fingerprint']), 'reference plan/model mismatch')
        references[key]=(path,run,info,raw_offers(ref_log))
    parts=session_segments(log,count)
    results=[]
    a_prompt=None
    for i,(part,published) in enumerate(parts,1):
        path=Path(str(prefix)+f'.r{i}')
        require(path.resolve() not in {q.resolve() for q,_ in refs},'resident reference aliases a live capture')
        m=marker(path)
        require(m['mtp_resident'] is True and m['mtp_generation'] is True and m['request_id']==i and
                m['max_new']==b and len(m['prompt_ids'])==p and m['plan']==session['plan'] and
                m['model_fingerprint']==session['model_fingerprint'],'request completion identity mismatch')
        key=tuple(m['prompt_ids'])
        if i==1:a_prompt=key
        require((key!=a_prompt if i==2 else key==a_prompt) and key in references,'wrong A/B/A prompt sequence')
        rp,ref,info,offers=references[key]
        run=read_run(path,allow_partial=True,generation_capture=True,resident_request=i)
        require(len(run)<=len(ref) and all(identity([row])==identity(ref) for row in run) and
                m['stop_ids']==info['stops'],'resident model/source/stop mismatch')
        fed=sum(x['n_keep'] for x,_ in run)
        last,lastp=run[-1]
        require(m['fed_rows']==fed and m['next']==lastp['predictions'][last['n_keep']-1],
                'resident fed extent or final target anchor mismatch')
        require(PROGRESS.findall(part)==[(str(i),str(j),str(x['pos0']+x['n_keep']))
                for j,(x,_) in enumerate(run,1)], 'missing or noncommitted progress boundary')
        require(published==dict(prompt_tokens=p,request_id=i,stop_token=m['stop_token'],
                finish_reason=m['finish_reason'],token_ids=m['token_ids']), 'published resident output mismatch')
        cancelled=i in (4,6)
        received=raw_offers(part)
        if cancelled:
            require(m['finish_reason']=='cancelled' and m['token_ids']==[] and
                    m['stop_token']==-1 and m['final_offer']==[] and len(run)>p,
                    'cancel must discard buffered output at a committed decode boundary')
            require(1<=last['n_keep']<last['rows'] and (last['n_keep']==1 if i==4 else last['n_keep']>1),
                    'cancellation must follow rejection-only then partial acceptance')
            require(received==offers[:len(run)-p],'cancelled native offer history differs')
        else:
            actual=reconstruct(m,run)
            require(len(run)==len(ref) and
                    all(actual[k]==info[k] for k in ('prefix','output','reason','stop_token','anchor')),
                    'resident generated output differs from independent actual generation')
            require(received==offers and m['final_offer']==[offers[-1][1]]+offers[-1][2],
                    'resident native offer history or terminal proposal differs')
        parity=compare(ref[:len(run)],run,0.001,prefill=True)
        state=trace_match(path,rp,cancelled,fed) if traced else dict(status='UNMEASURED')
        results.append(dict(request=i,cancelled=cancelled,finish_reason=m['finish_reason'],fed_rows=fed,
                            output=m['token_ids'],last_keep=last['n_keep'],last_rows=last['rows'],
                            target_parity=parity,state=state,status='PASS' if exact(parity) and
                            (not traced or state['status']=='PASS') else 'FAIL'))
    return dict(status='PASS' if all(r['status']=='PASS' for r in results) else 'FAIL',
                transport=transport,traced=traced,requests=results,
                scope='BOUNDED_RESIDENT_FIXED_SHAPE',cancellation_coverage=count==7,
                physical_suffix_erasure='NOT MEASURED',production_admission='NOT QUALIFIED')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session-prefix',type=Path,required=True)
    p.add_argument('--session-log',type=Path,required=True)
    p.add_argument('--reference',type=Path,nargs=2,action='append',required=True,metavar=('PREFIX','LOG'))
    a=p.parse_args()
    try:
        result=analyze(a.session_prefix,a.session_log.read_text(),[(q,l.read_text()) for q,l in a.reference])
        print(json.dumps(result,indent=2,allow_nan=False))
        return 0 if result['status']=='PASS' else 1
    except (OSError,ValueError,TypeError,KeyError,AttributeError,IndexError,OverflowError) as e:
        print(f'INVALID: {e}',file=sys.stderr);return 2


if __name__=='__main__':
    sys.exit(main())
