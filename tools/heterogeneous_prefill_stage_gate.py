#!/usr/bin/env python3
"""Bounded native stage replay evidence and unchanged, separate serial quality gate."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from heterogeneous_compare_parity import compare, require
from heterogeneous_prefill_parity import load, MAX_KL

PLAN = 'cuda:0:0-12,hip:0:12-30,hip:1:30-48'
INDEX = 'c344c01f59f547ed8a59003857a82eb90499e5299154241dbca787acb54b5b7c'
ROW_BYTES = 4 * 2560 * 4
RANGES = ((0, 12), (12, 30), (30, 48))
A = (248045, 846, 198, 2523, 513, 32767, 264)
CASES = {(1, 1, A[:2]), (2, 1, A[:3]), (2, 2, A[:3]), (6, 3, A), (6, 6, A)}
MODES = ('serial-a', 'native', 'serial-b')


def signature(path, size, kind):
    require(type(size) is int and 0 <= size <= 16 * 1024 * 1024 and
            path.is_file() and not path.is_symlink() and path.stat().st_size == size,
            f'invalid sample extent: {path}')
    raw = path.read_bytes()
    require(len(raw) == size, f'truncated sample: {path}')
    if kind == 'f32':
        require(size > 0 and size % 4 == 0 and np.isfinite(np.frombuffer(raw, '<f4')).all(),
                f'nonfinite or invalid FP32 sample: {path}')
    return hashlib.sha256(raw).hexdigest()


def read_state(stem, stage, cut):
    meta = json.loads(Path(stem + '.json').read_text())
    lo, hi = RANGES[stage]
    q0, q1 = lo // 4, hi // 4
    g0, g1 = lo - q0, hi - q1
    require(meta.get('schema') == 'strata-heterogeneous-prefix-state/v1' and
            (meta.get('stage'), meta.get('request'), meta.get('position')) == (stage, 2, cut) and
            (meta.get('layer_lo'), meta.get('layer_hi')) == (lo, hi) and
            (meta.get('gdn_ord0'), meta.get('gdn_alloc'), meta.get('qsa_ord0'), meta.get('qsa_alloc')) ==
            (g0, g1-g0, q0, q1-q0), 'invalid native state carve/identity')
    prev = meta.get('ple_prev')
    require(type(prev) is list and len(prev) == 2 and
            all(type(t) is int and -1 <= t < 248320 for t in prev), 'invalid PLE predecessor metadata')
    expected = {'ple.history': ('f32', 9*10240*4)}
    for j in range(g0, g1):
        expected[f'gdn{j}.recurrence'] = ('f32', 128*48*128*4)
        expected[f'gdn{j}.conv'] = ('f32', 10240*3*4)
    layouts = meta.get('qsa')
    require(type(layouts) is list and len(layouts) == q1-q0, 'incorrect QSA count')
    for j, q in zip(range(q0, q1), layouts):
        require(q.get('ordinal') == j and type(q.get('format')) is int and
                all(type(q.get(k)) is int and q[k] > 0 for k in ('cells','page_size','pooled_rows','heads','head_dim','idx_dim')) and
                q['cells'] == ((cut+q['page_size']-1)//q['page_size'])*q['page_size'] and
                q['cells'] <= 2048 and q['pooled_rows'] == cut//4+1 and
                (q['heads'],q['head_dim'],q['idx_dim']) == (2,256,128), 'invalid QSA layout')
        for name, kind, size in (('tail','f32',3*128*4),('dead','f32',128*4),('block_pos','i32',4),
                                 ('pooled','f32',q['pooled_rows']*128*4),('k','opaque',None),
                                 ('v','opaque',None),('k_scale','opaque',None),('v_scale','opaque',None)):
            expected[f'qsa{j}.{name}'] = (kind,size)
    fields = meta.get('fields')
    require(type(fields) is list and len(fields) == len(expected), 'missing state fields')
    seen, digests, sizes = set(), {}, {}
    for f in fields:
        name, kind, size = f.get('name'), f.get('type'), f.get('bytes')
        require(name in expected and name not in seen and kind == expected[name][0], 'invalid or duplicate state field')
        fixed = expected[name][1]
        require(type(size) is int and ((size == fixed) if fixed is not None else
                (size in (cut*288,cut*512,cut*1024) if name.endswith(('.k','.v')) else size in (0,cut*16))),
                f'wrong state extent: {name}')
        seen.add(name); sizes[name] = size
        digests[name] = signature(Path(stem+'.'+name+'.bin'),size,kind)
    for j in range(q0,q1):
        k,v,ks,vs = (sizes[f'qsa{j}.{s}']//cut for s in ('k','v','k_scale','v_scale'))
        require((k,v,ks,vs) in ((1024,1024,0,0),(512,512,16,16),(288,288,0,0),(512,288,16,0)),
                'unsupported or inconsistent K/V byte layout')
    identity = {k: v for k,v in meta.items() if k != 'fields'}
    identity['fields'] = sorted((f['name'],f['type'],f['bytes']) for f in fields)
    return {'identity':identity,'sha256':digests}


def packet_signatures(run):
    return [{'pos':m['pos0'],'tokens':m['token_ids'],'predictions':p['predictions'],
             'sha256':hashlib.sha256(np.asarray(p['logits'],dtype='<f4').tobytes()).hexdigest()}
            for m,p in run]


def experiment_identity(mark):
    if 'hc_experiment' not in mark:
        return None
    exp = mark['hc_experiment']
    require(type(exp) is dict and set(exp) == {'mode', 'backend'} and
            exp['mode'] in ('bf16x2', 'fp32') and exp['backend'] in ('all', 'cuda', 'hip'),
            'invalid HC experiment identity')
    return exp


def read_capture(prefix):
    mark,runs,bridge = load(prefix)
    experiment_identity(mark)
    cut,chunk=mark['prefix_rows'],mark['chunk']
    require(mark.get('stage_capture') is True and mark.get('state_capture') is True and
            mark.get('first_token_capture') is False and mark.get('hc_replay_capture') is False and
            mark.get('transport') in ('shared','socket') and isinstance(mark.get('stage_reference'),str) and
            mark['plan']==PLAN and (mark['hc'],mark['n_embd'])==(4,2560) and
            1<=chunk<=cut<=16 and 1<=len(mark['token_ids'])-cut<=128, 'not a completed bounded stage capture')
    m,p=runs['native'][0]
    require(m['pack_index_sha256']==INDEX and p['vocab']==248320,'unapproved source index or vocabulary')
    source=(mark['model_fingerprint'],p['fingerprint'],m['pack_index_sha256'],m['gguf_shards'],m['profile_path'])
    residuals, states, links = {}, {}, []
    for pos in range(0,cut,chunk):
        size=min(chunk,cut-pos)*ROW_BYTES
        for stage in range(3):
            for direction in ('in','out'):
                name=f'stage{stage}.p{pos}.{direction}'
                residuals[name]=signature(Path(f'{prefix}.native.{name}.f32'),size,'f32')
        for stage in range(2):
            links.append(residuals[f'stage{stage}.p{pos}.out']==residuals[f'stage{stage+1}.p{pos}.in'])
        for stage,direction in ((0,'out'),(1,'in')):
            raw=signature(Path(f'{prefix}.stage{stage}.r2.w{pos+1}.{direction}.f32'),size,'f32')
            links.append(raw==residuals[f'stage{stage}.p{pos}.{direction}'])
    for stage in range(3):
        states[str(stage)]=read_state(f'{prefix}.native-state.s{stage}.state.r2.p{cut}',stage,cut)
    # Existing CUDA verifier adoption observer must see the identical state.
    adopted=read_state(f'{prefix}.state.r2.p{cut}',0,cut)
    expected_prev=([-1]+mark['token_ids'][:cut])[-2:]
    links.append(adopted==states['0'])
    links.append(states['0']['identity']['ple_prev']==expected_prev)
    logits={mode:packet_signatures(runs[mode]) for mode in MODES}
    quality={f'{mode}_vs_native':compare(runs[mode],runs['native'],MAX_KL,prefill=True)
             for mode in ('serial-a','serial-b')}
    stable=logits['serial-a']==logits['serial-b']
    return {'marker':mark,'source':source,'bridge':bridge,'local_links_exact':all(links),
            'serial_controls_exact':stable,'residuals':residuals,'states':states,'logits':logits,'quality':quality}


def pair(wire, reference):
    w,r=read_capture(wire),read_capture(reference)
    a,b=w['marker'],r['marker']
    case=(a['prefix_rows'],a['chunk'],tuple(a['token_ids']))
    require(experiment_identity(a)==experiment_identity(b), 'wire/reference HC experiment differs')
    require(case==(b['prefix_rows'],b['chunk'],tuple(b['token_ids'])) and w['source']==r['source'] and
            a['transport']==b['transport'] and a['stage_reference']=='' and
            Path(b['stage_reference']).resolve()==wire.resolve() and wire.resolve()!=reference.resolve(),
            'wire/reference source, shape, transport or identity mismatch')
    residual_diff=[k for k in w['residuals'] if w['residuals'][k]!=r['residuals'][k]]
    state_diff={s: [k for k,v in w['states'][s]['sha256'].items() if v!=r['states'][s]['sha256'][k]]
                for s in w['states']}
    state_identity=all(w['states'][s]['identity']==r['states'][s]['identity'] for s in w['states'])
    interop=(all(x['bridge']['status']=='PASS' and x['local_links_exact'] and x['serial_controls_exact'] for x in (w,r)) and
             not residual_diff and not any(state_diff.values()) and state_identity and w['logits']==r['logits'])
    quality=all(q['status']=='PASS' for x in (w,r) for q in x['quality'].values())
    return {'case':case,'transport':a['transport'],'wire':str(wire),'reference':str(reference),
            'interop':'PASS' if interop else 'FAIL','quality':'PASS' if quality else 'FAIL',
            'residual_differences':residual_diff,'state_differences':state_diff,'state_identity_exact':state_identity,
            'continuation_exact':w['logits']==r['logits'],'wire_evidence':w,'reference_evidence':r}


def matrix(pairs):
    require(bool(pairs),'no replay cases')
    lookup={}
    for p in pairs:
        key=(p['case'],p['transport'])
        require(key not in lookup,'duplicate matrix case/transport')
        lookup[key]=p
    source=pairs[0]['wire_evidence']['source']
    require(all(p['wire_evidence']['source']==source for p in pairs),'matrix model source changed')
    hc_exp=experiment_identity(pairs[0]['wire_evidence'].get('marker', {}))
    require(all(experiment_identity(p['wire_evidence'].get('marker', {}))==hc_exp for p in pairs),
            'cannot mix HC experiment modes/backends in one qualification matrix')
    cross={}
    for case in {p['case'] for p in pairs}:
        if all((case,t) in lookup for t in ('shared','socket')):
            a,b=[lookup[(case,t)]['wire_evidence'] for t in ('shared','socket')]
            cross[str(case)]=all(a[k]==b[k] for k in ('residuals','states','logits'))
    complete=set(lookup)=={(c,t) for c in CASES for t in ('shared','socket')}
    interop=all(p['interop']=='PASS' for p in pairs) and all(cross.values())
    quality=all(p['quality']=='PASS' for p in pairs)
    return {'scope':'bounded stage-local native replay, not independent full vanilla caller',
            'coverage':'COMPLETE' if complete else 'INCOMPLETE','cross_transport_exact':cross,
            'interoperability':'PASS' if complete and interop else 'FAIL' if not interop else 'INCOMPLETE',
            'numerical_quality':'PASS' if complete and quality else 'FAIL' if not quality else 'INCOMPLETE',
            'status':'BOUNDED_MATRIX_PASS' if complete and interop and quality else 'NOT_QUALIFIED',
            'kl_threshold':MAX_KL,'production_admission':'NOT_QUALIFIED','bulk_mtp':'NOT_ENABLED','pairs':pairs}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pair',nargs=2,action='append',required=True,type=Path,metavar=('WIRE','REFERENCE'))
    a=p.parse_args()
    try:
        result=matrix([pair(w,r) for w,r in a.pair])
        print(json.dumps(result,indent=2,allow_nan=False))
        return 0 if result['status']=='BOUNDED_MATRIX_PASS' else 1
    except (OSError,ValueError,TypeError,KeyError,AttributeError,IndexError,OverflowError) as e:
        print(f'INVALID: {e}',file=sys.stderr);return 2


if __name__=='__main__':sys.exit(main())
