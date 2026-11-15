#!/usr/bin/env python3
"""Read-only native numerical trace analysis; not a production admission gate."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

N, HC, NE, K = 2560, 4, 512, 10
TOKENS = (248045, 846, 198, 2523, 513, 32767, 264)
FIELDS = {'r_attn.f32': HC*N, 'attn_mixed.f32': N, 'attn_bo.f32': N,
          'r_ffn.f32': HC*N, 'ffn_mixed.f32': N, 'router_logits.f32': NE,
          'router_ids.i32': K, 'router_weights.f32': K, 'ffn_bo.f32': N, 'ffn_inj.f32': HC}


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def raw(path, size):
    p = Path(path)
    require(0 <= size <= 16*1024*1024 and p.is_file() and not p.is_symlink() and
            p.stat().st_size == size, f'invalid extent: {p}')
    b = p.read_bytes()
    require(len(b) == size, f'truncated: {p}')
    return b


def read(path, dtype, count):
    dt = np.dtype(dtype)
    a = np.frombuffer(raw(path, count*dt.itemsize), dt)
    if dt.kind == 'f':
        require(np.isfinite(a).all(), f'nonfinite: {path}')
    return a


def bf16(bits):
    a = (np.asarray(bits, dtype='<u2').astype('<u4') << 16).view('<f4').astype(np.float64)
    require(np.isfinite(a).all(), 'nonfinite BF16')
    return a


def metric(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    require(a.shape == b.shape and a.size and np.isfinite(a).all() and np.isfinite(b).all(), 'invalid metric')
    d = a-b
    return {'max_abs': float(np.max(np.abs(d))), 'rmse': float(np.sqrt(np.mean(d*d))),
            'relative_l2_to_second': float(np.linalg.norm(d.ravel()))/max(float(np.linalg.norm(b.ravel())), 1e-30)}


def exact(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def top(logits, k=K):
    require(np.isfinite(logits).all() and 0 < k < len(logits), 'invalid router logits')
    ids = np.argsort(-logits, kind='stable')[:k]
    ranked = np.sort(logits)[::-1]
    return ids, float(ranked[k-1]-ranked[k])


def route_observation(logits, actual_ids, actual_weights):
    actual_ids, actual_weights = np.asarray(actual_ids), np.asarray(actual_weights)
    k = len(actual_ids)
    require(k and len(set(actual_ids.tolist())) == k and np.all((actual_ids >= 0) & (actual_ids < len(logits))) and
            np.isfinite(actual_weights).all() and np.all(actual_weights >= 0), 'invalid actual routes')
    ids, gap = top(logits.astype(np.float64), k)
    p = np.exp(logits.astype(np.float64)-float(np.max(logits)))
    p /= p.sum()
    weights = p[actual_ids]/max(p[actual_ids].sum(), 2**-14)
    return {'ids': actual_ids.tolist(), 'fp64_top_ids_from_observed_logits': ids.tolist(),
            'set_matches': set(ids.tolist()) == set(actual_ids.tolist()), 'rank10_to_11_gap': gap,
            'weights_error': metric(actual_weights, weights)}


def router_reference(w, a, b):
    """FP64 actual-operand projections and counterfactuals, not expert execution."""
    wa = w
    wb = w.astype(np.float16).astype(np.float64) if b['format'] == 'f16' else w
    require(np.isfinite(wb).all(), 'unsupported FP16 weight range')
    x, y = a['ffn_mixed.f32'].astype(np.float64), b['ffn_mixed.f32'].astype(np.float64)
    la, lb = a['router_logits.f32'].astype(np.float64), b['router_logits.f32'].astype(np.float64)
    ra, rb = wa@a['operand'], wb@b['operand']
    fa, fb = w@x, w@y
    # All deltas are vectors. Representation contribution excludes upstream change.
    observed, predicted = lb-la, rb-ra
    k = len(a['router_ids.i32'])
    ids_a, ids_b = set(a['router_ids.i32'].tolist()), set(b['router_ids.i32'].tolist())
    return {'serial_format': a['format'], 'native_format': b['format'],
            'serial_local_projection_error': metric(la, ra), 'native_local_projection_error': metric(lb, rb),
            'observed_delta': metric(lb, la), 'predicted_delta': metric(rb, ra),
            'unexplained_delta_vector': metric(observed, predicted),
            'upstream_only_delta': metric(fb, fa), 'representation_delta': metric(rb-fb, ra-fa),
            'actual_changed_experts': len(ids_a-ids_b), 'route_order_exact': exact(a['router_ids.i32'],b['router_ids.i32']),
            'actual_serial': route_observation(la, a['router_ids.i32'], a['router_weights.f32']),
            'actual_native': route_observation(lb, b['router_ids.i32'], b['router_weights.f32']),
            'fp64_top_serial_actual_operand': top(ra, k)[0].tolist(),
            'fp64_top_native_actual_operand': top(rb, k)[0].tolist(),
            'fp64_top_serial_unrounded': top(fa, k)[0].tolist(),
            'fp64_top_native_unrounded': top(fb, k)[0].tolist()}


def completed_r(record):
    # Native HC write: R += bo * 2*sigmoid(inject/HC). FP64, not GPU bit emulation.
    gate = record['ffn_inj.f32'].astype(np.float64)/HC
    e = np.exp(-np.abs(gate))
    sig = np.where(gate >= 0, 1/(1+e), e/(1+e))
    return (record['r_ffn.f32'].reshape(HC, -1).astype(np.float64) +
            2*sig[:, None]*record['ffn_bo.f32'].astype(np.float64)).ravel()


def row(prefix, request, layer, pos, cut):
    stem = f'{prefix}.trace.r{request}.l{layer}.p{pos}.'
    d = {key: read(stem+key, '<i4' if key.endswith('i32') else '<f4', count)
         for key, count in FIELDS.items()}
    formats = [f for f in ('f32', 'bf16', 'f16') if Path(stem+'router_operand.'+f).exists()]
    is_prompt = request == 2 and pos < cut
    require(len(formats) == 1 and formats[0] in (('bf16', 'f16') if is_prompt else ('f32', 'bf16')),
            'missing or invalid operand representation')
    fmt = formats[0]
    a = read(stem+'router_operand.'+fmt, {'f32':'<f4','bf16':'<u2','f16':'<f2'}[fmt], N)
    d['operand'] = bf16(a) if fmt == 'bf16' else a.astype(np.float64)
    low = Path(stem+'router_operand_lo.bf16')
    if low.exists():
        require(is_prompt and fmt == 'bf16', 'unexpected low operand')
        d['operand'] += bf16(read(low, '<u2', N))
        fmt = 'bf16x2'
    d['format'] = fmt
    if is_prompt:
        d['r_out.f32'] = read(stem+'r_out.f32', '<f4', HC*N)
    route_observation(d['router_logits.f32'], d['router_ids.i32'], d['router_weights.f32'])
    return d


def compare_state(prefix, capture, cut):
    reports = {}
    for stage in range(3):
        native = f'{prefix}.native-state.s{stage}.state.r2.p{cut}'
        meta = capture['states'][str(stage)]['identity']
        serial = [f'{prefix}.trace.r{r}.s{stage}.p{cut}.state.' for r in (1,3)]
        for q in meta['qsa']:
            expected = [q[k] for k in ('format','cells','page_size','heads','head_dim','pooled_rows','idx_dim')]
            for s in serial:
                require(read(s+f'qsa{q["ordinal"]}.layout.i32', '<i4', 7).tolist()==expected, 'serial K/V layout differs')
        prev = [read(s+'ple_prev.i32','<i4',2).tolist() for s in serial]
        require(prev[0]==prev[1], 'unstable serial PLE predecessors')
        fields = {'ple_prev': {'exact': prev[0]==meta['ple_prev'], 'serial':prev[0], 'native':meta['ple_prev']}}
        for key, kind, size in meta['fields']:
            n = raw(native+'.'+key+'.bin',size)
            a,b = [raw(s+key+'.bin',size) if size else b'' for s in serial]
            require(a==b, f'unstable serial cut state: stage {stage} {key}')
            fields[key]={'exact':a==n,'bytes':size}
            if kind=='f32':
                fields[key].update(metric(np.frombuffer(n,'<f4'), np.frombuffer(a,'<f4')))
            else:
                fields[key]['different_bytes']=sum(x!=y for x,y in zip(a,n))
        reports[str(stage)]=fields
    return reports


def digest(path):
    require(path.is_file() and not path.is_symlink(), f'invalid evidence file: {path}')
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def observer_check(prefix, baseline):
    def files(root):
        return {p.name[len(root.name):]:p for p in root.parent.glob(root.name+'.*')
                if p.name != root.name+'.prefill.json' and not p.name.startswith(root.name+'.trace.')}
    a,b=files(prefix),files(baseline)
    require(a and a.keys()==b.keys(), 'baseline file inventory differs')
    for key in sorted(a):
        require(digest(a[key])==digest(b[key]), f'observer changed baseline file {key}')
    return len(a)


def stage_row(prefix, stage, pos, cut, chunk, direction):
    start=pos//chunk*chunk
    rows=min(chunk,cut-start)
    a=read(f'{prefix}.native.stage{stage}.p{start}.{direction}.f32','<f4',rows*HC*N).reshape(rows,HC*N)
    return a[pos-start]


def analyze(prefix, baseline):
    from heterogeneous_prefill_stage_gate import read_capture, experiment_identity
    c,b = read_capture(prefix), read_capture(baseline)
    m,old = c['marker'],b['marker']; cut,chunk=m['prefix_rows'],m['chunk']
    require(experiment_identity(m)==experiment_identity(old), 'observer baseline must use same HC experiment')
    require(prefix.resolve()!=baseline.resolve() and m.get('numerical_trace') is True and
            not old.get('numerical_trace') and not m['stage_reference'] and not old['stage_reference'] and
            (cut,chunk) in ((2,1),(2,2),(6,3),(6,6)) and tuple(m['token_ids'])==TOKENS[:cut+1], 'invalid trace case')
    require(all(m[k]==old[k] for k in ('prefix_rows','chunk','token_ids','transport')) and c['source']==b['source'], 'baseline identity mismatch')
    for x in (c,b):
        require(x['bridge']['status']=='PASS' and x['local_links_exact'] and x['serial_controls_exact'], 'integrity check failed')
    require(all(c[k]==b[k] for k in ('residuals','states','logits')), 'observer changed qualified baseline')
    overlap_count=observer_check(prefix,baseline)
    for r in (1,2,3):
        for stage in range(3):
            for pos in range(cut+1):
                require(read(f'{prefix}.trace.r{r}.s{stage}.p{pos}.token.i32','<i4',1)[0]==TOKENS[pos], 'trace token identity mismatch')
    states=compare_state(prefix,c,cut)
    layers=[]
    first_route,first_phase,previous={},{},{}
    for layer in range(48):
        stem=f'{prefix}.trace.'
        wa,wb=[read(stem+mode+f'.l{layer}.router.bf16','<u2',N*NE) for mode in ('serial','native')]
        require(exact(wa,wb), f'actual router weight differs at layer {layer}')
        w=bf16(wa).reshape(NE,N)
        for pos in range(cut+1):
            a,n,z=[row(prefix,r,layer,pos,cut) for r in (1,2,3)]
            require(a['format']==z['format'] and all(exact(v,z[k]) for k,v in a.items() if isinstance(v,np.ndarray)), 'unstable serial phase/control')
            phases={k:dict(exact=exact(n[k],a[k]),**metric(n[k],a[k])) for k in FIELDS if k.endswith('f32')}
            if str(pos) not in first_phase:
                for field in FIELDS:
                    if not exact(n[field],a[field]):
                        first_phase[str(pos)]={'layer':layer,'field':field};break
            stage=0 if layer<12 else 1 if layer<30 else 2
            if pos<cut:
                if layer in (0,12,30):
                    require(exact(n['r_attn.f32'],stage_row(prefix,stage,pos,cut,chunk,'in')), 'native layer entry is not stage input')
                if layer in (11,29,47):
                    require(exact(n['r_out.f32'],stage_row(prefix,stage,pos,cut,chunk,'out')), 'native layer output is not stage output')
                if layer and layer!=1:
                    require(exact(n['r_attn.f32'],previous[pos][1]), 'native adjacent layer residual differs')
            rr=router_reference(w,a,n)
            if rr['actual_changed_experts'] and str(pos) not in first_route:
                first_route[str(pos)] = layer
            item={'layer':layer,'position':pos,'kind':'prompt' if pos<cut else 'continuation',
                  'phase_errors':phases,'router':rr,'completed_r_estimate':metric(completed_r(n),completed_r(a))}
            if pos<cut:
                item['native_write_reference_error']=metric(n['r_out.f32'],completed_r(n))
                item['native_actual_vs_serial_reconstructed_r']=metric(n['r_out.f32'],completed_r(a))
            if layer and layer!=1:
                item['serial_previous_write_reference_error']=metric(a['r_attn.f32'],previous[pos][0])
            previous[pos]=(completed_r(a),n['r_out.f32'] if pos<cut else completed_r(n))
            layers.append(item)
    return {'status':'DIAGNOSTIC_ONLY','case':[cut,chunk],'transport':m['transport'],'baseline_exact':True,
            'quality':c['quality'],'baseline_files_exact':overlap_count,'first_differing_phase_by_position':first_phase,
            'first_changed_route_layer_by_position':first_route,'layers':layers,'cut_state':states,
            'root_cause':'REQUIRES_REVIEW; router counterfactuals do not execute downstream experts or prove whole-model causality',
            'production_admission':'NOT_QUALIFIED','bulk_mtp':'NOT_ENABLED'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    a=p.parse_args()
    try:
        print(json.dumps(analyze(a.prefix,a.baseline),indent=2,allow_nan=False));return 0
    except (OSError,ValueError,TypeError,KeyError,IndexError,AttributeError,OverflowError) as e:
        print(f'INVALID: {e}',file=sys.stderr);return 2


if __name__=='__main__':sys.exit(main())
