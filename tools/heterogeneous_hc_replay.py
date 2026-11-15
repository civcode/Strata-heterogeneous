#!/usr/bin/env python3
"""Native-only HC replay and independent FP64 arithmetic contract analysis.

The real-model adapter supplies immutable device observations; the standalone
CUDA executable is independent of its transport. No quality/admission gate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

N, HC, LR = 2560, 4, 320
EPS = float(np.float32(1e-6))


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def shapes(n=N, hc=HC, rank=LR):
    d = n * hc
    inputs = {'residual.f32': d, 'norm_weight.f32': d,
              'down_weight.bf16': rank*d, 'up_weight.bf16': d*rank,
              'inject_weight.bf16': hc*d}
    prompt = {'scale.f32': hc, 'normalized.bf16': d, 'down.f32': rank,
              'silu.bf16': rank, 'up.f32': d, 'inject.f32': hc, 'mixed.f32': n}
    serial = {'scale.f32': hc, 'inject.f32': hc, 'mixed.f32': n}
    return inputs, prompt, serial


def dtype(name):
    return np.dtype('<u2' if name.endswith('.bf16') else '<f4')


def decode_bf16(bits):
    return (np.asarray(bits, dtype='<u2').astype('<u4') << 16).view('<f4').astype(np.float64)


def bf16_bits(x):
    # Native rounding first takes a finite FP32 value, then BF16 RN-even.
    f = np.asarray(x, dtype='<f4')
    require(np.isfinite(f).all(), 'nonfinite BF16 rounding input')
    u = f.view('<u4')
    return ((u + np.uint32(0x7fff) + ((u >> 16) & 1)) >> 16).astype('<u2')


def real(a, name):
    x = decode_bf16(a) if name.endswith('.bf16') else np.asarray(a, dtype=np.float64)
    require(np.isfinite(x).all(), f'nonfinite {name}')
    return x


def load_fields(path_for, layout):
    result = {}
    for name, count in layout.items():
        p = path_for(name)
        dt = dtype(name)
        require(p.is_file() and p.stat().st_size == count * dt.itemsize, f'invalid extent: {p}')
        raw = p.read_bytes()
        require(len(raw) == count*dt.itemsize, f'truncated input: {p}')
        a = np.frombuffer(raw, dtype=dt)
        real(a, name)
        result[name] = a
    return result


def same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def metric(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    require(a.shape == b.shape and a.size and np.isfinite(a).all() and np.isfinite(b).all(),
            'invalid metric operands')
    delta = a-b
    norm = float(np.linalg.norm(b.ravel()))
    return {'max_abs': float(np.max(np.abs(delta))),
            'rmse': float(np.sqrt(np.mean(delta*delta))),
            'relative_l2_to_reference': float(np.linalg.norm(delta.ravel()))/max(norm, 1e-30)}


def compare_fields(a, b):
    require(a.keys() == b.keys(), 'observed fields differ')
    return {k: dict(bytes_equal=same(a[k], b[k]), **metric(real(a[k], k), real(b[k], k))) for k in a}


def sigmoid(x):
    # Stable on both signs, no clipping of finite source operands.
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1/(1+e), e/(1+e))


def oracle(inputs, prompt, serial, n=N, hc=HC, rank=LR):
    """Independent real-valued HC equations using actual stored weights.

    BF16 models isolate exactly the two operand-rounding boundaries; reductions
    and nonlinear operations remain FP64, intentionally not emulated GPU kernels.
    Local residual errors are evaluated against actual preceding operands.
    """
    x = real(inputs['residual.f32'], 'residual.f32').reshape(hc,n)
    w = real(inputs['norm_weight.f32'], 'norm_weight.f32').reshape(hc,n)
    wd = decode_bf16(inputs['down_weight.bf16']).reshape(rank,hc*n)
    wu = decode_bf16(inputs['up_weight.bf16']).reshape(hc*n,rank)
    wi = decode_bf16(inputs['inject_weight.bf16']).reshape(hc,hc*n)
    scale = 1/np.sqrt(np.mean(x*x, axis=1)+EPS)
    xn = (x*scale[:,None]*w).ravel()
    def silu(down):
        z=down/hc
        return z*sigmoid(z)
    def mix(normalized, up):
        return np.mean((normalized*sigmoid(up)).reshape(hc,n), axis=0)
    def model(round_norm, round_silu):
        operand=decode_bf16(bf16_bits(xn)) if round_norm else xn
        down=wd@operand
        lo=silu(down)
        if round_silu:
            lo=decode_bf16(bf16_bits(lo))
        up=wu@lo
        return {'scale':scale,'normalized':operand,'down':down,'silu':lo,'up':up,
                'inject':wi@operand,'mixed':mix(xn, up)}
    full=model(False,False)
    norm_only=model(True,False)
    silu_only=model(False,True)
    bf=model(True,True)
    pv=real(prompt['mixed.f32'],'mixed.f32')
    sv=real(serial['mixed.f32'],'mixed.f32')
    a_scale=real(prompt['scale.f32'],'scale.f32')
    a_xn=(x*a_scale[:,None]*w).ravel()
    a_bf=decode_bf16(prompt['normalized.bf16'])
    a_down=real(prompt['down.f32'],'down.f32')
    a_lo=decode_bf16(prompt['silu.bf16'])
    a_up=real(prompt['up.f32'],'up.f32')
    v_scale=real(serial['scale.f32'],'scale.f32')
    v_xn=(x*v_scale[:,None]*w).ravel()
    local_prompt={
        'scale':metric(a_scale,scale),
        'normalized_bf16':metric(a_bf,decode_bf16(bf16_bits(a_xn))),
        'down':metric(a_down,wd@a_bf),
        'silu_bf16':metric(a_lo,decode_bf16(bf16_bits(silu(a_down)))),
        'up':metric(a_up,wu@a_lo),
        'inject':metric(prompt['inject.f32'],wi@a_bf),
        'mixed':metric(pv,mix(a_xn,a_up))}
    local_serial={'scale':metric(v_scale,scale),
                  'inject':metric(serial['inject.f32'],wi@v_xn),
                  'down_and_up_raw':'NOT_OBSERVED_IN_FUSED_KERNEL',
                  'silu':'NOT_OBSERVED_IN_FUSED_KERNEL',
                  'mixed_from_observed_silu':'NOT_OBSERVED_IN_FUSED_KERNEL'}
    if 'silu.f32' in serial:
        v_lo=real(serial['silu.f32'],'silu.f32')
        local_serial['silu']=metric(v_lo,silu(wd@v_xn))
        local_serial['mixed_from_observed_silu']=metric(sv,mix(v_xn,wu@v_lo))
    predicted=bf['mixed']-full['mixed']
    actual=pv-sv
    return {'reference_contract':'FP64 reductions and nonlinearities; BF16 RN-even at selected operand boundaries',
            'prompt_vs_full_fp64':metric(pv,full['mixed']),
            'verifier_vs_full_fp64':metric(sv,full['mixed']),
            'prompt_vs_bf16_contract':metric(pv,bf['mixed']),
            'verifier_vs_bf16_contract':metric(sv,bf['mixed']),
            'ideal_normalized_rounding_delta':metric(norm_only['mixed'],full['mixed']),
            'ideal_silu_rounding_delta':metric(silu_only['mixed'],full['mixed']),
            'ideal_both_rounding_delta':metric(bf['mixed'],full['mixed']),
            'ideal_observed_input_rounding':metric(bf['normalized'],xn),
            'observed_native_delta':metric(pv,sv),
            'actual_minus_predicted_delta':metric(actual,predicted),
            'local_prompt_errors':local_prompt,'local_verifier_errors':local_serial,
            'root_cause':'REQUIRES_NUMERICAL_REVIEW; NO AUTOMATIC TOLERANCE OR ADMISSION CHANGE'}


def qfuse_from_log(log):
    rows = re.findall(r'^STRATA_GR0_PATH down_q8=(\d+) up_q8=(\d+) down_scale=(\d+) up_scale=(\d+) qfuse=(\d+) ',log,re.M)
    require(rows and len(set(rows)) == 1, 'missing or inconsistent actual verifier HC dispatch')
    d,u,ds,us,q=map(int,rows[0])
    require((d,u,ds,us)==(0,0,0,0) and q in (0,1), 'not plain BF16 verifier HC dispatch')
    return bool(q)


def analyze(prefix, log, replay, expected_pci):
    from heterogeneous_prefill_parity import load
    from heterogeneous_layer0_arithmetic import load_probe
    marker,runs,bridge=load(prefix)
    require(marker.get('hc_replay_capture') is True and marker.get('first_token_capture') is True and
            marker.get('state_capture') is True and marker['prefix_rows']==marker['chunk']==1 and
            marker['hc']==HC and marker['n_embd']==N and len(marker['token_ids'])==2 and
            marker['plan']=='cuda:0:0-12,hip:0:12-30,hip:1:30-48', 'not completed bounded HC capture')
    require(bridge['status']=='PASS', 'physical bridge differs; investigate separately')
    qfuse=qfuse_from_log(log)
    done=json.loads((replay/'complete.json').read_text())
    require(done.get('schema')=='strata-native-hc-replay/v1' and
            (done.get('n_embd'),done.get('hc'),done.get('rank'),done.get('rows'))==(N,HC,LR,1) and
            type(done.get('qfuse')) is bool and done['qfuse']==qfuse and type(done.get('q8_written')) is bool and
            (qfuse or not done['q8_written']) and done.get('cuda_device')==0 and
            str(done.get('pci')).lower()==expected_pci.lower(), 'invalid replay completion or device/dispatch mismatch')
    inp_layout,pf_layout,v_layout=shapes()
    path=lambda name:Path(f'{prefix}.hc.{name}')
    inputs=load_fields(path,inp_layout)
    replay_inputs=load_fields(lambda name:replay/'inputs'/name,inp_layout)
    require(all(same(inputs[k],replay_inputs[k]) for k in inputs),'standalone replay consumed different input bytes')
    actual_pf=load_fields(path,pf_layout)
    replay_pf=load_fields(lambda name:replay/'native'/name,pf_layout)
    def serial(mode):
        root=replay/mode
        layout=dict(v_layout)
        if (root/'silu.f32').exists():layout['silu.f32']=LR
        return load_fields(lambda name:root/name,layout)
    sa,sb=serial('serial-a'),serial('serial-b')
    actual=[load_probe(Path(f'{prefix}.stage0.r{r}.w1.stl0'),request_id=r) for r in (1,3)]
    embedding=np.frombuffer(Path(f'{prefix}.first.r2.embedding.f32').read_bytes(),dtype='<f4')
    mixed=np.frombuffer(Path(f'{prefix}.first.r2.hc_mixed.f32').read_bytes(),dtype='<f4')
    require(embedding.shape==(N,) and mixed.shape==(N,) and np.isfinite(embedding).all() and
            np.isfinite(mixed).all() and same(mixed,actual_pf['mixed.f32']), 'native first-token probes differ')
    require(same(inputs['residual.f32'],np.tile(embedding,HC)), 'input residual is not actual broadcast embedding')
    for p in actual:
        require(p['window']==1 and p['position']==0 and p['token']==marker['token_ids'][0] and
                p['n_embd']==N and p['plan_fingerprint']==runs['serial-a'][0][1]['fingerprint'] and
                same(np.asarray(p['embedding'],dtype='<f4'),embedding), 'actual serial identity/embedding mismatch')
    va,vb=[np.asarray(p['hc_mixed'],dtype='<f4') for p in actual]
    actual_serial=[]
    for request,mixed_output in zip((1,3),(va,vb)):
        fields=load_fields(lambda name:Path(f'{prefix}.hc.serial-r{request}.{name}'),
                           {'residual.f32':HC*N,'scale.f32':HC,'inject.f32':HC})
        require(same(fields.pop('residual.f32'),inputs['residual.f32']),
                'actual serial HC residual differs from native HC input')
        fields['mixed.f32']=mixed_output
        actual_serial.append(fields)
    actual_control=compare_fields(*actual_serial)
    verifier_cmp=[compare_fields({k:v[k] for k in v_layout},a) for v,a in zip((sa,sb),actual_serial)]
    common=compare_fields({k:actual_pf[k] for k in v_layout},actual_serial[0])
    controls=compare_fields(sa,sb)
    pf_cmp=compare_fields(replay_pf,actual_pf)
    covered_q8='NOT_WRITTEN_BY_FUSED_READ'
    if done['q8_written']:
        qa,qb=[(replay/mode/'mixed.q8_1').read_bytes() for mode in ('serial-a','serial-b')]
        require(len(qa)==len(qb)==N//32*36, 'invalid fused Q8 extent')
        covered_q8='IDENTICAL' if qa==qb==actual[0]['q8_input']==actual[1]['q8_input'] else 'DIFFERENT'
    else:
        require(not any((replay/mode/'mixed.q8_1').exists() for mode in ('serial-a','serial-b')),
                'unexpected Q8 publication')
    stable=all(v['bytes_equal'] for v in controls.values()) and all(v['bytes_equal'] for v in actual_control.values())
    stable=stable and actual[0]['q8_input']==actual[1]['q8_input']
    replay_exact=(stable and all(v['bytes_equal'] for v in pf_cmp.values()) and
                  all(v['bytes_equal'] for c in verifier_cmp for v in c.values()) and covered_q8!='DIFFERENT')
    return {'status':'REPRODUCED_NATIVE_HC_OPERATOR' if replay_exact and not same(va,mixed) else
                     'IDENTICAL_NATIVE_HC_OPERATOR' if replay_exact else 'NOT_REPRODUCED',
            'scope':'CUDA_ONLY_OPERATOR_REPLAY_NOT_FULL_VANILLA_CALLER',
            'first_token':marker['token_ids'][0], 'model_fingerprint':marker['model_fingerprint'],
            'actual_source_identity':{k:runs['serial-a'][0][0][k] for k in ('pack_index_sha256','gguf_shards','profile_path')},
            'input_sha256':{k:hashlib.sha256(a.tobytes()).hexdigest() for k,a in inputs.items()},
            'replay_device':done,'actual_serial_controls':actual_control,'replay_serial_controls':controls,
            'prompt_intermediates':pf_cmp,'fused_q8_image':covered_q8,
            'serial_replay_vs_actual':dict(zip(('a','b'),verifier_cmp)),
            'actual_prompt_vs_verifier_common_fields':common,
            'first_different_observed_common_field':next((k for k in v_layout if not common[k]['bytes_equal']),None),
            'numerics_actual':oracle(inputs,actual_pf,actual_serial[0]),
            'numerics_replay':oracle(inputs,replay_pf,sa),'production_admission':'NOT_QUALIFIED'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix',type=Path,required=True)
    p.add_argument('--capture-log',type=Path,required=True)
    p.add_argument('--replay-dir',type=Path,required=True)
    p.add_argument('--expected-pci',default='0000:04:00.0')
    args=p.parse_args()
    try:
        result=analyze(args.prefix,args.capture_log.read_text(),args.replay_dir,args.expected_pci)
        print(json.dumps(result,indent=2,allow_nan=False))
        return 1 if result['status']=='NOT_REPRODUCED' else 0
    except (OSError,ValueError,TypeError,KeyError,AttributeError,IndexError,OverflowError) as exc:
        print(f'INVALID: {exc}',file=sys.stderr);return 2


if __name__=='__main__':sys.exit(main())
