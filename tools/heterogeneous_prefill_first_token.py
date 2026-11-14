#!/usr/bin/env python3
"""Locate observed first-token CUDA differences, without inferring causality."""
import argparse
from array import array
import json
import math
from pathlib import Path
import sys

STATES = ('recurrence', 'conv', 'ple')
OPS = ('embedding', 'hc_mixed', 'gdn_qkv')
FIELDS = dict(recurrence='gdn0.recurrence', conv='gdn0.conv', ple='ple.history')
CAP = 16 << 20


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def read_f32(path, size):
    require(0 < size <= CAP and size % 4 == 0 and path.stat().st_size == size,
            f'invalid FP32 sample extent: {path}')
    data = path.read_bytes()
    floats(data)
    return data


def floats(raw):
    a = array('f')
    a.frombytes(raw)
    if sys.byteorder != 'little':
        a.byteswap()
    require(bool(a) and all(math.isfinite(v) for v in a), 'nonfinite or empty FP32 sample')
    return a


def metric(a, b):
    require(len(a) == len(b) and 0 < len(a) <= CAP and len(a) % 4 == 0, 'sample shape mismatch')
    x, y = floats(a), floats(b)
    return dict(bytes_equal=a == b, max_abs=max(abs(v-w) for v,w in zip(x,y)),
                first_different_byte=next((i for i,(v,w) in enumerate(zip(a,b)) if v!=w),None))


def group(a, b, names):
    return {name: metric(a[name], b[name]) for name in names}


def equal(metrics):
    return all(v['bytes_equal'] for v in metrics.values())


def classify(reset, init, serial_a, serial_b, native, state_a, state_b, before_bridge, drained, adopted):
    """Inputs all refer to the same first token and owning CUDA stage."""
    require(len(reset) == 3, 'three reset observations required')
    reset_controls = [group(reset[0], r, STATES) for r in reset[1:]]
    zero = all(not any(r[k]) for r in reset for k in STATES)
    init_delta = group(reset[1], init, STATES)
    controls = group(serial_a, serial_b, OPS)
    control_state = group(state_a, state_b, STATES)
    ops = group(serial_a, native, OPS)
    pre = group(state_a, before_bridge, STATES)
    drain = group(before_bridge, drained, STATES)
    adopt = group(drained, adopted, STATES)
    first = next((k for k in OPS if not ops[k]['bytes_equal']), None)
    if not zero or not all(equal(d) for d in reset_controls):
        conclusion = 'RESET_STATE_DIFFERENCE'
    elif not equal(init_delta):
        conclusion = 'NATIVE_INITIALIZATION_STATE_DIFFERENCE'
    elif not equal(controls) or not equal(control_state):
        conclusion = 'UNSTABLE_SERIAL_CONTROL'
    elif first:
        conclusion = 'CUDA_LOCAL_OPERATION_DIFFERENCE_BEFORE_BRIDGE'
    elif not equal(pre):
        conclusion = 'CUDA_LOCAL_STATE_DIFFERENCE_BEFORE_BRIDGE'
    elif not equal(drain):
        conclusion = 'STATE_CHANGED_BETWEEN_BRIDGE_AND_DRAIN'
    elif not equal(adopt):
        conclusion = 'STATE_CHANGED_BETWEEN_DRAIN_AND_ADOPTION_OBSERVATION'
    else:
        conclusion = 'OBSERVED_ENTRY_FIELDS_IDENTICAL'
    return dict(status='DIAGNOSTIC_ONLY', conclusion=conclusion, first_nonidentical_common_operation=first,
                reset_zero=zero, reset_controls=reset_controls, native_init_delta=init_delta,
                serial_control_operations=controls, serial_control_state=control_state,
                serial_vs_native_operations=ops, serial_vs_before_bridge_state=pre,
                before_bridge_vs_drained=drain, drained_vs_adopted=adopt,
                root_cause='NOT ESTABLISHED', production_admission='NOT QUALIFIED')


def analyze(prefix):
    from heterogeneous_prefill_parity import load as load_parity, MAX_KL
    from heterogeneous_prefill_state import load as load_state
    from heterogeneous_compare_parity import compare
    from heterogeneous_layer0_arithmetic import load_probe
    marker, runs, bridge = load_parity(prefix)
    require(marker.get('first_token_capture') is True and marker.get('state_capture') is True and
            marker['prefix_rows']==marker['chunk']==1 and len(marker['token_ids'])==2 and
            marker['plan']=='cuda:0:0-12,hip:0:12-30,hip:1:30-48', 'not a completed first-token diagnostic')
    state_marker, states, expected = load_state(prefix)
    require(state_marker == marker, 'state and target capture identity mismatch')
    probes = [load_probe(Path(f'{prefix}.stage0.r{request}.w1.stl0'), request_id=request) for request in (1,3)]
    fingerprint = runs['serial-a'][0][1]['fingerprint']
    for probe in probes:
        require(probe['window']==1 and probe['position']==0 and probe['token']==marker['token_ids'][0] and
                probe['plan_fingerprint']==fingerprint and probe['n_embd']==marker['n_embd'], 'wrong serial probe identity')
    serials = [{k: p[k].tobytes() for k in OPS} for p in probes]
    if sys.byteorder != 'little':
        for result, probe in zip(serials,probes):
            for k in OPS:
                a=array('f',probe[k]);a.byteswap();result[k]=a.tobytes()
    native = {k: read_f32(Path(f'{prefix}.first.r2.{k}.f32'),len(serials[0][k])) for k in OPS}
    sizes = {k: states['serial-a'][1][f][1] for k,f in FIELDS.items()}
    def phase(request, name):
        return {k: read_f32(Path(f'{prefix}.first.r{request}.{name}.{k}.f32'),sizes[k]) for k in STATES}
    def committed(mode):
        return {k: read_f32(states[mode][1][f][2],sizes[k]) for k,f in FIELDS.items()}
    result=classify([phase(r,'reset') for r in (1,2,3)],phase(2,'after_init'),*serials,native,
                    committed('serial-a'),committed('serial-b'),phase(2,'before_bridge'),
                    phase(2,'after_drain'),committed('native'))
    result.update(bridge=bridge, ple_prev_expected=expected,
                  ple_prev_observed={mode: row[0] for mode,row in states.items()},
                  serial_q8_input_identical=probes[0]['q8_input']==probes[1]['q8_input'],
                  logits={f'{a}_vs_{b}': compare(runs[a],runs[b],MAX_KL,prefill=True)
                          for a,b in (('serial-a','serial-b'),('serial-a','native'),('serial-b','native'))},
                  scope='FIRST_TOKEN_CUDA_OBSERVATIONS_ONLY')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix',type=Path,required=True)
    a=p.parse_args()
    try:
        result=analyze(a.prefix)
        print(json.dumps(result,indent=2,allow_nan=False))
        return 0  # Completed diagnostic is not a numerical admission PASS.
    except (OSError,ValueError,TypeError,KeyError,AttributeError,IndexError,OverflowError) as e:
        print(f'INVALID: {e}',file=sys.stderr);return 2


if __name__=='__main__':sys.exit(main())
