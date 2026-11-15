"""CPU fixture tests; synthetic bytes, not inference results."""
import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
import heterogeneous_compare_parity as base
import heterogeneous_prefill_parity as pf
import heterogeneous_prefill_stage_gate as gate


def doc(path, value):
    path.write_text(json.dumps(value))


def state(stem, stage, cut=1):
    lo,hi=gate.RANGES[stage];q0,q1=lo//4,hi//4;g0,g1=lo-q0,hi-q1
    fields=[('ple.history','f32',9*10240*4)]
    for j in range(g0,g1):
        fields += [(f'gdn{j}.recurrence','f32',128*48*128*4),(f'gdn{j}.conv','f32',10240*3*4)]
    layouts=[]
    for j in range(q0,q1):
        layouts.append(dict(ordinal=j,format=0,cells=16,page_size=16,pooled_rows=cut//4+1,heads=2,head_dim=256,idx_dim=128))
        fields += [(f'qsa{j}.{n}',k,s) for n,k,s in [('tail','f32',1536),('dead','f32',512),('block_pos','i32',4),
                   ('k','opaque',cut*1024),('v','opaque',cut*1024),('k_scale','opaque',0),('v_scale','opaque',0),('pooled','f32',(cut//4+1)*512)]]
    doc(Path(stem+'.json'),dict(schema='strata-heterogeneous-prefix-state/v1',stage=stage,request=2,position=cut,
        layer_lo=lo,layer_hi=hi,gdn_ord0=g0,gdn_alloc=g1-g0,qsa_ord0=q0,qsa_alloc=q1-q0,
        ple_prev=[-1,248045] if stage==0 else [-1,-1],qsa=layouts,fields=[dict(name=n,type=k,bytes=s) for n,k,s in fields]))
    for name,_,size in fields:
        with Path(stem+'.'+name+'.bin').open('wb') as f:f.truncate(size)


def record(prefix, reference=''):
    doc(Path(f'{prefix}.prefill.json'),dict(schema=pf.PREFILL_SCHEMA,plan=gate.PLAN,model_fingerprint=9,
        prefix_rows=1,chunk=1,hc=4,n_embd=2560,token_ids=list(gate.A[:2]),state_capture=True,
        first_token_capture=False,hc_replay_capture=False,stage_capture=True,stage_reference=reference,transport='shared'))
    for request,mode in enumerate(gate.MODES,1):
        stem=f'{prefix}.{mode}.w2'
        meta=dict(schema=base.SCHEMA,plan=gate.PLAN,model_fingerprint=9,plan_fingerprint=123,pack_index_sha256=gate.INDEX,
                  gguf_shards='/model/a;/model/b',profile_path='/profile',window_id=2,rows=1,n_keep=1,pos0=1,token_ids=[846])
        doc(Path(stem+'.json'),meta)
        logits=np.zeros(248320,dtype='<f4');logits[-1]=5
        if mode=='native':logits[:100000]=.4
        payload=struct.pack('<iI',248319,0)+logits.tobytes()
        packet=bytearray(base.HEADER.pack(base.MAGIC,1,1,2,1,248320,0,request,2,123,len(payload),0,0)+payload)
        h=2166136261
        for i,b in enumerate(packet):
            if not 52<=i<56:h=((h^b)*16777619)&0xffffffff
        struct.pack_into('<I',packet,52,h);Path(stem+'.stfr').write_bytes(packet)
    for stage in range(3):
        for direction in ('in','out'):
            Path(f'{prefix}.native.stage{stage}.p0.{direction}.f32').write_bytes(bytes(gate.ROW_BYTES))
        state(f'{prefix}.native-state.s{stage}.state.r2.p1',stage)
    for stage,direction in ((0,'out'),(1,'in')):Path(f'{prefix}.stage{stage}.r2.w1.{direction}.f32').write_bytes(bytes(gate.ROW_BYTES))
    state(f'{prefix}.state.r2.p1',0)


class StageGate(unittest.TestCase):
    def test_raw_pair_quality_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            w=Path(tmp)/'wire';r=Path(tmp)/'ref';record(w);record(r,str(w))
            result=gate.pair(w,r)
            self.assertEqual(result['interop'],'PASS');self.assertEqual(result['quality'],'FAIL')
            for run in ('wire_evidence','reference_evidence'):
                self.assertGreater(result[run]['quality']['serial-a_vs_native']['max_kl_ref_to_hetero'],.001)
            m=gate.matrix([result]);self.assertEqual(m['interoperability'],'INCOMPLETE');self.assertEqual(m['numerical_quality'],'FAIL')
            self.assertEqual(m['production_admission'],'NOT_QUALIFIED')
            path=Path(f'{r}.native.stage2.p0.out.f32');path.write_bytes(struct.pack('<f',1)+bytes(gate.ROW_BYTES-4))
            # Reopen actual raw records: the quality result is still independent.
            changed=gate.pair(w,r);self.assertEqual(changed['interop'],'FAIL');self.assertEqual(changed['quality'],'FAIL')
            self.assertEqual(changed['residual_differences'],['stage2.p0.out'])

    def test_reject_bad_state_extent_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as tmp:
            stem=str(Path(tmp)/'state');state(stem,1)
            self.assertEqual(len(gate.read_state(stem,1,1)['sha256']),61)
            p=Path(stem+'.gdn9.recurrence.bin');p.write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError,'extent'):gate.read_state(stem,1,1)
            path=Path(tmp)/'nan';path.write_bytes(struct.pack('<f',float('nan')))
            with self.assertRaisesRegex(ValueError,'nonfinite'):gate.signature(path,4,'f32')

    def test_hc_experiment_identity(self):
        self.assertIsNone(gate.experiment_identity({}))
        valid={'mode':'fp32','backend':'hip'}
        self.assertEqual(gate.experiment_identity({'hc_experiment':valid}),valid)
        for invalid in (None, {}, {'mode':'other','backend':'hip'}, {'mode':'fp32','backend':'other'}):
            with self.assertRaisesRegex(ValueError,'experiment identity'):
                gate.experiment_identity({'hc_experiment':invalid})
        c1,c2=list(gate.CASES)[:2]
        p={'case':c1,'transport':'shared','interop':'PASS','quality':'PASS',
           'wire_evidence':{'source':(1,2),'residuals':{},'states':{},'logits':{},'marker':{}}}
        q=copy.deepcopy(p);q['case']=c2;q['wire_evidence']['marker']['hc_experiment']=valid
        with self.assertRaisesRegex(ValueError,'cannot mix HC experiment'):
            gate.matrix([p,q])

    def test_matrix_requires_all_cases_and_two_modes(self):
        p={'case':next(iter(gate.CASES)),'transport':'shared','interop':'PASS','quality':'PASS',
           'wire_evidence':{'source':(1,2),'residuals':{},'states':{},'logits':{}}}
        pairs=[]
        for case in gate.CASES:
            for transport in ('shared','socket'):
                pairs.append(dict(copy.deepcopy(p),case=case,transport=transport))
        good=gate.matrix(pairs);self.assertEqual(good['status'],'BOUNDED_MATRIX_PASS')
        self.assertEqual(good['production_admission'],'NOT_QUALIFIED')
        self.assertEqual(gate.matrix(pairs[:-1])['coverage'],'INCOMPLETE')
        with self.assertRaisesRegex(ValueError,'duplicate'):gate.matrix(pairs+[pairs[0]])
        pairs[0]['quality']='FAIL';m=gate.matrix(pairs)
        self.assertEqual(m['interoperability'],'PASS');self.assertEqual(m['numerical_quality'],'FAIL')
        pairs[0]['wire_evidence']['residuals']['changed']='x';m=gate.matrix(pairs)
        self.assertEqual(m['interoperability'],'FAIL')


if __name__=='__main__':unittest.main()
