import copy
import os
import sys
import tempfile
import types
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import heterogeneous_numerical_trace as t


class TraceTests(unittest.TestCase):
    def test_ties_and_weights(self):
        logits=np.array([2.,2.,1.,0.]);ids=np.array([0,1],dtype='<i4')
        obs=t.route_observation(logits,ids,np.array([.5,.5]))
        self.assertTrue(obs['set_matches']);self.assertEqual(obs['rank10_to_11_gap'],1.)
        self.assertEqual(obs['weights_error']['max_abs'],0.)
        with self.assertRaises(ValueError):t.route_observation(logits,np.array([1,1]),[.5,.5])

    def test_router_decomposition(self):
        w=np.eye(3);a={'format':'f32','operand':np.array([3.,2.,1.]),'ffn_mixed.f32':np.array([3.,2.,1.]),
            'router_logits.f32':np.array([3.,2.,1.]),'router_ids.i32':np.array([0]),'router_weights.f32':np.array([1.])}
        b=dict(a,format='bf16',operand=np.array([1.,3.,2.]),**{'ffn_mixed.f32':np.array([1.,3.,2.]),
            'router_logits.f32':np.array([1.,3.,2.]),'router_ids.i32':np.array([1])})
        ref=t.router_reference(w,a,b)
        self.assertEqual(ref['actual_changed_experts'],1);self.assertEqual(ref['unexplained_delta_vector']['max_abs'],0.)
        self.assertEqual(ref['representation_delta']['max_abs'],0.)
        b['router_logits.f32']=np.array([1.,3.1,2.]);self.assertGreater(t.router_reference(w,a,b)['unexplained_delta_vector']['max_abs'],.09)

    def test_residual_write(self):
        r={'r_ffn.f32':np.arange(12,dtype='<f4'),'ffn_inj.f32':np.zeros(4,dtype='<f4'),'ffn_bo.f32':np.array([1,2,3],dtype='<f4')}
        np.testing.assert_array_equal(t.completed_r(r),np.arange(12)+np.tile([1,2,3],4))

    def test_raw_and_bf16(self):
        np.testing.assert_array_equal(t.bf16([0x3f80,0xbf80]),[1.,-1.])
        with self.assertRaises(ValueError):t.bf16([0x7f80])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'x';p.write_bytes(np.array([1.,2.],dtype='<f4').tobytes());self.assertEqual(t.read(p,'<f4',2).tolist(),[1.,2.])
            with self.assertRaises(ValueError):t.read(p,'<f4',3)
            p.write_bytes(np.array([np.nan],dtype='<f4').tobytes())
            with self.assertRaises(ValueError):t.read(p,'<f4',1)
            q=Path(tmp)/'link';q.symlink_to(p)
            with self.assertRaises(ValueError):t.raw(q,4)

    def test_row_representation_and_missing_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'cap';stem=f'{root}.trace.r2.l0.p0.'
            for field,count in t.FIELDS.items():
                a=np.zeros(count,dtype='<i4' if field.endswith('i32') else '<f4')
                if field=='router_ids.i32':a[:]=np.arange(t.K)
                if field=='router_weights.f32':a[:]=1/t.K
                Path(stem+field).write_bytes(a.tobytes())
            Path(stem+'r_out.f32').write_bytes(bytes(4*t.N*t.HC))
            Path(stem+'router_operand.bf16').write_bytes(np.full(t.N,0x3f80,dtype='<u2').tobytes())
            Path(stem+'router_operand_lo.bf16').write_bytes(np.full(t.N,0x3b80,dtype='<u2').tobytes())
            row=t.row(root,2,0,0,2);self.assertEqual(row['format'],'bf16x2')
            self.assertEqual(row['operand'][0],1.00390625)
            Path(stem+'router_operand.f16').write_bytes(bytes(2*t.N))
            with self.assertRaisesRegex(ValueError,'representation'):t.row(root,2,0,0,2)
            Path(stem+'router_operand.f16').unlink();Path(stem+'r_ffn.f32').unlink()
            with self.assertRaisesRegex(ValueError,'extent'):t.row(root,2,0,0,2)

    def test_observer_inventory_and_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b=[Path(tmp)/x for x in ('a','b')]
            for p in (a,b):Path(str(p)+'.x').write_bytes(b'123')
            Path(str(a)+'.trace.extra').write_bytes(b'new')
            self.assertEqual(t.observer_check(a,b),1)
            Path(str(a)+'.x').write_bytes(b'124')
            with self.assertRaisesRegex(ValueError,'observer changed'):t.observer_check(a,b)
            Path(str(a)+'.x').unlink()
            with self.assertRaisesRegex(ValueError,'inventory'):t.observer_check(a,b)

    def test_all_layers_control_flow_with_synthetic_parser_records(self):
        # Real-sized tensor files and all 48 layers, but mocked prevalidated
        # stage-parser records. This is not an STFR parser or GPU integration test.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cap=root/'cap';base=root/'baseline';cut=2
            def seed(name,a):
                p=root/name;p.write_bytes(a.tobytes());return p
            blobs={key:seed('seed-'+key,np.arange(count,dtype='<i4') if key.endswith('i32') else
                            np.full(count,.1 if key=='router_weights.f32' else 0,dtype='<f4'))
                   for key,count in t.FIELDS.items()}
            blobs['router_operand.f32']=seed('seed-operand-f32',np.zeros(t.N,dtype='<f4'))
            blobs['router_operand.bf16']=seed('seed-operand-bf16',np.zeros(t.N,dtype='<u2'))
            blobs['r_out.f32']=blobs['r_attn.f32']
            weights=seed('seed-router-weight',np.zeros(t.N*t.NE,dtype='<u2'))
            altered=np.zeros(t.NE,dtype='<f4');altered[t.K]=1
            changed_logits=seed('seed-changed-logits',altered)
            changed_ids=seed('seed-changed-ids',np.array([t.K]+list(range(t.K-1)),dtype='<i4'))
            states={}
            def legacy(suffix,data):
                p=Path(str(cap)+suffix);p.write_bytes(data);os.link(p,Path(str(base)+suffix))
            for stage in range(3):
                for pos in range(cut):
                    for direction in ('in','out'):
                        legacy(f'.native.stage{stage}.p{pos}.{direction}.f32',bytes(t.N*t.HC*4))
                q=(0,3,7)[stage];layout=dict(ordinal=q,format=0,cells=16,page_size=16,heads=2,head_dim=256,pooled_rows=1,idx_dim=128)
                key=f'qsa{q}.k';state_fields=[('gdn0.conv','f32',4),(key,'opaque',4)]
                states[str(stage)]={'identity':{'qsa':[layout],'ple_prev':[-1,-1],'fields':state_fields}}
                for field,_,size in state_fields:legacy(f'.native-state.s{stage}.state.r2.p{cut}.{field}.bin',bytes(size))
                for r in (1,2,3):
                    for pos in range(cut+1):
                        Path(f'{cap}.trace.r{r}.s{stage}.p{pos}.token.i32').write_bytes(np.array([t.TOKENS[pos]],dtype='<i4').tobytes())
                    if r==2:continue
                    s=f'{cap}.trace.r{r}.s{stage}.p{cut}.state.'
                    Path(s+'ple_prev.i32').write_bytes(np.array([-1,-1],dtype='<i4').tobytes())
                    Path(s+f'qsa{q}.layout.i32').write_bytes(np.array([layout[k] for k in ('format','cells','page_size','heads','head_dim','pooled_rows','idx_dim')],dtype='<i4').tobytes())
                    for field,_,size in state_fields:Path(s+field+'.bin').write_bytes(bytes(size))
            for layer in range(48):
                for mode in ('serial','native'):os.link(weights,Path(f'{cap}.trace.{mode}.l{layer}.router.bf16'))
                for r in (1,2,3):
                    for pos in range(cut+1):
                        stem=f'{cap}.trace.r{r}.l{layer}.p{pos}.'
                        prompt=r==2 and pos<cut
                        fields=list(t.FIELDS)+(['router_operand.bf16','r_out.f32'] if prompt else ['router_operand.f32'])
                        for field in fields:
                            src=blobs[field]
                            if (r,layer,pos)==(2,5,1):
                                if field=='router_logits.f32':src=changed_logits
                                if field=='router_ids.i32':src=changed_ids
                            os.link(src,Path(stem+field))
            marker={'numerical_trace':True,'prefix_rows':cut,'chunk':1,'stage_reference':'','token_ids':list(t.TOKENS[:cut+1]),'transport':'shared'}
            rec={'marker':marker,'source':(1,2),'bridge':{'status':'PASS'},'local_links_exact':True,'serial_controls_exact':True,
                 'residuals':{},'states':states,'logits':{},'quality':{'status':'FAIL'}}
            old=copy.deepcopy(rec);old['marker'].pop('numerical_trace')
            stub=types.SimpleNamespace(read_capture=lambda p:rec if p==cap else old)
            with patch.dict(sys.modules,{'heterogeneous_prefill_stage_gate':stub}):
                out=t.analyze(cap,base)
                self.assertEqual(len(out['layers']),48*3)
                self.assertEqual(out['first_changed_route_layer_by_position'],{'1':5})
                self.assertEqual(out['first_differing_phase_by_position'],{'1':{'layer':5,'field':'router_logits.f32'}})
                self.assertEqual(out['production_admission'],'NOT_QUALIFIED');self.assertEqual(out['quality']['status'],'FAIL')
                path=Path(f'{cap}.trace.r3.l47.p2.attn_mixed.f32');path.unlink();path.write_bytes(np.ones(t.N,dtype='<f4').tobytes())
                with self.assertRaisesRegex(ValueError,'unstable serial'):t.analyze(cap,base)

    def test_exact_and_metric(self):
        a=np.array([0.],dtype='<f4');b=np.array([-0.],dtype='<f4')
        self.assertFalse(t.exact(a,b));self.assertEqual(t.metric(a,b)['max_abs'],0.)
        with self.assertRaises(ValueError):t.metric([1,2],[1])


if __name__=='__main__':unittest.main()
