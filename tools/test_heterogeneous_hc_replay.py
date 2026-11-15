"""Synthetic small arithmetic fixtures; no claim of actual native GPU outputs."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
import heterogeneous_hc_replay as h


class HCReplay(unittest.TestCase):
    def test_bf16_rne(self):
        x=np.array([0x3f808000,0x3f818000,0xbf808000,0xbf818000,0x00000000],dtype='<u4').view('<f4')
        self.assertEqual(h.bf16_bits(x).tolist(),[0x3f80,0x3f82,0xbf80,0xbf82,0])
        self.assertEqual(h.decode_bf16(h.bf16_bits(x)).tolist(),[1,1.015625,-1,-1.015625,0])
        with self.assertRaises(ValueError):h.bf16_bits([float('inf')])

    def test_zero_gate_reference(self):
        n,hc,rank=2,2,1
        x=np.array([1.,1.,1.,1.],dtype='<f4')
        scale=np.full(hc,1/np.sqrt(1+h.EPS),dtype='<f4')
        inp={'residual.f32':x,'norm_weight.f32':x,'down_weight.bf16':h.bf16_bits([0]*4),
            'up_weight.bf16':h.bf16_bits([0]*4),'inject_weight.bf16':h.bf16_bits([0]*8)}
        prompt={'scale.f32':scale,'normalized.bf16':h.bf16_bits(x),'down.f32':np.zeros(1,dtype='<f4'),
                'silu.bf16':h.bf16_bits([0]),'up.f32':np.zeros(4,dtype='<f4'),
                'inject.f32':np.zeros(2,dtype='<f4'),'mixed.f32':scale/2}
        serial={k:prompt[k] for k in ('scale.f32','inject.f32','mixed.f32')}
        g=h.oracle(inp,prompt,serial,n,hc,rank)
        self.assertEqual(g['ideal_both_rounding_delta']['max_abs'],0)
        self.assertEqual(g['observed_native_delta']['max_abs'],0)
        self.assertLess(g['prompt_vs_full_fp64']['max_abs'],1e-7)
        self.assertEqual(g['local_verifier_errors']['silu'],'NOT_OBSERVED_IN_FUSED_KERNEL')
        serial['silu.f32']=np.zeros(1,dtype='<f4')
        self.assertEqual(h.oracle(inp,prompt,serial,n,hc,rank)['local_verifier_errors']['silu']['max_abs'],0)

    def test_two_rounding_boundaries(self):
        n,hc,rank=3,2,2
        rng=np.random.default_rng(5)
        x=rng.normal(size=(hc,n)).astype('<f4');w=rng.uniform(.3,1.3,size=(hc,n)).astype('<f4')
        wd=h.bf16_bits(rng.normal(size=(rank,hc*n)));wu=h.bf16_bits(rng.normal(size=(hc*n,rank)))
        wi=h.bf16_bits(rng.normal(size=(hc,hc*n)))
        xf=x.astype(np.float64);wf=w.astype(np.float64)
        sc=1/np.sqrt(np.mean(xf*xf,axis=1)+h.EPS);xn=(xf*sc[:,None]*wf).ravel()
        xb=h.bf16_bits(xn);d=h.decode_bf16(wd)@h.decode_bf16(xb)
        z=d/hc;lo=h.bf16_bits(z*h.sigmoid(z));up=h.decode_bf16(wu)@h.decode_bf16(lo)
        out=np.mean((xn*h.sigmoid(up)).reshape(hc,n),axis=0)
        d2=h.decode_bf16(wd)@xn;z2=d2/hc;lo2=z2*h.sigmoid(z2)
        up2=h.decode_bf16(wu)@lo2;out2=np.mean((xn*h.sigmoid(up2)).reshape(hc,n),axis=0)
        inp={'residual.f32':x.ravel(),'norm_weight.f32':w.ravel(),'down_weight.bf16':wd,'up_weight.bf16':wu,'inject_weight.bf16':wi}
        pf={'scale.f32':sc,'normalized.bf16':xb,'down.f32':d,'silu.bf16':lo,'up.f32':up,
            'inject.f32':h.decode_bf16(wi)@h.decode_bf16(xb),'mixed.f32':out}
        v={'scale.f32':sc,'inject.f32':h.decode_bf16(wi)@xn,'mixed.f32':out2,'silu.f32':lo2}
        g=h.oracle(inp,pf,v,n,hc,rank)
        self.assertGreater(g['ideal_normalized_rounding_delta']['max_abs'],0)
        self.assertGreater(g['ideal_silu_rounding_delta']['max_abs'],0)
        self.assertEqual(g['actual_minus_predicted_delta']['max_abs'],0)
        for phase in g['local_prompt_errors'].values():self.assertLess(phase['max_abs'],1e-12)
        self.assertLess(g['local_verifier_errors']['mixed_from_observed_silu']['max_abs'],1e-12)
        pf['up.f32']=up+.125
        self.assertGreater(h.oracle(inp,pf,v,n,hc,rank)['local_prompt_errors']['up']['max_abs'],.12)

    def test_exact_bytes_and_finite(self):
        a=np.array([0.],dtype='<f4');b=np.array([-0.],dtype='<f4')
        self.assertFalse(h.compare_fields({'x.f32':a},{'x.f32':b})['x.f32']['bytes_equal'])
        self.assertEqual(h.metric(a,b)['max_abs'],0)
        with self.assertRaises(ValueError):h.metric([float('nan')],[1])
        with self.assertRaises(ValueError):h.compare_fields({'x.f32':a},{})

    def test_shape_and_truncation(self):
        self.assertEqual(h.shapes()[0]['down_weight.bf16'],3276800)
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'x.bf16';p.write_bytes(h.bf16_bits([1,2]).tobytes())
            self.assertEqual(h.load_fields(lambda _:p,{'x.bf16':2})['x.bf16'].size,2)
            with self.assertRaises(ValueError):h.load_fields(lambda _:p,{'x.bf16':3})
            p.write_bytes(b'\x80\x7f\x00\x00')
            with self.assertRaises(ValueError):h.load_fields(lambda _:p,{'x.bf16':2})

    def test_dispatch_refusal(self):
        s='STRATA_GR0_PATH down_q8=0 up_q8=0 down_scale=0 up_scale=0 qfuse=1 hc_q8_env=(unset)'
        self.assertTrue(h.qfuse_from_log(s+'\n'+s))
        self.assertFalse(h.qfuse_from_log(s.replace('qfuse=1','qfuse=0')))
        for bad in ('',s.replace('down_q8=0','down_q8=1'), s+'\n'+s.replace('qfuse=1','qfuse=0')):
            with self.assertRaises(ValueError):h.qfuse_from_log(bad)


if __name__=='__main__':unittest.main()
