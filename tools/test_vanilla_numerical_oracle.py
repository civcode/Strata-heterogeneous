import unittest
import numpy as np
from vanilla_numerical_oracle import *
from vanilla_numerical_report import summarize

class OracleTests(unittest.TestCase):
    def test_rounding(self):
        x=np.array([1+2**-8,1+3*2**-8,-1-2**-8])
        np.testing.assert_array_equal(image(x,"bf16"),[1,1+2**-6,-1])
        np.testing.assert_array_equal(image(x,"bf16x2"),x)
        with self.assertRaises(ValueError): image([np.nan],"bf16")
    def test_hc(self):
        r=np.ones((2,3)); zero=np.zeros((1,6))
        a=hc_read(r,r,zero,zero.T,np.zeros((2,6)))
        np.testing.assert_allclose(a["mixed"],.5/np.sqrt(1+float(np.float32(1e-6))))
        np.testing.assert_array_equal(hc_write(r,np.ones(3),np.zeros(2)),r+1)
    def test_gdn(self):
        s,y=gdn_step(np.zeros((2,2)),[1,0],[1,0],[2,3],0,1)
        np.testing.assert_array_equal(s,[[2,3],[0,0]])
        np.testing.assert_allclose(y,np.array([2,3])/np.sqrt(2))
        with self.assertRaises(ValueError): gdn_step(s,[1,0],[1,0],[2,3],1,1)
    def test_conv_chunk(self):
        h=np.zeros((3,2)); x=np.arange(8).reshape(4,2); w=np.ones((4,2))
        y,end=causal_conv(h,x,w)
        a,m=causal_conv(h,x[:2],w); b,z=causal_conv(m,x[2:],w)
        np.testing.assert_allclose(y,np.concatenate((a,b)))
        np.testing.assert_array_equal(end,z)
    def test_attention(self):
        np.testing.assert_allclose(selected_attention([0,0],[[1,0],[0,1]],[[2],[4]],1),[3])
    def test_router(self):
        ids,weights=route([1,1,0],1)
        np.testing.assert_array_equal(ids,[0]); np.testing.assert_allclose(weights,[1])
        a=router_sensitivity(np.eye(3),[3,2,1],[3,2,1],1)
        self.assertTrue(a["sufficient_set_stability"])
        b=router_sensitivity(np.eye(3),[3,2,1],[1,4,2],1)
        self.assertEqual(b["total_swaps"],1); self.assertFalse(b["sufficient_set_stability"])
    def test_cost(self):
        c=hc_cost()
        self.assertEqual(c["macs_per_read"],6594560)
        self.assertEqual(c["bf16x2_projection_flops"],2*c["projection_flops"])
        self.assertEqual(c["extra_low_image_bytes_per_chunk"],21120)
    def test_incomplete_report(self):
        with self.assertRaises(ValueError):
            summarize({"status":"DIAGNOSTIC_ONLY","baseline_exact":True,"case":[2,1],
                       "transport":"shared","layers":[]},"forward")
if __name__=="__main__": unittest.main()
