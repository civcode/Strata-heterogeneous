import unittest
import numpy as np
from heterogeneous_statistical_quality import metrics
from heterogeneous_numerical_qualification import summarize
from heterogeneous_prefill_stage_gate import weight_identity

class TestNumericalQualification(unittest.TestCase):
    def test_metrics(self):
        a=metrics([0,0],[0,0],1)
        self.assertAlmostEqual(a["nll_ref"],np.log(2))
        self.assertEqual(a["kl"],0)
        self.assertTrue(a["top1_equal"])
        self.assertFalse(metrics([1,0],[0,1],0)["top1_equal"])
        with self.assertRaises(ValueError): metrics([0,0],[0,0],2)
    def test_weight_identity(self):
        self.assertIsNone(weight_identity({}))
        self.assertEqual(weight_identity({"hc_weight_contract":"bf16"}),"bf16")
        with self.assertRaises(Exception): weight_identity({"hc_weight_contract":"other"})
    def test_gate(self):
        c={"plan":"p","weight_contract":"bf16","activation_mode":"off","transport":"shared",
           "cut":2,"chunk":1,"interop":"PASS","quality":{"max_kl_ref_to_hetero":.002,
           "prediction_disagreements":0,"top1_disagreements":0},
           "matched_latency_ns":{"baseline":[1,2,3],"candidate":[2,4,6]}}
        d={"schema":"strata-numerical-qualification/v1","cases":[c]}
        r=summarize(d)
        self.assertEqual(r["cases"][0]["quality"],"FAIL")
        self.assertEqual(r["cases"][0]["latency"]["ratio"],2)
        self.assertEqual(r["production_admission"],"NOT_QUALIFIED")
        d["cases"].append(c)
        with self.assertRaises(ValueError): summarize(d)
if __name__=="__main__": unittest.main()
