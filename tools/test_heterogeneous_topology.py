"""Synthetic forward/reverse serial comparison contract; no GPU execution."""
import unittest
from heterogeneous_compare_parity import compare

FORWARD = "cuda:0:0-12,hip:0:12-30,hip:1:30-48"
REVERSE = "hip:0:0-18,hip:1:18-36,cuda:0:36-48"

def records(plan, logits=(2., 1.)):
    return [({"plan": plan, "model_fingerprint": 1, "pack_index_sha256": "x",
              "gguf_shards": "g", "profile_path": "p", "window_id": 1,
              "pos0": 0, "rows": 1, "n_keep": 1, "token_ids": [0]},
             {"rows": 1, "vocab": 2, "logits": list(logits),
              "predictions": [int(logits[1] > logits[0])]})]

class TopologyTest(unittest.TestCase):
    def test_same_and_changed_predictions(self):
        self.assertEqual(compare(records(FORWARD), records(REVERSE), .001, topology=True)["status"], "PASS")
        self.assertEqual(compare(records(FORWARD), records(REVERSE, (1.,2.)), .001, topology=True)["status"], "FAIL")

    def test_refuses_wrong_plan_and_identity(self):
        for wrong in (FORWARD, "hip:1:0-18,hip:0:18-36,cuda:0:36-48"):
            with self.assertRaises(ValueError):
                compare(records(FORWARD), records(wrong), .001, topology=True)
        changed = records(REVERSE)
        changed[0][0]["token_ids"] = [1]
        with self.assertRaises(ValueError):
            compare(records(FORWARD), changed, .001, topology=True)

if __name__ == "__main__":
    unittest.main()
