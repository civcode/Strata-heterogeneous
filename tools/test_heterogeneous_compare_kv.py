import unittest

from heterogeneous_compare_kv import compare, metrics


def capture(mode):
    meta = dict(kv_mode=mode, plan="hip:0:0-24,hip:1:24-48", model_fingerprint=1,
                pack_index_sha256="a"*64, gguf_shards="shards", profile_path="profile",
                window_id=1, pos0=0, rows=2, n_keep=1, token_ids=[0, 1])
    packet = dict(rows=2, vocab=3, predictions=[2, 2], logits=[0., 1., 2.]*2)
    return [(meta, packet)]


class KVTests(unittest.TestCase):
    def test_identical(self):
        result = compare(capture("f16"), capture("int8"))
        self.assertEqual(result["status"], "DIAGNOSTIC_ONLY")
        self.assertEqual(result["accepted"]["max_kl"], 0.)
        self.assertEqual(result["discarded"]["rows"], 1)

    def test_discarded_divergence(self):
        q = capture("int8")
        q[0][1]["logits"][3:] = [3., 1., 0.]
        q[0][1]["predictions"][1] = 0
        result = compare(capture("f16"), q)
        self.assertIsNone(result["accepted"]["first_disagreement"])
        self.assertEqual(result["discarded"]["first_disagreement"]["position"], 1)
        self.assertGreater(result["discarded"]["max_kl"], 0.)

    def test_identity_and_modes(self):
        with self.assertRaisesRegex(ValueError, "explicit F16"):
            compare(capture("int8"), capture("f16"))
        q = capture("int8")
        q[0][0]["token_ids"] = [1, 0]
        with self.assertRaisesRegex(ValueError, "token_ids"):
            compare(capture("f16"), q)
        q = capture("int8")
        del q[0][0]["kv_mode"]
        with self.assertRaisesRegex(ValueError, "explicit F16"):
            compare(capture("f16"), q)

    def test_stable_kl_and_margin(self):
        row = metrics([1000., 1001.], [-1000., -999.])
        self.assertAlmostEqual(row["kl_f16_to_int8"], 0.)
        self.assertEqual(row["f16_margin"], 1.)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            metrics([0., float("nan")], [0., 1.])


if __name__ == "__main__":
    unittest.main()
