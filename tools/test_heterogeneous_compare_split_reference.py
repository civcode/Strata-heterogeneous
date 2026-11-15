import unittest

from heterogeneous_compare_split_reference import compare, HIP, MIXED


def capture(producer, mode="int8", plan=HIP):
    return [(dict(plan=plan, capture_producer=producer, source_revision="a"*40,
                  binary_sha256=("b" if producer == "vanilla" else "c")*64, kv_mode=mode,
                  window_id=i+1, pos0=i, rows=1, n_keep=1, model_fingerprint=1,
                  pack_index_sha256="d"*64, gguf_shards="shards", profile_path="profile", token_ids=[i]),
             dict(rows=1, vocab=3, predictions=[2], logits=[0., 1., 2.])) for i in range(2)]


class SplitReferenceTests(unittest.TestCase):
    def test_matched_hip_and_mixed(self):
        for mode in ("f16", "int8"):
            for plan in (HIP, MIXED):
                self.assertEqual(compare(capture("vanilla", mode), capture("heterogeneous", mode, plan))["status"], "PASS")

    def test_numerical_failure_first_row(self):
        c = capture("heterogeneous")
        c[1][1]["logits"] = [-3., -1., 2.]
        result = compare(capture("vanilla"), c)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["first_failure"]["position"], 1)

    def test_reject_self_mode_identity_and_sequence(self):
        r = capture("vanilla")
        with self.assertRaises(ValueError):
            compare(r, r)
        with self.assertRaises(ValueError):
            compare(r, capture("heterogeneous", "f16"))
        c = capture("heterogeneous")
        c[1][0]["binary_sha256"] = "f"*64
        with self.assertRaises(ValueError):
            compare(r, c)
        c = capture("heterogeneous")
        c[1][0]["token_ids"] = [2]
        with self.assertRaises(ValueError):
            compare(r, c)

    def test_reject_relaxed_threshold_and_partial(self):
        r, c = capture("vanilla"), capture("heterogeneous")
        with self.assertRaises(ValueError):
            compare(r, c, 0.1)
        c[0][0]["n_keep"] = 0
        with self.assertRaises(ValueError):
            compare(r, c)


if __name__ == "__main__":
    unittest.main()
