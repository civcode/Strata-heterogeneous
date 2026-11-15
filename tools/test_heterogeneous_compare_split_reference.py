import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from heterogeneous_compare_split_reference import compare, HIP, MIXED
from heterogeneous_compare_parity import SCHEMA
from test_heterogeneous_compare_parity import packet


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


class SplitReaderIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ref, self.candidate = self.root / 'ref', self.root / 'candidate'
        self.make_run(self.ref, 'vanilla')
        self.make_run(self.candidate, 'heterogeneous')

    def make_run(self, prefix, producer):
        plan_fp = 123
        Path(f'{prefix}.complete.json').write_text(json.dumps(dict(
            schema=SCHEMA, windows=1, plan=HIP, placement_capture=True, model_fingerprint=1)))
        meta = dict(schema=SCHEMA, plan=HIP, model_fingerprint=1, plan_fingerprint=plan_fp,
                    pack_index_sha256='d'*64, gguf_shards='shards', profile_path='profile',
                    window_id=1, pos0=0, rows=1, n_keep=1, token_ids=[1], kv_mode='int8',
                    capture_producer=producer, source_revision='a'*40, binary_sha256='b'*64)
        Path(f'{prefix}.w1.json').write_text(json.dumps(meta))
        Path(f'{prefix}.w1.stfr').write_bytes(packet(stage=1, plan_fp=plan_fp))

    def run_cli(self):
        return subprocess.run([sys.executable, str(Path(__file__).with_name('heterogeneous_compare_split_reference.py')),
                               '--vanilla-prefix', str(self.ref), '--candidate-prefix', str(self.candidate)],
                              capture_output=True, text=True)

    def test_end_to_end_and_bad_manifests(self):
        self.assertEqual(self.run_cli().returncode, 0)
        path = Path(f'{self.candidate}.w1.json')
        original = json.loads(path.read_text())
        for field, value in [('kv_mode', 'f16'), ('pos0', 1), ('source_revision', 'wrong'),
                             ('capture_producer', None), ('pack_index_sha256', 'e'*64)]:
            with self.subTest(field=field):
                meta = dict(original)
                if value is None:
                    del meta[field]
                else:
                    meta[field] = value
                path.write_text(json.dumps(meta))
                self.assertEqual(self.run_cli().returncode, 2)
        path.write_text(json.dumps(original))
        stfr = Path(f'{self.candidate}.w1.stfr')
        blob = bytearray(stfr.read_bytes())
        blob[-1] ^= 1
        stfr.write_bytes(blob)
        self.assertEqual(self.run_cli().returncode, 2)


if __name__ == "__main__":
    unittest.main()
