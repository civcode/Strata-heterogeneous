#!/usr/bin/env python3
"""M2.13 CPU-only 2-run HC read precision-isolation contract tests."""
import json
from pathlib import Path
import tempfile
import unittest

import heterogeneous_hc_mode_compare as hc
import heterogeneous_layer0_arithmetic as l0
from test_heterogeneous_layer0_arithmetic import make_stl0
from test_heterogeneous_compare_parity import packet


class HCModeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.ref = self.root / "ref"
        self.mixed = self.root / "mixed"
        self.write(self.ref, hc.EXPECTED_REF)
        self.write(self.mixed, hc.EXPECTED_HET)

    def write(self, prefix, plan, *, mixed_delta=0.0, q8_byte=None,
              token=1, model="a"):
        Path(f"{prefix}.complete.json").write_text(json.dumps({
            "schema": parity_schema(), "windows": 1, "plan": plan,
            "model_fingerprint": 2026111301
        }))
        Path(f"{prefix}.w1.json").write_text(json.dumps({
            "schema": parity_schema(), "plan": plan,
            "model_fingerprint": 2026111301,
            "plan_fingerprint": 123, "pack_index_sha256": model * 64,
            "gguf_shards": "/model/a.gguf;/model/b.gguf",
            "profile_path": "/pack/profile.strp", "window_id": 1,
            "pos0": 0, "rows": 1, "n_keep": 1, "token_ids": [token]
        }))
        Path(f"{prefix}.w1.stfr").write_bytes(packet(stage=plan.count(",")))
        Path(f"{prefix}.stage0.w1.stl0").write_bytes(
            make_stl0(mixed_delta=mixed_delta, q8_byte=q8_byte))

    def compare(self):
        return hc.compare(l0.read_run(self.ref), l0.read_run(self.mixed),
                          "matched_bf16")

    def test_exact_match(self):
        result = self.compare()
        self.assertEqual(result["status"], "DIAGNOSTIC_ONLY")
        self.assertIsNone(result["rows"][0]["first_nonidentical_component"])

    def test_hc_mixed_first(self):
        self.write(self.mixed, hc.EXPECTED_HET, mixed_delta=0.1)
        result = self.compare()
        self.assertEqual(result["rows"][0]["first_nonidentical_component"],
                         "hc_mixed")
        self.assertGreater(
            result["rows"][0]["components"]["hc_mixed"]["relative_l2"], 0)

    def test_q8_quantizer_first(self):
        self.write(self.mixed, hc.EXPECTED_HET, q8_byte=1)
        result = self.compare()
        self.assertEqual(result["rows"][0]["first_nonidentical_component"],
                         "q8_input")

    def test_model_mismatch_is_rejected(self):
        self.write(self.mixed, hc.EXPECTED_HET, model="b")
        with self.assertRaisesRegex(ValueError, "pack_index_sha256"):
            self.compare()

    def test_invalid_plan_is_rejected(self):
        self.write(self.ref, hc.EXPECTED_HET)
        with self.assertRaisesRegex(ValueError, "full HIP reference"):
            self.compare()

    def test_missing_complete_marker_rejected(self):
        Path(f"{self.mixed}.complete.json").unlink()
        with self.assertRaisesRegex(ValueError, "completion marker"):
            self.compare()

    def test_corrupted_stl0_fails(self):
        path = Path(f"{self.mixed}.stage0.w1.stl0")
        data = bytearray(path.read_bytes())
        data[-5] ^= 0x08
        path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "STL0 checksum"):
            self.compare()

    def test_blank_arm_rejected(self):
        with self.assertRaisesRegex(ValueError, "arm must"):
            hc.compare(l0.read_run(self.ref), l0.read_run(self.mixed), "")


def parity_schema():
    return "strata-heterogeneous-parity/v1"


if __name__ == "__main__":
    unittest.main()
