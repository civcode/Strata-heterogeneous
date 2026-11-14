#!/usr/bin/env python3
"""M2.17 CPU-only integrity, cross-record and native microtrace fixtures."""
from __future__ import annotations

from array import array
import json
from pathlib import Path
import struct
import tempfile
import unittest

import heterogeneous_stage_micro as micro
from test_heterogeneous_stage_branch import fixture as make_stb0
from test_heterogeneous_compare_parity import packet as make_stfr


def make_stm0(*, change=None, delta=0.0, change_q8=None,
              position=0, window=1, n=2560, ff=128, c=48, d=4):
    sizes = micro.field_counts(n, ff, c, d)
    data = array("f")
    for field, size in enumerate(sizes):
        if field in (5, 6, 8):
            old_field = {5: 2, 6: 5, 8: 6}[field]
            v = 0.25 + old_field * 0.1
        else:
            v = 0.15 + field * 0.1
        start = len(data)
        for i in range(size):
            data.append(v + 0.0001 * i)
        # Match the legacy STB0 fixture's FP32 rounding order: first
        # round into array('f'), then apply the deliberate perturbation.
        if change == field:
            data[start] += delta
    q8in = bytearray((i * 7) % 256 for i in range(n // 32 * 36))
    q8sw = bytearray((i * 11) % 256 for i in range(ff // 32 * 36))
    if change_q8 == "shared_input_q8":
        q8in[0] ^= 0x10
    if change_q8 == "swiglu_q8":
        q8sw[0] ^= 0x20
    result = bytearray(micro.HEADER.pack(
        micro.MAGIC, 1, 0, n, ff, c, d, 48,
        1, window, position, 123, 0, 0))
    result.extend(struct.pack("<i", 1))
    result.extend(data.tobytes())
    result.extend(q8in)
    result.extend(q8sw)
    h = 2166136261
    for i, b in enumerate(result):
        if i >= 64 and i < 68:
            continue
        h = ((h ^ b) * 16777619) & 0xffffffff
    struct.pack_into("<I", result, 64, h)
    return bytes(result)


class MicroTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)
        self.ref = self.dir / "ref"
        self.het = self.dir / "het"
        self.write(self.ref, micro.REF_PLAN)
        self.write(self.het, micro.HET_PLAN)

    def write(self, prefix, plan, *, change=None, delta=0.0,
              change_q8=None, pack="a", token=1, windows=1,
              overlap_change=None):
        Path(f"{prefix}.complete.json").write_text(json.dumps({
            "schema": "strata-heterogeneous-parity/v1", "windows": windows,
            "plan": plan, "model_fingerprint": 2026111301}))
        for i in range(1, windows + 1):
            meta = {"schema": "strata-heterogeneous-parity/v1",
                    "plan": plan, "model_fingerprint": 2026111301,
                    "plan_fingerprint": 123, "pack_index_sha256": pack * 64,
                    "gguf_shards": "/model/a.gguf;/model/b.gguf",
                    "profile_path": "/pack/profile.strp",
                    "window_id": i, "pos0": i - 1, "rows": 1,
                    "n_keep": 1, "token_ids": [token]}
            Path(f"{prefix}.w{i}.json").write_text(json.dumps(meta))
            Path(f"{prefix}.w{i}.stfr").write_bytes(
                make_stfr(stage=plan.count(","), window=i))
            old_field = overlap_change
            if old_field is None and change in (5, 6, 8):
                old_field = {5: 2, 6: 5, 8: 6}[change]
            Path(f"{prefix}.stage0.w{i}.stb0").write_bytes(make_stb0(
                change=old_field, delta=delta if old_field is not None else 0,
                window=i, position=i-1))
            Path(f"{prefix}.stage0.w{i}.stm0").write_bytes(make_stm0(
                change=change, delta=delta, change_q8=change_q8,
                position=i - 1, window=i))

    def compare(self):
        return micro.compare(micro.read_run(self.ref),
                             micro.read_run(self.het))

    def test_header_size(self):
        self.assertEqual(micro.HEADER.size, 72)

    def test_exact_capture(self):
        r = self.compare()
        self.assertEqual(r["status"], "DIAGNOSTIC_ONLY")
        self.assertIsNone(r["positions"][0]["first_nonidentical"])
        self.assertEqual(len(r["positions"][0]["components"]), 11)

    def test_shared_input_first(self):
        self.write(self.het, micro.HET_PLAN, change=0, delta=0.1)
        self.assertEqual(self.compare()["positions"][0]["first_nonidentical"],
                         "shared_input")

    def test_q8_input_first(self):
        self.write(self.het, micro.HET_PLAN, change_q8="shared_input_q8")
        r = self.compare()["positions"][0]
        self.assertEqual(r["first_nonidentical"], "shared_input_q8")
        self.assertEqual(r["components"]["shared_input_q8"]["mismatched_bytes"], 1)

    def test_gate_raw_first(self):
        self.write(self.het, micro.HET_PLAN, change=1, delta=0.2)
        self.assertEqual(self.compare()["positions"][0]["first_nonidentical"],
                         "gate_raw")

    def test_up_raw_first(self):
        self.write(self.het, micro.HET_PLAN, change=2, delta=0.2)
        self.assertEqual(self.compare()["positions"][0]["first_nonidentical"],
                         "up_raw")

    def test_swiglu_q8_first(self):
        self.write(self.het, micro.HET_PLAN, change_q8="swiglu_q8")
        self.assertEqual(self.compare()["positions"][0]["first_nonidentical"],
                         "swiglu_q8")

    def test_down_unscaled_first(self):
        self.write(self.het, micro.HET_PLAN, change=3, delta=0.1)
        self.assertEqual(self.compare()["positions"][0]["first_nonidentical"],
                         "down_unscaled")

    def test_gate_logit_first(self):
        self.write(self.het, micro.HET_PLAN, change=4, delta=0.25)
        self.assertEqual(self.compare()["positions"][0]["first_nonidentical"],
                         "gate_logit")

    def test_shared_final_with_old_overlap(self):
        self.write(self.het, micro.HET_PLAN, change=5, delta=0.3)
        self.assertEqual(self.compare()["positions"][0]["first_nonidentical"],
                         "shared_final")

    def test_gdn_qkv_with_old_overlap(self):
        self.write(self.het, micro.HET_PLAN, change=6, delta=0.3)
        self.assertEqual(self.compare()["positions"][0]["first_nonidentical"],
                         "layer6_qkv")

    def test_preconv_history_first(self):
        self.write(self.het, micro.HET_PLAN, change=7, delta=0.3)
        r = self.compare()["positions"][0]
        self.assertEqual(r["first_nonidentical"], "layer6_conv_history")
        self.assertGreater(r["components"]["layer6_conv_history"]["relative_l2"], 0)

    def test_gdn_conv_output_with_old_overlap(self):
        self.write(self.het, micro.HET_PLAN, change=8, delta=0.3)
        self.assertEqual(self.compare()["positions"][0]["first_nonidentical"],
                         "layer6_conv_output")

    def test_old_overlap_inconsistent_fails(self):
        self.write(self.het, micro.HET_PLAN, change=5, delta=0.3,
                   overlap_change=6)
        with self.assertRaisesRegex(ValueError, "overlap mismatch"):
            micro.read_run(self.het)

    def test_checksum_corruption_fails(self):
        p = Path(f"{self.het}.stage0.w1.stm0")
        b = bytearray(p.read_bytes())
        b[-1] ^= 2
        p.write_bytes(b)
        with self.assertRaisesRegex(ValueError, "checksum"):
            micro.read_run(self.het)

    def test_truncated_payload_fails(self):
        p = Path(f"{self.het}.stage0.w1.stm0")
        p.write_bytes(p.read_bytes()[:-4])
        with self.assertRaisesRegex(ValueError, "payload length"):
            micro.read_run(self.het)

    def test_missing_completion_fails(self):
        Path(f"{self.het}.complete.json").unlink()
        with self.assertRaisesRegex(ValueError, "completion marker"):
            micro.read_run(self.het)

    def test_model_source_mismatch_fails(self):
        self.write(self.het, micro.HET_PLAN, pack="b")
        with self.assertRaisesRegex(ValueError, "pack_index_sha256"):
            self.compare()

    def test_two_windows(self):
        self.write(self.ref, micro.REF_PLAN, windows=2)
        self.write(self.het, micro.HET_PLAN, windows=2)
        self.assertEqual(len(self.compare()["positions"]), 2)

    def test_swapped_plan_rejected(self):
        with self.assertRaisesRegex(ValueError, "full HIP reference"):
            micro.compare(micro.read_run(self.het), micro.read_run(self.ref))

    def test_invalid_threshold_fails(self):
        with self.assertRaisesRegex(ValueError, "threshold"):
            micro.compare(micro.read_run(self.ref), micro.read_run(self.het), -1)


if __name__ == "__main__":
    unittest.main()
