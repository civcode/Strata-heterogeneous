#!/usr/bin/env python3
"""CPU-only STB0/v1 model integrity and stage branch localization tests."""
from __future__ import annotations

import json
from array import array
from pathlib import Path
import struct
import tempfile
import unittest

import heterogeneous_stage_branch as stage
from test_heterogeneous_compare_parity import packet as make_stfr


def fixture(*, selected=None, change=None, delta=0.0, position=0, window=1,
            shape=(2560, 512, 10, 48, 4, 32)):
    n, ne, k, c, hv, zv = shape
    fields = dict(n_embd=n, n_expert=ne, k=k, conv_channels=c,
                  value_heads=hv, value_dim=zv)
    ids = list(range(k)) if selected is None else selected
    values = array("f")
    for index, length in enumerate(stage.lengths(fields)):
        values.extend((0.25 + 0.1 * index + 0.0001 * i for i in range(length)))
        if change == index:
            offset = len(values) - length
            values[offset] += delta
    blob = bytearray(stage.HEADER.pack(
        stage.MAGIC, 1, 0, n, ne, k, c, hv,
        1, window, position, 123, 0, zv))
    blob.extend(struct.pack("<i", 1))
    blob.extend(struct.pack(f"<{k}i", *ids))
    blob.extend(values.tobytes())
    digest = 2166136261
    for i, byte in enumerate(blob):
        if 64 <= i < 68:
            continue
        digest = ((digest ^ byte) * 16777619) & 0xffffffff
    struct.pack_into("<I", blob, 64, digest)
    return bytes(blob)


class BranchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.reference = self.folder / "ref"
        self.heterogeneous = self.folder / "het"
        self.write_run(self.reference, stage.REF_PLAN)
        self.write_run(self.heterogeneous, stage.HET_PLAN)

    def write_run(self, prefix, plan, *, selected=None, change=None,
                  delta=0.0, windows=1, pack="a", token=1):
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
            Path(f"{prefix}.stage0.w{i}.stb0").write_bytes(fixture(
                selected=selected, change=change, delta=delta,
                position=i - 1, window=i))

    def compare(self):
        return stage.compare(stage.read_run(self.reference),
                             stage.read_run(self.heterogeneous))

    def test_wire_header_is_72_bytes(self):
        self.assertEqual(stage.HEADER.size, 72)

    def test_identical_reference_and_heterogeneous(self):
        result = self.compare()
        self.assertEqual(result["status"], "DIAGNOSTIC_ONLY")
        self.assertEqual(len(result["positions"]), 1)
        self.assertTrue(result["positions"][0]["route_ids_match"])
        self.assertIsNone(result["positions"][0]["first_nonidentical_float_component"])

    def test_router_ids_diverge(self):
        selected = list(range(10))
        selected[0] = 20
        self.write_run(self.heterogeneous, stage.HET_PLAN, selected=selected)
        result = self.compare()["positions"][0]
        self.assertFalse(result["route_ids_match"])
        self.assertEqual(result["first_route_id_mismatch_slot"], 0)

    def test_router_logits_diverge(self):
        self.write_run(self.heterogeneous, stage.HET_PLAN, change=0, delta=0.15)
        result = self.compare()["positions"][0]
        self.assertEqual(result["first_nonidentical_float_component"],
                         "layer5_router_logits")

    def test_router_weights_diverge(self):
        self.write_run(self.heterogeneous, stage.HET_PLAN, change=1, delta=0.15)
        result = self.compare()["positions"][0]
        self.assertEqual(result["first_nonidentical_float_component"],
                         "layer5_route_weights")

    def test_shared_expert_diverges(self):
        self.write_run(self.heterogeneous, stage.HET_PLAN, change=2, delta=0.15)
        result = self.compare()["positions"][0]
        self.assertEqual(result["first_nonidentical_float_component"], "layer5_shared")

    def test_expert_parts_diverge(self):
        self.write_run(self.heterogeneous, stage.HET_PLAN, change=3, delta=0.15)
        result = self.compare()["positions"][0]
        self.assertEqual(result["first_nonidentical_float_component"], "layer5_expert_parts")
        self.assertGreater(result["expert_slots"][0]["max_abs"], 0)

    def test_moe_combine_diverges(self):
        self.write_run(self.heterogeneous, stage.HET_PLAN, change=4, delta=0.15)
        result = self.compare()["positions"][0]
        self.assertEqual(result["first_nonidentical_float_component"], "layer5_moe_bo")

    def test_gdn_qkv_diverges(self):
        self.write_run(self.heterogeneous, stage.HET_PLAN, change=5, delta=0.15)
        result = self.compare()["positions"][0]
        self.assertEqual(result["first_nonidentical_float_component"], "layer6_qkv")

    def test_gdn_conv_diverges(self):
        self.write_run(self.heterogeneous, stage.HET_PLAN, change=6, delta=0.15)
        result = self.compare()["positions"][0]
        self.assertEqual(result["first_nonidentical_float_component"], "layer6_conv")

    def test_gdn_gate_diverges(self):
        self.write_run(self.heterogeneous, stage.HET_PLAN, change=7, delta=0.15)
        result = self.compare()["positions"][0]
        self.assertEqual(result["first_nonidentical_float_component"], "layer6_gate")

    def test_gdn_beta_diverges(self):
        self.write_run(self.heterogeneous, stage.HET_PLAN, change=8, delta=0.15)
        result = self.compare()["positions"][0]
        self.assertEqual(result["first_nonidentical_float_component"], "layer6_beta")

    def test_gdn_state_output_diverges(self):
        self.write_run(self.heterogeneous, stage.HET_PLAN, change=10, delta=0.15)
        result = self.compare()["positions"][0]
        self.assertEqual(result["first_nonidentical_float_component"], "layer6_recurrent_y")

    def test_two_windows(self):
        self.write_run(self.reference, stage.REF_PLAN, windows=2)
        self.write_run(self.heterogeneous, stage.HET_PLAN, windows=2)
        self.assertEqual(len(self.compare()["positions"]), 2)

    def test_source_mismatch_rejected(self):
        self.write_run(self.heterogeneous, stage.HET_PLAN, pack="b")
        with self.assertRaisesRegex(ValueError, "pack_index_sha256"):
            self.compare()

    def test_checksum_tampering_rejected(self):
        path = Path(f"{self.heterogeneous}.stage0.w1.stb0")
        b = bytearray(path.read_bytes())
        b[-1] ^= 1
        path.write_bytes(b)
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.compare()

    def test_truncated_payload_rejected(self):
        path = Path(f"{self.heterogeneous}.stage0.w1.stb0")
        path.write_bytes(path.read_bytes()[:-4])
        with self.assertRaisesRegex(ValueError, "payload size"):
            self.compare()

    def test_missing_completion_marker_rejected(self):
        Path(f"{self.heterogeneous}.complete.json").unlink()
        with self.assertRaisesRegex(ValueError, "completion marker"):
            self.compare()

    def test_wrong_plan_rejected(self):
        self.write_run(self.reference, stage.HET_PLAN)
        with self.assertRaisesRegex(ValueError, "unsplit HIP reference"):
            self.compare()


if __name__ == "__main__":
    unittest.main()
