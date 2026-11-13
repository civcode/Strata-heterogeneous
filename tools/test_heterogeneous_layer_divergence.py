#!/usr/bin/env python3
"""CPU-only STLT/v1 integrity and post-layer divergence fixtures."""
from __future__ import annotations

import json
import math
from pathlib import Path
import struct
import tempfile
import unittest

import heterogeneous_layer_divergence as layers


def stlt(*, stage=0, first=0, count=47, width=4,
         total=48, rows=1, window=1, pos0=0, plan_fp=123,
         perturb_layer=None, perturb=0.0):
    payload = bytearray(layers.HEADER.pack(
        layers.MAGIC, layers.VERSION, stage, first, count,
        width, rows, total, 1, window, pos0, plan_fp, 0, 0))
    payload.extend(struct.pack("<" + "i" * rows, *([1] * rows)))
    for row in range(rows):
        for l in range(first, first + count):
            for column in range(width):
                value = 1.0 + l * 0.01 + column * 0.001 + row * 0.1
                if l == perturb_layer:
                    value += perturb
                payload.extend(struct.pack("<f", value))
    h = 2166136261
    for i, b in enumerate(payload):
        if 64 <= i < 68:
            continue
        h = ((h ^ b) * 16777619) & 0xffffffff
    struct.pack_into("<I", payload, 64, h)
    return bytes(payload)


def record(raw, where):
    where.write_bytes(raw)
    return layers.decode_tap(where)


def runs(root, perturb_layer=None, perturb=0.0):
    r = record(stlt(), root / "reference.stlt")
    h = [record(stlt(stage=0, first=0, count=12,
                     perturb_layer=perturb_layer, perturb=perturb), root / "h0.stlt"),
         record(stlt(stage=1, first=12, count=18,
                     perturb_layer=perturb_layer, perturb=perturb), root / "h1.stlt"),
         record(stlt(stage=2, first=30, count=17,
                     perturb_layer=perturb_layer, perturb=perturb), root / "h2.stlt")]
    common = {"model_fingerprint": 2026111301, "pack_index_sha256": "a" * 64,
              "gguf_shards": "/m/a.gguf;/m/b.gguf", "profile_path": "/p/strp",
              "pos0": 0, "rows": 1, "token_ids": [1], "n_keep": 1,
              "window_id": 1}
    a = {**common, "plan": "hip:0:0-48"}
    b = {**common, "plan": "cuda:0:0-12,hip:0:12-30,hip:1:30-48"}
    stfr = {"vocab": 248320}
    return [(a, stfr, [r])], [(b, stfr, h)]


class LayerTapContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_wire_struct_exactly_72_bytes(self):
        self.assertEqual(layers.HEADER.size, 72)

    def test_full_record_checksum_and_shape(self):
        r = record(stlt(), self.root / "ok.stlt")
        self.assertEqual((r["first"], r["count"], r["width"]), (0, 47, 4))
        self.assertEqual(len(r["samples"]), 47 * 4)
        self.assertEqual(r["tokens"], (1,))

    def test_checksum_tampering_rejected(self):
        raw = bytearray(stlt())
        raw[-1] ^= 7
        with self.assertRaisesRegex(ValueError, "checksum"):
            record(raw, self.root / "bad.stlt")

    def test_nonfinite_tap_rejected(self):
        raw = stlt(perturb_layer=5, perturb=float("nan"))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            record(raw, self.root / "nan.stlt")

    def test_exact_layer_agreement(self):
        ref, het = runs(self.root)
        r = layers.compare(ref, het, 0.001)
        self.assertEqual(r["status"], "DIAGNOSTIC_ONLY")
        self.assertIsNone(r["rows"][0]["first_layer_above_diagnostic_threshold"])
        self.assertEqual(len(r["layer_metrics"]), 47)

    def test_first_difference_in_cuda_stage(self):
        ref, het = runs(self.root, perturb_layer=0, perturb=0.1)
        r = layers.compare(ref, het, 0.001)
        self.assertEqual(r["rows"][0]["first_layer_above_diagnostic_threshold"], 0)

    def test_first_difference_at_cuda_to_hip_boundary(self):
        ref, het = runs(self.root, perturb_layer=12, perturb=0.1)
        r = layers.compare(ref, het, 0.001)
        self.assertEqual(r["rows"][0]["first_layer_above_diagnostic_threshold"], 12)

    def test_first_difference_at_hip_to_hip_boundary(self):
        ref, het = runs(self.root, perturb_layer=30, perturb=0.1)
        r = layers.compare(ref, het, 0.001)
        self.assertEqual(r["rows"][0]["first_layer_above_diagnostic_threshold"], 30)

    def test_first_difference_before_boundary(self):
        ref, het = runs(self.root, perturb_layer=11, perturb=0.1)
        r = layers.compare(ref, het, 0.001)
        self.assertEqual(r["rows"][0]["first_layer_above_diagnostic_threshold"], 11)

    def test_model_source_mismatch_fails(self):
        ref, het = runs(self.root)
        het[0][0]["pack_index_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "pack_index_sha256"):
            layers.compare(ref, het, 0.001)

    def test_invalid_diagnostic_threshold_rejected(self):
        ref, het = runs(self.root)
        with self.assertRaisesRegex(ValueError, "threshold"):
            layers.compare(ref, het, -0.1)

    def test_heterogeneous_self_reference_rejected(self):
        _, het = runs(self.root)
        with self.assertRaisesRegex(ValueError, "unsplit HIP"):
            layers.compare(het, het, 0.001)


if __name__ == "__main__":
    unittest.main()
