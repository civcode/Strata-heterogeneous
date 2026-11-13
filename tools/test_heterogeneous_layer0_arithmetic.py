#!/usr/bin/env python3
"""CPU-only STL0/v1 and three-run arithmetic control fixtures."""
from __future__ import annotations

import json
from pathlib import Path
import struct
import tempfile
import unittest

import heterogeneous_layer0_arithmetic as l0
from heterogeneous_hip_control import prepare
from test_heterogeneous_compare_parity import packet


def make_stl0(*, emb_delta=0.0, mixed_delta=0.0, q8_byte=None,
              qkv_delta=0.0, position=0, window=1, n=2560):
    c = 8
    q = (n // 32) * 36
    data = bytearray(l0.HEADER.pack(
        l0.MAGIC, 1, 0, n, c, q, 1, 48, 1, window, position, 123, 0, 0))
    data.extend(struct.pack("<i", 1))
    emb = [0.01 * (i % 100) for i in range(n)]
    mixed = [0.02 * (i % 50) for i in range(n)]
    qbytes = bytearray((i * 7) % 256 for i in range(q))
    qkv = [i * 0.05 for i in range(c)]
    emb[0] += emb_delta
    mixed[0] += mixed_delta
    if q8_byte is not None:
        qbytes[3] ^= 0x40
    qkv[0] += qkv_delta
    data.extend(struct.pack(f"<{n}f", *emb))
    data.extend(struct.pack(f"<{n}f", *mixed))
    data.extend(qbytes)
    data.extend(struct.pack(f"<{c}f", *qkv))
    h = 2166136261
    for i, b in enumerate(data):
        if 64 <= i < 68:
            continue
        h = ((h ^ b) * 16777619) & 0xffffffff
    struct.pack_into("<I", data, 64, h)
    return bytes(data)


class LayerZeroTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ref = self.root / "ref"
        self.hip = self.root / "hip"
        self.cuda = self.root / "cuda"
        self.write(self.ref, "hip:0:0-48")
        self.write(self.hip, "hip:0:0-12,hip:1:12-48")
        self.write(self.cuda, "cuda:0:0-12,hip:0:12-30,hip:1:30-48")

    def write(self, prefix, plan, *, emb_delta=0, mixed_delta=0,
              q8_byte=None, qkv_delta=0):
        Path(f"{prefix}.complete.json").write_text(json.dumps({
            "schema": "strata-heterogeneous-parity/v1", "windows": 1,
            "plan": plan, "model_fingerprint": 2026111301}))
        Path(f"{prefix}.w1.json").write_text(json.dumps({
            "schema": "strata-heterogeneous-parity/v1",
            "plan": plan, "model_fingerprint": 2026111301,
            "plan_fingerprint": 123, "pack_index_sha256": "a" * 64,
            "gguf_shards": "/models/a.gguf;/models/b.gguf",
            "profile_path": "/pack/profile.strp", "window_id": 1,
            "pos0": 0, "rows": 1, "n_keep": 1, "token_ids": [1]}))
        Path(f"{prefix}.w1.stfr").write_bytes(packet(stage=plan.count(",")))
        Path(f"{prefix}.stage0.w1.stl0").write_bytes(make_stl0(
            emb_delta=emb_delta, mixed_delta=mixed_delta,
            q8_byte=q8_byte, qkv_delta=qkv_delta))

    def run_compare(self):
        return l0.compare(l0.read_run(self.ref), l0.read_run(self.hip),
                          l0.read_run(self.cuda))

    def test_header_72_bytes(self):
        self.assertEqual(l0.HEADER.size, 72)

    def test_identical_layer0_components(self):
        result = self.run_compare()
        self.assertEqual(result["status"], "DIAGNOSTIC_ONLY")
        self.assertIsNone(result["positions"][0]["comparisons"]
                          ["cuda_vs_reference"]["first_nonidentical_component"])

    def test_embedding_first_divergence(self):
        self.write(self.cuda, "cuda:0:0-12,hip:0:12-30,hip:1:30-48",
                   emb_delta=0.5)
        result = self.run_compare()["positions"][0]["comparisons"]
        self.assertEqual(result["cuda_vs_reference"]["first_nonidentical_component"],
                         "embedding")
        self.assertIsNone(result["hip_control_vs_reference"]
                          ["first_nonidentical_component"])

    def test_mixer_first_divergence(self):
        self.write(self.cuda, "cuda:0:0-12,hip:0:12-30,hip:1:30-48",
                   mixed_delta=0.5)
        result = self.run_compare()["positions"][0]["comparisons"]
        self.assertEqual(result["cuda_vs_reference"]["first_nonidentical_component"],
                         "hc_mixed")

    def test_q8_first_divergence(self):
        self.write(self.cuda, "cuda:0:0-12,hip:0:12-30,hip:1:30-48",
                   q8_byte=3)
        result = self.run_compare()["positions"][0]["comparisons"]
        self.assertEqual(result["cuda_vs_reference"]["first_nonidentical_component"],
                         "q8_input")
        self.assertEqual(result["cuda_vs_reference"]["components"]
                         ["q8_input"]["mismatched_bytes"], 1)

    def test_qkv_first_divergence(self):
        self.write(self.cuda, "cuda:0:0-12,hip:0:12-30,hip:1:30-48",
                   qkv_delta=0.75)
        result = self.run_compare()["positions"][0]["comparisons"]
        self.assertEqual(result["cuda_vs_reference"]["first_nonidentical_component"],
                         "gdn_qkv")

    def test_split_control_differs(self):
        self.write(self.hip, "hip:0:0-12,hip:1:12-48", mixed_delta=0.1)
        result = self.run_compare()["positions"][0]["comparisons"]
        self.assertEqual(result["hip_control_vs_reference"]["first_nonidentical_component"],
                         "hc_mixed")

    def test_corrupted_checksum_fails(self):
        path = Path(f"{self.cuda}.stage0.w1.stl0")
        b = bytearray(path.read_bytes())
        b[-1] ^= 3
        path.write_bytes(b)
        with self.assertRaisesRegex(ValueError, "checksum"):
            l0.read_run(self.cuda)

    def test_truncated_component_fails(self):
        path = Path(f"{self.cuda}.stage0.w1.stl0")
        path.write_bytes(path.read_bytes()[:-8])
        with self.assertRaisesRegex(ValueError, "payload length"):
            l0.read_run(self.cuda)

    def test_missing_capture_fails(self):
        Path(f"{self.cuda}.stage0.w1.stl0").unlink()
        with self.assertRaisesRegex(ValueError, "missing layer-zero"):
            l0.read_run(self.cuda)

    def test_model_identity_mismatch_fails(self):
        path = Path(f"{self.cuda}.w1.json")
        doc = json.loads(path.read_text())
        doc["pack_index_sha256"] = "b" * 64
        path.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError, "pack_index_sha256"):
            self.run_compare()

    def test_wrong_plan_fails(self):
        with self.assertRaisesRegex(ValueError, "expected unsplit HIP"):
            l0.compare(l0.read_run(self.cuda), l0.read_run(self.hip),
                       l0.read_run(self.ref))

    def test_incomplete_run_fails(self):
        Path(f"{self.cuda}.complete.json").unlink()
        with self.assertRaisesRegex(ValueError, "completion marker"):
            l0.read_run(self.cuda)

    def test_hip_control_config_generation(self):
        a = self.root / "cuda-stage0.cfg"
        b = self.root / "hip-stage1.cfg"
        required = {
            "version": "2", "pack_dir": "/pack", "profile_path": "/pack/rank.strp",
            "index_sha256": "a" * 64, "gguf_shards": "/model/a.gguf;/model/b.gguf",
            "model_fingerprint": "2026111301", "n_layers": "48",
            "n_vocab": "248320", "hc": "4", "n_embd": "2560",
            "max_cells": "2048", "max_t": "8",
            "free_reserve_bytes": "536870912", "arena_budget_bytes": "3456000000",
            "cpu_workers": "4", "max_cache_slots": "16",
        }
        for path, ext in ((a, dict(stage_index="0", first_layer="0",
                                  end_layer="12", device_ordinal="0",
                                  ple_gguf_path="/model/b.gguf")),
                          (b, dict(stage_index="1", first_layer="12",
                                  end_layer="30", device_ordinal="0"))):
            path.write_text("".join(f"{k}={v}\n"
                                    for k, v in (required | ext).items()))
        x, y = self.root / "hip-stage0.cfg", self.root / "hip-stage1-full.cfg"
        prepare(a, b, x, y)
        self.assertIn("first_layer=0\nend_layer=12\n", x.read_text())
        self.assertIn("first_layer=12\nend_layer=48\n", y.read_text())
        self.assertIn("device_ordinal=1\n", y.read_text())
        self.assertIn("arena_budget_bytes=0\n", x.read_text())
        self.assertIn("arena_budget_bytes=0\n", y.read_text())
        with self.assertRaises(FileExistsError):
            prepare(a, b, x, y)

    def test_hip_control_rejects_source_divergence(self):
        self.test_hip_control_config_generation()
        a, b = self.root / "cuda-stage0.cfg", self.root / "hip-stage1.cfg"
        b.write_text(b.read_text().replace("index_sha256=" + "a" * 64,
                                          "index_sha256=" + "b" * 64))
        with self.assertRaisesRegex(ValueError, "index_sha256"):
            prepare(a, b, self.root / "fresh1", self.root / "fresh2")


if __name__ == "__main__":
    unittest.main()
