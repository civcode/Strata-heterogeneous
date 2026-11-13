#!/usr/bin/env python3
"""CPU-only synthetic committed-STFR parity contract tests."""
import json
import struct
import tempfile
import unittest
from pathlib import Path

import heterogeneous_compare_parity as parity
from heterogeneous_reference_config import make_reference


def packet(pred=3, logits=(0.0, 1.0, 2.0, 4.0),
           window=1, plan_fp=123, include_logits=True, stage=2):
    rows = 1
    vocab = 4
    payload = struct.pack("<iI", pred, 0)
    if include_logits:
        payload += struct.pack("<4f", *logits)
    h = bytearray(parity.HEADER.pack(
        parity.MAGIC, 1, 1 if include_logits else 0,
        stage, rows, vocab, 0, 1, window, plan_fp, len(payload), 0, 0))
    blob = bytearray(h + payload)
    checksum = 2166136261
    for i, byte in enumerate(blob):
        if 52 <= i < 56:
            continue
        checksum = ((checksum ^ byte) * 16777619) & 0xffffffff
    struct.pack_into("<I", blob, 52, checksum)
    return blob


class ParityFixtures(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ref = self.root / "ref"
        self.het = self.root / "het"
        self.write_run(self.ref, "hip:0:0-48")
        self.write_run(self.het, "cuda:0:0-12,hip:0:12-30,hip:1:30-48")

    def write_run(self, prefix, plan, *, count=1, predictions=None, logits=None,
                  indices=None, include_logits=True):
        Path(f"{prefix}.complete.json").write_text(json.dumps({
            "schema": parity.SCHEMA, "windows": count, "plan": plan,
            "model_fingerprint": 2026111301}))
        for i in range(1, count + 1):
            token = 1 if i == 1 else 2
            meta = {"schema": parity.SCHEMA, "plan": plan,
                    "model_fingerprint": 2026111301,
                    "plan_fingerprint": 123, "pack_index_sha256": "a" * 64,
                    "gguf_shards": "/models/part1.gguf;/models/part2.gguf",
                    "profile_path": "/pack/profile.strp", "window_id": i,
                    "pos0": i - 1, "rows": 1, "n_keep": 1, "token_ids": [token]}
            Path(f"{prefix}.w{i}.json").write_text(json.dumps(meta))
            Path(f"{prefix}.w{i}.stfr").write_bytes(packet(
                pred=predictions[i - 1] if predictions is not None else 3,
                logits=logits if logits is not None else (0.0, 1.0, 2.0, 4.0),
                window=i, include_logits=include_logits, stage=plan.count(",")))

    def test_identical_logits_and_predictions_pass(self):
        result = parity.compare(parity.read_run(self.ref), parity.read_run(self.het), 0.001)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["max_kl_ref_to_hetero"], 0.0)

    def test_two_windows_at_contiguous_positions_pass(self):
        self.write_run(self.ref, "hip:0:0-48", count=2)
        self.write_run(self.het, "cuda:0:0-12,hip:0:12-30,hip:1:30-48", count=2)
        self.assertEqual(parity.compare(parity.read_run(self.ref),
                        parity.read_run(self.het), 0.001)["rows"], 2)

    def test_prediction_mismatch_fails_even_when_logits_match(self):
        self.write_run(self.het, "cuda:0:0-12,hip:0:12-30,hip:1:30-48",
                       predictions=[2])
        self.assertEqual(parity.compare(parity.read_run(self.ref),
                         parity.read_run(self.het), 0.001)["status"], "FAIL")

    def test_kl_mismatch_fails_even_if_prediction_agrees(self):
        self.write_run(self.het, "cuda:0:0-12,hip:0:12-30,hip:1:30-48",
                       logits=(-2.0, -1.0, 1.0, 4.0))
        self.assertEqual(parity.compare(parity.read_run(self.ref),
                         parity.read_run(self.het), 0.001)["status"], "FAIL")

    def test_missing_commit_marker_fails(self):
        Path(f"{self.het}.complete.json").unlink()
        with self.assertRaisesRegex(parity.ParityError, "completion marker"):
            parity.read_run(self.het)

    def test_corrupt_checksum_fails(self):
        path = Path(f"{self.het}.w1.stfr")
        blob = bytearray(path.read_bytes())
        blob[-1] ^= 1
        path.write_bytes(blob)
        with self.assertRaisesRegex(parity.ParityError, "checksum"):
            parity.read_run(self.het)

    def test_missing_logits_fails(self):
        self.write_run(self.ref, "hip:0:0-48", include_logits=False)
        with self.assertRaisesRegex(parity.ParityError, "full FP32 logits"):
            parity.read_run(self.ref)

    def test_wrong_token_sequence_fails(self):
        path = Path(f"{self.het}.w1.json")
        meta = json.loads(path.read_text())
        meta["token_ids"] = [2]  # valid token ID, but mismatched sequence
        path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(parity.ParityError, "token_ids"):
            parity.compare(parity.read_run(self.ref), parity.read_run(self.het), 0.001)

    def test_wrong_pack_identity_fails(self):
        path = Path(f"{self.het}.w1.json")
        meta = json.loads(path.read_text())
        meta["pack_index_sha256"] = "b" * 64
        path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(parity.ParityError, "pack_index_sha256"):
            parity.compare(parity.read_run(self.ref), parity.read_run(self.het), 0.001)

    def test_missing_record_fails(self):
        Path(f"{self.het}.w1.stfr").unlink()
        with self.assertRaisesRegex(parity.ParityError, "missing STFR"):
            parity.read_run(self.het)

    def test_nonfinite_logits_fails(self):
        self.write_run(self.het, "cuda:0:0-12,hip:0:12-30,hip:1:30-48",
                       logits=(0.0, 1.0, float("nan"), 4.0))
        with self.assertRaisesRegex(parity.ParityError, "nonfinite"):
            parity.read_run(self.het)

    def test_reference_config_is_full_stage_and_unbounded_by_old_budget(self):
        src = self.root / "stage0.cfg"
        target = self.root / "hip-full.cfg"
        fields = dict(version="2", stage_index="0", first_layer="0",
                      end_layer="12", n_layers="48", device_ordinal="0",
                      pack_dir="/data/pack", profile_path="/data/rank.strp",
                      index_sha256="a" * 64, model_fingerprint="2026111301",
                      gguf_shards="/models/a.gguf;/models/b.gguf",
                      ple_gguf_path="/models/b.gguf", arena_budget_bytes="3000000000",
                      free_reserve_bytes="536870912", max_cells="2048",
                      max_t="8", n_vocab="248320", hc="4", n_embd="2560",
                      cpu_workers="4", max_cache_slots="16")
        src.write_text("".join(f"{k}={v}\n" for k, v in fields.items()))
        make_reference(src, target)
        data = target.read_text()
        self.assertIn("end_layer=48\n", data)
        self.assertIn("arena_budget_bytes=0\n", data)
        self.assertIn("device_ordinal=0\n", data)
        self.assertIn("ple_gguf_path=/models/b.gguf\n", data)
        with self.assertRaises(FileExistsError):
            make_reference(src, target)


if __name__ == "__main__":
    unittest.main()
