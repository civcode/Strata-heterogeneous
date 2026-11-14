"""CPU-only M7 paired prefill capture tests using real STFR decoder/comparator."""
import json
import struct
import tempfile
import unittest
from pathlib import Path

import heterogeneous_compare_parity as base
import heterogeneous_prefill_parity as prefill


class PrefillParityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.prefix = Path(self.tmp.name) / "capture"
        self.plan = "cuda:0:0-12,hip:0:12-30,hip:1:30-48"
        self.tokens = [1, 2, 0, 3, 2]
        self.cut = 3
        self.write_capture()

    def write_capture(self, native_logits=(0., 1., 2., 4.), native_pred=3):
        p = self.prefix
        Path(f"{p}.prefill.json").write_text(json.dumps({
            "schema": prefill.PREFILL_SCHEMA, "plan": self.plan, "model_fingerprint": 9,
            "prefix_rows": self.cut, "chunk": 2, "hc": 1, "n_embd": 2, "token_ids": self.tokens}))
        for request, mode in enumerate(prefill.MODES, 1):
            for pos in range(self.cut, len(self.tokens)):
                window = pos - self.cut + 2 if mode == "native" else pos + 1
                stem = f"{p}.{mode}.w{window}"
                Path(stem + ".json").write_text(json.dumps({
                    "schema": base.SCHEMA, "plan": self.plan, "model_fingerprint": 9,
                    "plan_fingerprint": 123, "pack_index_sha256": "hash", "gguf_shards": "model",
                    "profile_path": "profile", "window_id": window, "rows": 1, "n_keep": 1,
                    "pos0": pos, "token_ids": [self.tokens[pos]]}))
                logits = native_logits if mode == "native" else (0., 1., 2., 4.)
                pred = native_pred if mode == "native" else 3
                payload = struct.pack("<iI4f", pred, 0, *logits)
                buf = bytearray(base.HEADER.pack(base.MAGIC, 1, 1, 2, 1, 4, 0,
                                               request, window, 123, len(payload), 0, 0) + payload)
                digest = 2166136261
                for i, byte in enumerate(buf):
                    if not 52 <= i < 56:
                        digest = ((digest ^ byte) * 16777619) & 0xffffffff
                struct.pack_into("<I", buf, 52, digest)
                Path(stem + ".stfr").write_bytes(buf)
        for window, rows in ((1, 2), (3, 1)):
            data = bytes(range(rows * 8))
            Path(f"{p}.stage0.r2.w{window}.out.f32").write_bytes(data)
            Path(f"{p}.stage1.r2.w{window}.in.f32").write_bytes(data)

    def test_same_teacher_forced_rows_with_distinct_actual_windows_pass(self):
        result = prefill.analyze(self.prefix)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["bridge"]["matched_pieces"], 2)
        self.assertEqual(result["comparisons"]["serial-a_vs_native"]["row_metrics"][0]["position"], 3)

    def test_kl_failure_without_top1_change(self):
        self.write_capture(native_logits=(-2., -1., 1., 4.))
        self.assertEqual(prefill.analyze(self.prefix)["status"], "FAIL")

    def test_prediction_failure(self):
        self.write_capture(native_pred=2)
        self.assertEqual(prefill.analyze(self.prefix)["status"], "FAIL")

    def test_bridge_corruption(self):
        path = Path(f"{self.prefix}.stage1.r2.w3.in.f32")
        path.write_bytes(b"!" * 8)
        result = prefill.analyze(self.prefix)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["bridge"]["first_mismatched_piece_position"], 2)

    def test_wrong_physical_stage_filename(self):
        src = Path(f"{self.prefix}.stage1.r2.w3.in.f32")
        src.rename(Path(f"{self.prefix}.stage2.r2.w3.in.f32"))
        with self.assertRaisesRegex(base.ParityError, "physical residual"):
            prefill.analyze(self.prefix)

    def test_wrong_teacher_token(self):
        path = Path(f"{self.prefix}.native.w2.json")
        meta = json.loads(path.read_text())
        meta["token_ids"] = [0]
        path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(base.ParityError, "capture identity"):
            prefill.analyze(self.prefix)

    def test_stale_request_packet(self):
        a = Path(f"{self.prefix}.serial-a.w4.stfr")
        b = Path(f"{self.prefix}.serial-b.w4.stfr")
        b.write_bytes(a.read_bytes())
        with self.assertRaisesRegex(base.ParityError, "capture identity"):
            prefill.analyze(self.prefix)

    def test_missing_completion(self):
        Path(f"{self.prefix}.prefill.json").unlink()
        with self.assertRaises(FileNotFoundError):
            prefill.analyze(self.prefix)

    def test_default_comparison_still_rejects_same_mixed_plan(self):
        _, runs, _ = prefill.load(self.prefix)
        with self.assertRaisesRegex(base.ParityError, "unsplit"):
            base.compare(runs["serial-a"], runs["native"], .001)


if __name__ == "__main__":
    unittest.main()
