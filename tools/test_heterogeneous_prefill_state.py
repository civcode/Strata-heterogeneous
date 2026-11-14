"""Synthetic state files; no GPU claims."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

import heterogeneous_prefill_state as state
from heterogeneous_compare_parity import ParityError


class StateTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.prefix = Path(tmp.name) / "capture"
        self.run_marker = Path(f"{self.prefix}.prefill.json")
        self.run_marker.write_text(json.dumps({
            "schema": "strata-heterogeneous-prefill-parity/v1", "state_capture": True,
            "prefix_rows": 2, "token_ids": [1, 3, 2],
            "plan": "cuda:0:0-4,hip:0:4-24,hip:1:24-48"}))
        payloads = {"ple.history": ("f32", struct.pack("<2f", 1., 2.))}
        for j in range(3):
            for name in ("recurrence", "conv"):
                payloads[f"gdn{j}.{name}"] = ("f32", struct.pack("<2f", j, 2.))
        for name in ("tail", "dead", "pooled"):
            payloads[f"qsa0.{name}"] = ("f32", struct.pack("<2f", 1., 2.))
        payloads["qsa0.block_pos"] = ("i32", struct.pack("<i", 2))
        for name in ("k", "v", "k_scale", "v_scale"):
            payloads[f"qsa0.{name}"] = ("opaque", bytes(8))
        for request in (1, 2, 3):
            stem = f"{self.prefix}.state.r{request}.p2"
            meta = {"schema": state.SCHEMA, "stage": 0, "request": request,
                    "position": 2, "layer_lo": 0, "layer_hi": 4, "gdn_ord0": 0,
                    "gdn_alloc": 3, "qsa_ord0": 0, "qsa_alloc": 1, "ple_prev": [1, 3],
                    "qsa": [{"ordinal": 0, "format": 1, "cells": 16, "page_size": 16,
                             "heads": 1, "head_dim": 4, "pooled_rows": 1, "idx_dim": 2}],
                    "fields": []}
            for name, (kind, payload) in payloads.items():
                meta["fields"].append({"name": name, "type": kind, "bytes": len(payload)})
                Path(f"{stem}.{name}.bin").write_bytes(payload)
            Path(stem + ".json").write_text(json.dumps(meta))

    def native(self, name):
        return Path(f"{self.prefix}.state.r2.p2.{name}")

    def mutate_meta(self, fn):
        p = self.native("json")
        data = json.loads(p.read_text())
        fn(data)
        p.write_text(json.dumps(data))

    def test_equal(self):
        out = state.analyze(self.prefix)
        self.assertEqual(out["status"], "IDENTICAL")
        self.assertEqual(out["expected_ple_prev"], [1, 3])

    def test_gdn_difference(self):
        self.native("gdn0.conv.bin").write_bytes(struct.pack("<2f", 1., 2.))
        out = state.analyze(self.prefix)
        self.assertEqual(out["status"], "DIFFERENT")
        self.assertEqual(out["comparisons"]["serial-a_vs_native"]["gdn0.conv"]["max_abs"], 1.)
        self.assertTrue(out["comparisons"]["serial-a_vs_serial-b"]["gdn0.conv"]["bytes_equal"])

    def test_kv_difference(self):
        self.native("qsa0.k.bin").write_bytes(b"\x01" + bytes(7))
        out = state.analyze(self.prefix)
        field = out["comparisons"]["serial-a_vs_native"]["qsa0.k"]
        self.assertEqual(field["first_different_byte"], 0)

    def test_history_difference(self):
        self.mutate_meta(lambda m: m.update(ple_prev=[-1, 3]))
        self.assertEqual(state.analyze(self.prefix)["status"], "DIFFERENT")

    def test_nonfinite(self):
        self.native("ple.history.bin").write_bytes(struct.pack("<2f", float("nan"), 2.))
        out = state.analyze(self.prefix)
        self.assertEqual(out["comparisons"]["serial-a_vs_native"]["ple.history"]["nonfinite_values"], 1)

    def test_truncated_file(self):
        self.native("ple.history.bin").write_bytes(b"")
        with self.assertRaisesRegex(ParityError, "truncated"):
            state.analyze(self.prefix)

    def test_wrong_request(self):
        self.mutate_meta(lambda m: m.update(request=1))
        with self.assertRaisesRegex(ParityError, "identity"):
            state.analyze(self.prefix)

    def test_duplicate_field(self):
        self.mutate_meta(lambda m: m["fields"].__setitem__(1, m["fields"][0]))
        with self.assertRaisesRegex(ParityError, "duplicate"):
            state.analyze(self.prefix)

    def test_missing_marker(self):
        self.run_marker.unlink()
        with self.assertRaises(FileNotFoundError):
            state.analyze(self.prefix)


if __name__ == "__main__":
    unittest.main()
