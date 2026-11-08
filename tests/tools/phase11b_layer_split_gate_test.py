"""CPU-only negative/positive controls for the Phase 11B layer-split gate."""
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import sys

FILE = Path(__file__).resolve().parents[2] / "tools" / "phase11b_layer_split_gate.py"
spec = importlib.util.spec_from_file_location("phase11b_layer_split_gate", FILE)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class GateChecks(unittest.TestCase):
    def setUp(self):
        self.order = ("GPU-card1", "GPU-card2")
        self.log = "strata serve: layer split: layers 0-18 (CUDA0), 19-47 (CUDA1), one hand-off\n"
        self.records = [
            {"name": "A", "tokens": [17], "finish": "length", "reused": 0, "log_excerpt": ""},
            {"name": "B", "tokens": [19], "finish": "length", "reused": 0, "log_excerpt": ""},
            {"name": "A-return", "tokens": [17], "finish": "length", "reused": 384,
             "log_excerpt": "conversation cache: restored 384 tokens (checkpoint)\n"},
        ]

    def test_verified(self):
        gate.verify_run(self.log, self.records, self.order)

    def test_missing_second_stage(self):
        with self.assertRaisesRegex(gate.GateFailure, "two-stage"):
            gate.verify_run("one stage CUDA0\n", self.records, self.order)

    def test_missing_restore(self):
        self.records[-1]["log_excerpt"] = "nothing restored"
        with self.assertRaisesRegex(gate.GateFailure, "restore log"):
            gate.verify_run(self.log, self.records, self.order)

    def test_missing_reuse(self):
        self.records[-1]["reused"] = 0
        with self.assertRaisesRegex(gate.GateFailure, "reused"):
            gate.verify_run(self.log, self.records, self.order)

    def test_failed_done(self):
        self.records[1]["finish"] = "error"
        with self.assertRaisesRegex(gate.GateFailure, "DONE"):
            gate.verify_run(self.log, self.records, self.order)

    def test_missing_tokens(self):
        self.records[0]["tokens"] = []
        with self.assertRaisesRegex(gate.GateFailure, "token"):
            gate.verify_run(self.log, self.records, self.order)

    def test_same_uuid(self):
        with self.assertRaisesRegex(gate.GateFailure, "distinct"):
            gate.verify_run(self.log, self.records, ("GPU-1", "GPU-1"))

    def test_no_token_parity_requirement(self):
        self.records[-1]["tokens"] = [2468]
        gate.verify_run(self.log, self.records, self.order)

    def test_fixture_parse(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ids.txt"
            p.write_text(",".join(str(i) for i in range(128)))
            self.assertEqual(len(gate.parse_ids_file(p)), 128)
            p.write_text("1,2,banana\n")
            with self.assertRaises(gate.GateFailure):
                gate.parse_ids_file(p)

    def test_fake_engine_executes_full_aba_gate(self):
        order = ("GPU-one", "GPU-two")
        calls = []
        class FakeEngine:
            def __init__(self, executable, args, *, cwd, log, env):
                calls.append(("launch", executable, list(args), dict(env)))
                self.log = Path(log)
                self.log.write_text(
                    "strata serve: layer split: layers 0-23 (CUDA0), 24-47 (CUDA1), one hand-off\n")
                self.n = 0
                self.last = {}
            def generate(self, ids, new, sampling, cancel):
                self.n += 1
                calls.append(("gen", self.n, tuple(ids), new))
                self.last = {"finish": "length", "reused": 300 if self.n == 3 else 0}
                if self.n == 3:
                    with self.log.open("a") as f:
                        f.write("strata serve: conversation cache: restored 300 tokens (checkpoint)\n")
                yield 201 + self.n
            def close(self):
                calls.append(("close",))

        server = types.ModuleType("serve.server")
        server.StrataEngine = FakeEngine
        server.child_env = lambda c: {}
        server.engine_args = lambda c: list(c["args"]) + ["--layer-split", "auto"]
        serve = types.ModuleType("serve")
        serve.server = server
        with tempfile.TemporaryDirectory() as d:
            with patch.dict(sys.modules, {"serve": serve, "serve.server": server}):
                r = gate.run_one({"args": ["--serve"], "cwd": None}, Path("/dummy/strata"),
                                 Path(d), order, [100] * 200, [101] * 200, 1, 256)
            self.assertEqual(r["status"], "pass")
            self.assertEqual([v["name"] for v in r["records"]], ["A", "B", "A-return"])
            self.assertIn("--layer-split", r["engine_args"])
            self.assertEqual(calls[0][3]["ROCR_VISIBLE_DEVICES"], "GPU-one,GPU-two")
            self.assertEqual(calls[0][3]["HSA_ENABLE_SDMA"], "0")
            self.assertEqual(calls[-1], ("close",))

    def test_dry_run_no_write(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "evidence"
            result = gate.main(["--config", "/nonexistent/config", "--engine", "/nonexistent/engine",
                                "--output", str(out), "--uuid-a", "GPU-1", "--uuid-b", "GPU-2"])
            self.assertEqual(result, 0)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
