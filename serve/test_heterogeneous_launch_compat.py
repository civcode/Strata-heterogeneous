"""CPU-only coverage for shared launch, environment and diagnostics paths."""
import contextlib
import io
import json
import os
from pathlib import Path
import queue
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from serve import server
from serve.heterogeneous import ResidentEngine
from strata_cli.heterogeneous import _native


class LaunchCompatibilityTests(unittest.TestCase):
    def test_preflight_keeps_protocol_separate_from_log(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "engine.log"
            code = ('import os, sys, json; '
                    'print("PREFLIGHT marker"); '
                    'print("HETERO_SOURCE " + json.dumps({"cwd": os.getcwd(), "flag": os.environ["FIXTURE"]})); '
                    'print("HETERO_CAPACITY sample", file=sys.stderr)')
            with log.open("a", encoding="utf-8") as stream, contextlib.redirect_stderr(io.StringIO()) as terminal:
                record = _native([sys.executable, "-c", code], marker="HETERO_SOURCE",
                                 diagnostic=stream, cwd=directory, env={**os.environ, "FIXTURE": "native"})
            self.assertEqual(record, {"cwd": directory, "flag": "native"})
            self.assertEqual(terminal.getvalue(), "")
            contents = log.read_text()
            self.assertIn("PREFLIGHT marker", contents)
            self.assertIn("HETERO_CAPACITY sample", contents)
            self.assertNotIn("HETERO_SOURCE", contents)

    def test_resident_log_does_not_consume_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = object.__new__(ResidentEngine)
            engine.log_path = str(Path(directory) / "engine.log")
            engine._records = queue.Queue()
            engine.proc = SimpleNamespace(stdout=io.StringIO("PREFLIGHT example\nHETERO_READY 1\n"))
            with contextlib.redirect_stderr(io.StringIO()) as terminal:
                engine._read_records()
                engine._diagnostic("HETERO_REQUEST_DONE request=1")
            self.assertEqual(terminal.getvalue(), "")
            self.assertEqual(engine._records.get_nowait(), "HETERO_READY 1")
            self.assertIsNone(engine._records.get_nowait())
            self.assertEqual(Path(engine.log_path).read_text(),
                             "PREFLIGHT example\nHETERO_REQUEST_DONE request=1\n")

    def test_mixed_child_environment_uses_native_common_rules(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                "HIP_VISIBLE_DEVICES": "0,1", "CUDA_VISIBLE_DEVICES": "0"}, clear=True), \
                patch.object(server, "gpu_all_off", side_effect=AssertionError("homogeneous GPU selection")), \
                patch.object(server, "gdn_stack_env", side_effect=AssertionError("homogeneous kernel defaults")), \
                patch.object(server, "parallel_stack_env", side_effect=AssertionError("homogeneous slot defaults")):
            env = server.child_env({"env": {"HSA_ENABLE_SDMA": "0", "STRATA_OFFLINE": "0"},
                                    "lib_dirs": [directory]}, offline=True, mixed_runtime=True)
            self.assertEqual(env["HSA_ENABLE_SDMA"], "0")
            self.assertEqual(env["STRATA_OFFLINE"], "1")
            self.assertEqual(env["HIP_VISIBLE_DEVICES"], "0,1")
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
            self.assertEqual(env["PATH" if os.name == "nt" else "LD_LIBRARY_PATH"].split(os.pathsep)[0], directory)

    def test_native_entrypoint_and_legacy_entrypoint_reach_same_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "server.json"
            config.write_text(json.dumps({"heterogeneous": {}, "parallel": 1, "port": 8081}))
            for name in ("strata", "heterogeneous"):
                with self.subTest(engine=name), patch.dict(os.environ, {}, clear=True), \
                        patch.object(server.lifeline, "install_stop_handlers"), \
                        patch.object(server, "rttime_limit_note", return_value=None), \
                        patch.object(server, "Server"), \
                        patch("serve.heterogeneous.prepare_server",
                              side_effect=ValueError("fixture reached preflight")) as preflight, \
                        contextlib.redirect_stderr(io.StringIO()) as stderr:
                    with self.assertRaises(SystemExit) as error:
                        server.main(["--engine", name, "--config", str(config),
                                     "--port", "8081", "--open", "--offline"])
                    self.assertEqual(error.exception.code, 2)
                    self.assertIn("fixture reached preflight", stderr.getvalue())
                    preflight.assert_called_once()
                    effective = preflight.call_args.args[0]
                    if name == "strata":
                        self.assertEqual(effective["log"], str(config.with_suffix(".log")))
                    else:
                        self.assertNotIn("log", effective)

    def test_unimplemented_engine_args_are_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "server.json"
            config.write_text(json.dumps({"heterogeneous": {}, "args": ["--spec", "4"]}))
            with patch.object(server.lifeline, "install_stop_handlers"), \
                    patch.object(server, "rttime_limit_note", return_value=None), \
                    patch("serve.heterogeneous.prepare_server", wraps=__import__("serve.heterogeneous", fromlist=["prepare_server"]).prepare_server) as preflight, \
                    contextlib.redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as error:
                    server.main(["--engine", "strata", "--config", str(config)])
                self.assertEqual(error.exception.code, 2)
                self.assertIn("requires explicit --max-context and --kv", stderr.getvalue())
                preflight.assert_called_once()


if __name__ == "__main__":
    unittest.main()
