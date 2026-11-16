import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from tools.heterogeneous_oberon_launch import bind, find_reference, guard_engine, read_json, write_config
from serve.heterogeneous_config import native_run_config, profile_binding
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


class LaunchTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.profile = read_json(ROOT / "strata-q2_0-heterogeneous.json")
        self.source = profile_binding(self.profile)
        model = ["--pack", str(self.root / "pack"), "--native", str(self.root / "first.gguf"),
                 "--ple-gguf", str(self.root / "second.gguf"), "--expert-profile", str(self.root / "profile")]
        for target in ("tools.heterogeneous_oberon_launch.stage_model_args",
                       "serve.heterogeneous_config.stage_model_args"):
            mock = patch(target, return_value=model)
            mock.start()
            self.addCleanup(mock.stop)
        opts = self.source["heterogeneous"]
        opts["max_new_tokens"] = 4
        for name in ("coordinator", "cuda_worker", "hip_worker"):
            path = self.root / name
            path.touch()
            path.chmod(0o700)
            opts[name] = str(path)
        opts["config"] = []
        for n in range(3):
            path = self.root / f"stage{n}.cfg"
            path.touch()
            opts["config"].append(str(path))

    def test_bind_mixed(self):
        cfg = bind(self.profile, self.source)
        self.assertEqual(cfg["heterogeneous"]["max_new_tokens"], 2048)
        self.assertEqual(cfg["heterogeneous"]["config"], self.source["heterogeneous"]["config"])
        self.assertNotIn("heterogeneous", self.profile)
        self.assertEqual(cfg["exe"], self.source["heterogeneous"]["coordinator"])
        self.assertNotIn("coordinator", cfg["heterogeneous"])
        self.assertIn("--pack", cfg["args"])
        self.assertNotIn("max_context", cfg["heterogeneous"])
        path = write_config(self.root / "runs", cfg)
        self.assertEqual(read_json(path), cfg)
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(path.parent.stat().st_mode & 0o777, 0o700)

    def test_native_run_config(self):
        cfg = bind(self.profile, self.source)
        opts = SimpleNamespace(stream_tokens=True, lookup_draft=False, prompt_chunk=1,
                               restart_clean=False, plan=profile_binding(self.profile)["heterogeneous"]["plan"])
        argv = [cfg["exe"], "--plan", opts.plan, "--fingerprint", "2026111301", "--kv", "int8",
                "--context-limit", "262144", "--generate", "2048", "--source-gguf", "/unused",
                "--source-index", "index", "--stop-ids", "99", "--native-prefill-chunk", "128",
                "--shared", "--group-native-hip", "--workload-mode", "--long-context",
                "--resident", "--stream-tokens"]
        result = native_run_config(cfg, (opts, argv, {"context_limit":262144}, None, {99}, None))
        self.assertNotIn("heterogeneous", result)
        self.assertNotIn("--resident", result["args"])
        self.assertNotIn("--context-limit", result["args"])
        self.assertEqual(result["args"].count("--kv"), 1)
        self.assertEqual(result["args"][result["args"].index("--layer-split")+1], "15,33")
        self.assertEqual(result["args"][result["args"].index("--eos-ids")+1], "99")
        self.assertIn("--hetero-max-new", result["args"])
        self.assertNotIn("--plan", result["args"])
        self.assertEqual(result["args"][result["args"].index("--hetero-devices")+1], "cuda:0,hip:0,hip:1")
        self.assertIn("heterogeneous", cfg)
        with self.assertRaises(ValueError):
            native_run_config(cfg, (opts, argv+["--unknown"], {"context_limit":262144}, None, {99}, None))

    def test_hip_without_cuda(self):
        profile = read_json(ROOT / "strata-q2_0-hip-only.json")
        src = self.source["heterogeneous"]
        src["plan"] = profile_binding(profile)["heterogeneous"]["plan"]
        src["config"] = src["config"][1:]
        del src["cuda_worker"]
        self.assertNotIn("cuda_worker", bind(profile, self.source)["heterogeneous"])

    def test_wrong_reference(self):
        for field, value in (("kv", "f16"), ("max_context", 32768), ("plan", "wrong"),
                             ("restart_clean", True), ("fingerprint", 123), ("native_prefill_chunk", 256)):
            bad = copy.deepcopy(self.source)
            bad["heterogeneous"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                bind(self.profile, bad)
        self.source["heterogeneous"]["config"][0] = "relative.cfg"
        with self.assertRaises(ValueError):
            bind(self.profile, self.source)

    def test_hash_discovery(self):
        path = self.root / "some-server.json"
        raw = json.dumps(self.source).encode()
        path.write_bytes(raw)
        (self.root / "newer.json").write_text("{}")
        self.assertEqual(find_reference(self.root, hashlib.sha256(raw).hexdigest()), path)
        with self.assertRaises(ValueError):
            find_reference(self.root, "0" * 64)

    def test_process_guard(self):
        with patch("tools.heterogeneous_oberon_launch.subprocess.run", return_value=Mock(returncode=1)) as run:
            guard_engine()
            self.assertEqual(run.call_count, 3)
        for status in (0, 2):
            with patch("tools.heterogeneous_oberon_launch.subprocess.run", return_value=Mock(returncode=status)):
                with self.assertRaises(ValueError):
                    guard_engine()


if __name__ == "__main__":
    unittest.main()
