import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from tools.heterogeneous_oberon_launch import bind, find_reference, guard_engine, read_json, write_config

ROOT = Path(__file__).resolve().parents[1]


class LaunchTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.profile = read_json(ROOT / "strata-q2_0-heterogeneous.json")
        self.source = copy.deepcopy(self.profile)
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
        self.assertNotIn("config", self.profile["heterogeneous"])
        path = write_config(self.root / "runs", cfg)
        self.assertEqual(read_json(path), cfg)
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(path.parent.stat().st_mode & 0o777, 0o700)

    def test_hip_without_cuda(self):
        profile = read_json(ROOT / "strata-q2_0-hip-only.json")
        src = self.source["heterogeneous"]
        src["plan"] = profile["heterogeneous"]["plan"]
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
