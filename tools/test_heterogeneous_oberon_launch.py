import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from tools.heterogeneous_oberon_launch import bind, find_reference, guard_engine, read_json, write_config
from strata_control.mixed_runconfig import resolve

ROOT = Path(__file__).resolve().parents[1]


class LaunchTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.public_profile = read_json(ROOT / "strata-q2_0-heterogeneous.json")
        self.profile = copy.deepcopy(self.public_profile)
        self.profile["heterogeneous"] = resolve(self.profile)
        self.profile["heterogeneous"].pop("native_spec_window", None)
        self.profile.pop("args")
        self.profile.pop("args_written")
        self.source = copy.deepcopy(self.profile)
        opts = self.source["heterogeneous"]
        # The original 0044 reference predates MTP; it retains chunk 128.
        for name in ("mtp", "spec", "spec_min_p"):
            opts.pop(name, None)
        opts["native_prefill_chunk"] = 128
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
        self.assertEqual(cfg["heterogeneous"]["max_new_tokens"], 256)
        self.assertEqual(cfg["heterogeneous"]["native_prefill_chunk"], 32)
        self.assertEqual(cfg["heterogeneous"]["mtp"], self.profile["heterogeneous"]["mtp"])
        self.assertEqual(cfg["heterogeneous"]["spec"], 4)
        self.assertEqual(cfg["heterogeneous"]["spec_min_p"], 0.5)
        self.assertEqual(self.source["heterogeneous"]["native_prefill_chunk"], 128)
        self.assertNotIn("mtp", self.source["heterogeneous"])
        self.assertEqual(cfg["heterogeneous"]["config"], self.source["heterogeneous"]["config"])
        self.assertNotIn("config", self.profile["heterogeneous"])
        path = write_config(self.root / "runs", cfg)
        self.assertEqual(read_json(path), cfg)
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(path.parent.stat().st_mode & 0o777, 0o700)

    def test_hip_without_cuda(self):
        profile = read_json(ROOT / "strata-q2_0-hip-only.json")
        src = self.source["heterogeneous"]
        src["plan"] = resolve(profile)["plan"]
        src["config"] = src["config"][1:]
        del src["cuda_worker"]
        self.assertNotIn("cuda_worker", bind(profile, self.source)["heterogeneous"])

    def test_ordinary_mixed_still_binds(self):
        ordinary = copy.deepcopy(self.profile)
        opts = ordinary["heterogeneous"]
        for name in ("mtp", "spec", "spec_min_p"):
            opts.pop(name)
        opts.update(native_prefill_chunk=128, max_new_tokens=2048)
        self.assertEqual(bind(ordinary, self.source)["heterogeneous"]["max_new_tokens"], 2048)
        opts["native_prefill_chunk"] = 32
        with self.assertRaisesRegex(ValueError, "native_prefill_chunk"):
            bind(ordinary, self.source)

    def test_mtp_profile_limits(self):
        for field, value in (("native_prefill_chunk", 0), ("native_prefill_chunk", 65),
                             ("native_prefill_chunk", 128), ("max_new_tokens", 257),
                             ("max_new_tokens", True), ("spec", 3), ("spec", True),
                             ("spec_min_p", 0.4), ("mtp", ""), ("mtp", "relative/rt"),
                             ("mtp", None), ("pipeline_mode", "fast"),
                             ("restart_clean", True), ("long_context", False)):
            bad = copy.deepcopy(self.profile)
            bad["heterogeneous"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                bind(bad, self.source)
        for chunk in (1, 32, 64):
            profile = copy.deepcopy(self.profile)
            profile["heterogeneous"]["native_prefill_chunk"] = chunk
            self.assertEqual(bind(profile, self.source)["heterogeneous"]["native_prefill_chunk"], chunk)

    def test_mtp_reference_is_not_a_target_reference(self):
        bad = copy.deepcopy(self.source)
        bad["heterogeneous"]["mtp"] = self.profile["heterogeneous"]["mtp"]
        with self.assertRaisesRegex(ValueError, "ordinary 0044 target reference"):
            bind(self.profile, bad)

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

    def test_native_profile_binding_keeps_public_args(self):
        cfg = bind(self.public_profile, self.source)
        self.assertEqual(cfg["args"], self.public_profile["args"])
        self.assertEqual(cfg["exe"], self.source["heterogeneous"]["coordinator"])
        self.assertNotIn("coordinator", cfg["heterogeneous"])
        self.assertNotIn("spec", cfg["heterogeneous"])
        self.assertEqual(resolve(cfg)["native_spec_window"], 4)

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
