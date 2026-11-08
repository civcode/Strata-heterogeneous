"""CPU-only controls for the opt-in Phase 11B real-model startup fault gate."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
FILE = ROOT / "tools" / "phase11b_layer_split_startup_failure.py"
spec = importlib.util.spec_from_file_location("phase11b_startup_failure", FILE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class Verify(unittest.TestCase):
    ok = ("PHASE11B_FAULT_INJECTED stream-create device=1 skipped=0\n"
          "strata generate: layer split, CUDA1: its streams failed\n")

    def test_expected_negative(self):
        mod.verify(1, self.ok, target=1, skip=0)

    def test_no_injection(self):
        with self.assertRaisesRegex(mod.GateFailure, "injection markers"):
            mod.verify(1, "strata generate: layer split, CUDA1: its streams failed\n", target=1, skip=0)

    def test_duplicate_injection(self):
        with self.assertRaisesRegex(mod.GateFailure, "multiple"):
            mod.verify(1, self.ok + self.ok, target=1, skip=0)

    def test_wrong_device(self):
        with self.assertRaisesRegex(mod.GateFailure, "wrong GPU"):
            mod.verify(1, self.ok, target=0, skip=0)

    def test_wrong_call(self):
        with self.assertRaisesRegex(mod.GateFailure, "wrong GPU or call"):
            mod.verify(1, self.ok, target=1, skip=1)

    def test_unrelated_failure(self):
        with self.assertRaisesRegex(mod.GateFailure, "stream-creation refusal"):
            mod.verify(1, self.ok.split("strata generate:")[0], target=1, skip=0)

    def test_success_not_accepted(self):
        with self.assertRaisesRegex(mod.GateFailure, "exit 1"):
            mod.verify(0, self.ok, target=1, skip=0)

    def test_crash_not_accepted(self):
        with self.assertRaisesRegex(mod.GateFailure, "exit 1"):
            mod.verify(139, self.ok, target=1, skip=0)

    def test_refusal_before_marker_not_accepted(self):
        reversed_log = ("strata generate: layer split, CUDA1: its streams failed\n"
                        "PHASE11B_FAULT_INJECTED stream-create device=1 skipped=0\n")
        with self.assertRaisesRegex(mod.GateFailure, "preceded"):
            mod.verify(1, reversed_log, target=1, skip=0)

    def test_ready_not_accepted(self):
        with self.assertRaisesRegex(mod.GateFailure, "READY"):
            mod.verify(1, self.ok + "READY 4096\n", target=1, skip=0)

    def test_dry_run_does_not_access_resources(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "output"
            ret = mod.main(["--config", str(Path(tmp) / "absent-config"),
                            "--engine", "/missing/exe", "--interposer", "/missing/interposer",
                            "--output", str(out), "--uuid-first", "GPU-a", "--uuid-second", "GPU-b"])
            self.assertEqual(ret, 0)
            self.assertFalse(out.exists())

    def test_duplicate_uuid_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(mod.GateFailure, "distinct"):
                mod.main(["--config", "x", "--engine", "x", "--interposer", "x",
                          "--output", str(Path(tmp) / "out"), "--uuid-first", "GPU-a", "--uuid-second", "GPU-a"])

    def test_owned_child_success_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "child.log"
            rc, duration = mod.run_owned([sys.executable, "-c", "print('test child')"],
                                         env=os.environ.copy(), cwd=None, log_path=log, timeout_s=5)
            self.assertEqual(rc, 0)
            self.assertIn("test child", log.read_text())
            self.assertGreaterEqual(duration, 0)

    def test_timeout_kills_only_child_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "child.log"
            with self.assertRaisesRegex(mod.GateFailure, "exceeded"):
                mod.run_owned([sys.executable, "-c", "import time;time.sleep(20)"],
                              env=os.environ.copy(), cwd=None, log_path=log, timeout_s=.12)

    def test_interposer_host_abi(self):
        compiler = shutil.which("g++") or shutil.which("clang++")
        if not compiler:
            self.skipTest("C++ host compiler not installed")
        fake = ROOT / "tests" / "tools" / "phase11b_fakehip"
        interposer = ROOT / "tests" / "hip" / "phase11b_stream_create_fault.cpp"
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            defs = [compiler, "-std=c++17", "-I", str(fake)]
            subprocess.run(defs + ["-fPIC", "-shared", str(fake / "fake_runtime.cpp"),
                                   "-o", str(t / "libfakehip.so")], check=True, capture_output=True)
            subprocess.run(defs + ["-fPIC", "-shared", str(interposer), "-ldl",
                                   "-o", str(t / "libinject.so")], check=True, capture_output=True)
            subprocess.run(defs + [str(fake / "fake_client.cpp"), "-L", tmp,
                                   "-Wl,-rpath," + tmp, "-lfakehip", "-o", str(t / "fakeengine")],
                           check=True, capture_output=True)
            base = dict(os.environ, STRATA_PHASE11B_FAULT_ENABLE="stream-create-once",
                        STRATA_PHASE11B_FAULT_DEVICE="1", STRATA_PHASE11B_FAULT_SKIP="0",
                        PHASE11B_FAKE_DEVICE="1", LD_PRELOAD=str(t / "libinject.so"))
            result = subprocess.run([str(t / "fakeengine")], env=base, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            mod.verify(result.returncode, result.stdout + result.stderr, target=1, skip=0)
            # Skip exactly one successful target-device call, then fail the next.
            second = subprocess.run([str(t / "fakeengine")],
                                    env=dict(base, STRATA_PHASE11B_FAULT_SKIP="1", PHASE11B_FAKE_CALLS="3"),
                                    capture_output=True, text=True)
            self.assertEqual(second.returncode, 1)
            mod.verify(second.returncode, second.stdout + second.stderr, target=1, skip=1)
            self.assertEqual(second.stderr.count("PHASE11B_FAULT_INJECTED"), 1)
            # Negative control: the same hook on the wrong device must never report a hit.
            wrong = subprocess.run([str(t / "fakeengine")], env=dict(base, PHASE11B_FAKE_DEVICE="0"),
                                   capture_output=True, text=True)
            self.assertEqual(wrong.returncode, 0)
            self.assertIn("READY", wrong.stdout)
            self.assertNotIn("PHASE11B_FAULT_INJECTED", wrong.stderr)
            # Explicit opt-in absent: no fault even on the target device.
            noopt = subprocess.run([str(t / "fakeengine")],
                                   env=dict(base, STRATA_PHASE11B_FAULT_ENABLE="disabled"),
                                   capture_output=True, text=True)
            self.assertEqual(noopt.returncode, 0)
            self.assertNotIn("PHASE11B_FAULT_INJECTED", noopt.stderr)


if __name__ == "__main__":
    unittest.main()
