#!/usr/bin/env python3
"""Fake-GPU ownership tests for Phase 11B cvec tables and penalty history."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
FAKE = ROOT / "tests/core/phase11b_control_shutdown_fake"
TEST = ROOT / "tests/core/phase11b_control_shutdown_test.cpp"


class ControlShutdownTest(unittest.TestCase):
    def check_compiler(self, compiler, flags):
        if shutil.which(compiler) is None:
            self.skipTest(f"{compiler} unavailable")
        with tempfile.TemporaryDirectory(prefix="strata-control-shutdown-") as tmp:
            executable = Path(tmp) / "control-shutdown"
            build = subprocess.run(
                [compiler, "-std=c++20", "-Wall", "-Wextra", "-Werror", *flags,
                 "-I", str(FAKE), "-I", str(ROOT / "include"),
                 str(TEST), "-o", str(executable)],
                capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            modes = ("tables-empty", "tables", "partial", "buffer", "buffer-same",
                     "fail-get", "fail-set", "fail-sync", "fail-free")
            for mode in modes:
                with self.subTest(compiler=compiler, flags=flags, mode=mode):
                    env = dict(os.environ,
                               ASAN_OPTIONS="detect_leaks=0:halt_on_error=1",
                               UBSAN_OPTIONS="halt_on_error=1")
                    result = subprocess.run([str(executable), mode], text=True,
                                            capture_output=True, timeout=5, env=env)
                    expected = 1 if mode.startswith("fail-") else 0
                    self.assertEqual(result.returncode, expected,
                                     f"{mode}: {result.stdout}\n{result.stderr}")
                    if expected:
                        self.assertIn("exiting without releasing transfer owners", result.stderr)
                        self.assertNotIn("PASSED", result.stdout)
                    else:
                        self.assertIn("PASSED", result.stdout)
                        self.assertNotIn("ASSERTION_FAILED", result.stderr)

    def test_gcc_release(self):
        self.check_compiler("g++", ["-O2", "-DNDEBUG"])

    def test_clang_release(self):
        self.check_compiler("clang++", ["-O2", "-DNDEBUG"])

    def test_gcc_sanitized(self):
        self.check_compiler("g++", ["-O1", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"])

    def test_clang_sanitized(self):
        self.check_compiler("clang++", ["-O1", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"])


if __name__ == "__main__":
    unittest.main()
