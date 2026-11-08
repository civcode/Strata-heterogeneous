#!/usr/bin/env python3
"""CPU-only regression for early primary-device resource teardown."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
FAKE = ROOT / "tests/core/primary_early_resources_fake"
TEST = ROOT / "tests/core/primary_early_resources_test.cpp"


class PrimaryEarlyResourcesTest(unittest.TestCase):
    def check_compiler(self, compiler, flags):
        if not shutil.which(compiler):
            self.skipTest(f"{compiler} unavailable")
        with tempfile.TemporaryDirectory(prefix="strata-primary-early-") as temp:
            exe = Path(temp) / "primary-early"
            compile_run = subprocess.run(
                [compiler, "-std=c++20", "-Wall", "-Wextra", "-Werror", *flags,
                 "-I", str(FAKE), "-I", str(ROOT / "include"),
                 str(TEST), "-o", str(exe)],
                text=True, capture_output=True, timeout=60,
            )
            self.assertEqual(compile_run.returncode, 0,
                             compile_run.stdout + compile_run.stderr)
            modes = ("normal", "owner-same", "empty", "get-fail",
                     "select-fail", "sync-fail", "free-fail",
                     "stream-fail", "restore-fail")
            for mode in modes:
                with self.subTest(compiler=compiler, flags=flags, mode=mode):
                    env = dict(os.environ,
                               ASAN_OPTIONS="detect_leaks=0:halt_on_error=1",
                               UBSAN_OPTIONS="halt_on_error=1")
                    run = subprocess.run([str(exe), mode], capture_output=True,
                                         text=True, timeout=5, env=env)
                    expected = 1 if mode.endswith("-fail") else 0
                    self.assertEqual(run.returncode, expected,
                                     f"{mode}\n{run.stdout}\n{run.stderr}")
                    if expected:
                        self.assertIn("exiting without releasing transfer owners", run.stderr)
                        self.assertNotIn("PASSED", run.stdout)
                    else:
                        self.assertIn("PASSED", run.stdout)
                        self.assertNotIn("FAILED", run.stderr)
                        if mode == "empty":
                            self.assertNotIn("FREE ", run.stdout)
                            self.assertNotIn("STREAM ", run.stdout)
                        else:
                            self.assertEqual(run.stdout.count("FREE "), 5)
                            self.assertEqual(run.stdout.count("STREAM "), 1)

    def test_gcc_release(self):
        self.check_compiler("g++", ["-O2", "-DNDEBUG"])

    def test_clang_release(self):
        self.check_compiler("clang++", ["-O2", "-DNDEBUG"])

    def test_gcc_sanitized(self):
        self.check_compiler("g++", ["-O1", "-g", "-fsanitize=address,undefined",
                                   "-fno-omit-frame-pointer"])

    def test_clang_sanitized(self):
        self.check_compiler("clang++", ["-O1", "-g", "-fsanitize=address,undefined",
                                       "-fno-omit-frame-pointer"])


if __name__ == "__main__":
    unittest.main()
