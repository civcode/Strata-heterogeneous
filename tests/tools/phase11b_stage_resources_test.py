#!/usr/bin/env python3
"""CPU-only regression for late destruction of later-stage raw device owners."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "tests/core/layer_split_stage_resources_test.cpp"
FAKE = ROOT / "tests/core/layer_split_stage_resources_fake"


class StageResourcesTest(unittest.TestCase):
    def check(self, compiler, flags=()):
        if not shutil.which(compiler):
            self.skipTest(f"{compiler} unavailable")
        with tempfile.TemporaryDirectory(prefix="strata-stage-resources-") as temp:
            exe = Path(temp) / "stage-resources"
            build = subprocess.run(
                [compiler, "-std=c++20", "-Wall", "-Wextra", "-Werror",
                 *flags, "-I", str(FAKE), "-I", str(ROOT / "include"),
                 str(TEST), "-o", str(exe)],
                capture_output=True, text=True,
            )
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            cases = ("owner-switch", "owner-same", "empty",
                     "fail-free", "fail-stream", "fail-event")
            for case in cases:
                with self.subTest(compiler=compiler, flags=flags, case=case):
                    env = dict(os.environ, ASAN_OPTIONS="detect_leaks=0:halt_on_error=1",
                               UBSAN_OPTIONS="halt_on_error=1")
                    run = subprocess.run([str(exe), case], capture_output=True,
                                         text=True, timeout=5, env=env)
                    terminal = case.startswith("fail-")
                    self.assertEqual(run.returncode, 1 if terminal else 0, run.stdout + run.stderr)
                    if terminal:
                        self.assertIn("exiting without releasing transfer owners", run.stderr)
                        self.assertNotIn("PASSED", run.stdout)
                    else:
                        self.assertIn("PASSED", run.stdout)
                        self.assertNotIn("FAILED", run.stderr)
                        if case == "empty":
                            self.assertNotIn("FREE ", run.stdout)
                            self.assertNotIn("STREAM ", run.stdout)
                            self.assertNotIn("EVENT ", run.stdout)
                        else:
                            self.assertEqual(run.stdout.count("FREE "), 4)
                            self.assertEqual(run.stdout.count("STREAM "), 2)
                            self.assertEqual(run.stdout.count("EVENT "), 1)

    def test_gcc_release(self):
        self.check("g++", ("-O2", "-DNDEBUG"))

    def test_clang_release(self):
        self.check("clang++", ("-O2", "-DNDEBUG"))

    def test_gcc_sanitized(self):
        self.check("g++", ("-O1", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"))

    def test_clang_sanitized(self):
        self.check("clang++", ("-O1", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer"))


if __name__ == "__main__":
    unittest.main()
