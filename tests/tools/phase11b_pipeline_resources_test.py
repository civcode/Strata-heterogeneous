#!/usr/bin/env python3
"""CPU fake GPU lifetime gate for pipeline and refill owner registry."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
FAKE = ROOT / "tests/core/phase11b_pipeline_fake"
TEST = ROOT / "tests/core/phase11b_pipeline_resources_test.cpp"


class PipelineResourcesTest(unittest.TestCase):
    def check_compiler(self, compiler, flags):
        if not shutil.which(compiler):
            self.skipTest(f"{compiler} unavailable")
        with tempfile.TemporaryDirectory(prefix="strata-pipeline-owners-") as temp:
            executable = Path(temp) / "pipeline-resources"
            build = subprocess.run(
                [compiler, "-std=c++20", "-Wall", "-Wextra", "-Werror", *flags,
                 "-I", str(FAKE), "-I", str(ROOT / "include"),
                 str(TEST), "-o", str(executable)],
                capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            modes = ("empty", "partial", "normal", "fail-get", "fail-set",
                     "fail-sync", "fail-sync-second", "fail-free", "fail-stream", "fail-event",
                     "fail-restore", "fail-track", "fail-invalid", "fail-overflow")
            for mode in modes:
                with self.subTest(compiler=compiler, flags=flags, mode=mode):
                    env = dict(
                        os.environ,
                        ASAN_OPTIONS="detect_leaks=0:halt_on_error=1"
                        if mode.startswith("fail-") else "detect_leaks=1:halt_on_error=1",
                        UBSAN_OPTIONS="halt_on_error=1",
                    )
                    run = subprocess.run([str(executable), mode], env=env,
                                         capture_output=True, text=True, timeout=5)
                    terminal = mode.startswith("fail-")
                    self.assertEqual(run.returncode, 1 if terminal else 0,
                                     f"{mode}: {run.stdout}\n{run.stderr}")
                    if terminal:
                        self.assertIn("exiting without releasing transfer owners", run.stderr)
                        self.assertNotIn("PASSED", run.stdout)
                    else:
                        self.assertIn("PASSED", run.stdout)
                        self.assertNotIn("ASSERTION_FAILED", run.stderr)

    def test_gcc_release(self):
        self.check_compiler("g++", ("-O2", "-DNDEBUG"))

    def test_clang_release(self):
        self.check_compiler("clang++", ("-O2", "-DNDEBUG"))

    def test_gcc_sanitized(self):
        self.check_compiler(
            "g++", ("-O1", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer")
        )

    def test_clang_sanitized(self):
        self.check_compiler(
            "clang++", ("-O1", "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer")
        )


if __name__ == "__main__":
    unittest.main()
