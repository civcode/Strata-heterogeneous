#!/usr/bin/env python3
"""CPU-only tests of the later-stage deletion policy; no GPU or model required."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
FAKE = ROOT / 'tests/core/layer_split_stage_fake'
TEST = ROOT / 'tests/core/layer_split_stage_owner_test.cpp'


class StageOwner(unittest.TestCase):
    def check_compiler(self, compiler, extra=()):
        if shutil.which(compiler) is None:
            self.skipTest(f'{compiler} unavailable')
        with tempfile.TemporaryDirectory(prefix='strata-stage-owner-') as tmp:
            exe = Path(tmp) / 'stage-owner-test'
            build = subprocess.run([compiler, '-std=c++20', '-Wall', '-Wextra', '-Werror',
                                    '-I', str(FAKE), '-I', str(ROOT / 'include'),
                                    *extra, str(TEST), '-o', str(exe)],
                                   capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            positive = ('owner-switch', 'owner-same', 'no-resources')
            terminal = ('get-fail', 'select-fail', 'sync-fail', 'restore-fail', 'missing-owner')
            for case in (*positive, *terminal, 'negative-plain'):
                with self.subTest(compiler=compiler, flags=extra, case=case):
                    env = dict(os.environ, ASAN_OPTIONS='detect_leaks=0:halt_on_error=1',
                               UBSAN_OPTIONS='halt_on_error=1')
                    result = subprocess.run([str(exe), case], capture_output=True, text=True,
                                            env=env, timeout=5)
                    expected = 0 if case in positive else 91 if case == 'negative-plain' else 1
                    self.assertEqual(result.returncode, expected,
                                     f'{case}: {result.stdout}\n{result.stderr}')
                    if case in positive:
                        self.assertIn('PASSED', result.stdout)
                        self.assertNotIn('WRONG_DEVICE_RELEASE', result.stderr)
                    elif case == 'negative-plain':
                        self.assertIn('WRONG_DEVICE_RELEASE', result.stderr)
                    else:
                        self.assertIn('exiting without releasing transfer owners', result.stderr)
                        if case == 'restore-fail':
                            # Completion and release are safe; only restoring the caller failed.
                            self.assertIn('MEMBER_RELEASE device=1', result.stdout)
                        else:
                            self.assertNotIn('MEMBER_RELEASE', result.stdout)
                    if case == 'owner-switch':
                        self.assertLess(result.stdout.index('DEVICE_SYNC 1'),
                                        result.stdout.index('MEMBER_RELEASE device=1'))
                        self.assertLess(result.stdout.index('MEMBER_RELEASE device=1'),
                                        result.stdout.index('DEVICE_SELECT 0'))
                    elif case == 'owner-same':
                        self.assertNotIn('DEVICE_SELECT', result.stdout)
                    elif case == 'no-resources':
                        self.assertNotIn('DEVICE_SYNC', result.stdout)
                        self.assertNotIn('DEVICE_SELECT', result.stdout)

    def test_gcc(self):
        self.check_compiler('g++')

    def test_clang(self):
        self.check_compiler('clang++')

    def test_gcc_asan_ubsan(self):
        self.check_compiler('g++', ('-fsanitize=address,undefined', '-fno-omit-frame-pointer'))

    def test_clang_asan_ubsan(self):
        self.check_compiler('clang++', ('-fsanitize=address,undefined', '-fno-omit-frame-pointer'))


if __name__ == '__main__':
    unittest.main()
