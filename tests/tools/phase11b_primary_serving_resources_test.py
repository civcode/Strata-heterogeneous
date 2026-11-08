#!/usr/bin/env python3
"""CPU fake-device regression: primary serve-local allocation ownership."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / 'tests/core/primary_serving_resources_test.cpp'
FAKE = ROOT / 'tests/core/primary_serving_fake'

class PrimaryServingTest(unittest.TestCase):
    def run_compiler(self, compiler, flags=()):
        if not shutil.which(compiler):
            self.skipTest(f'{compiler} unavailable')
        with tempfile.TemporaryDirectory(prefix='strata-primary-serving-') as temp:
            executable = Path(temp) / 'resource-test'
            p = subprocess.run([compiler, '-std=c++20','-Wall','-Wextra','-Werror', *flags,
                                '-I',str(FAKE),'-I',str(ROOT / 'include'),str(TEST),'-o',str(executable)],
                               capture_output=True,text=True,timeout=60)
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)
            modes = ('normal','same','partial','empty','fail-get','fail-select','fail-sync','fail-free','fail-event','fail-restore')
            for mode in modes:
                with self.subTest(compiler=compiler, flags=flags, mode=mode):
                    env=dict(os.environ, ASAN_OPTIONS='detect_leaks=0:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1')
                    r=subprocess.run([str(executable),mode],capture_output=True,text=True,env=env,timeout=5)
                    failure=mode.startswith('fail-')
                    self.assertEqual(r.returncode,1 if failure else 0,f'{mode}:\n{r.stdout}\n{r.stderr}')
                    if failure:
                        self.assertIn('exiting without releasing transfer owners',r.stderr)
                        self.assertNotIn('PASSED',r.stdout)
                    else:
                        self.assertIn('PASSED',r.stdout)
                        self.assertNotIn('ASSERTION_FAILED',r.stderr)
                        if mode=='normal':
                            self.assertEqual(r.stdout.count('FREE '),12)
                            self.assertEqual(r.stdout.count('EVENT '),1)
    def test_gcc_release(self): self.run_compiler('g++',('-O2','-DNDEBUG'))
    def test_clang_release(self): self.run_compiler('clang++',('-O2','-DNDEBUG'))
    def test_gcc_sanitized(self): self.run_compiler('g++',('-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer'))
    def test_clang_sanitized(self): self.run_compiler('clang++',('-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer'))

if __name__=='__main__': unittest.main()
