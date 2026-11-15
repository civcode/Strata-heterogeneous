"""CPU-only retirement checks, with no GPU workers or signal delivery."""
import subprocess
import sys
import threading
import unittest
from unittest.mock import Mock, patch

from serve.heterogeneous_retirement import retire_process


class TestRetirement(unittest.TestCase):
    def fixture(self):
        proc = Mock(pid=123)
        proc.wait.return_value = 0
        reader = Mock()
        reader.is_alive.return_value = False
        return proc, reader

    def test_only_verified_absence_allows_retirement(self):
        for code in (0, 1):
            proc, reader = self.fixture()
            proc.wait.return_value = code
            with patch("serve.heterogeneous_retirement.os.killpg", side_effect=ProcessLookupError) as probe:
                self.assertEqual(retire_process(proc, reader, 1), code)
            probe.assert_called_once_with(123, 0)
            reader.join.assert_called_once_with(timeout=1)
            proc.stdout.close.assert_called_once_with()

    def test_still_live_or_unverifiable_group(self):
        for rejection in (None, PermissionError("denied")):
            proc, reader = self.fixture()
            with patch("serve.heterogeneous_retirement.os.killpg", side_effect=rejection) as probe:
                with self.assertRaisesRegex(RuntimeError, "process group"):
                    retire_process(proc, reader, 1)
            probe.assert_called_once_with(123, 0)
            proc.stdout.close.assert_not_called()

    def test_timeout_or_live_reader_never_checks_group(self):
        for wait_timeout in (True, False):
            proc, reader = self.fixture()
            if wait_timeout:
                proc.wait.side_effect = subprocess.TimeoutExpired("coordinator", 1)
            else:
                reader.is_alive.return_value = True
            with patch("serve.heterogeneous_retirement.os.killpg") as probe:
                with self.assertRaisesRegex(RuntimeError, "did not exit"):
                    retire_process(proc, reader, 1)
            probe.assert_not_called()
            proc.stdout.close.assert_not_called()
            if wait_timeout:
                reader.join.assert_not_called()

    def test_real_cooperative_eof(self):
        for code in (0, 7):
            proc = subprocess.Popen([sys.executable, "-c",
                "import sys; sys.stdin.read(); print('done', flush=True); sys.exit(" + str(code) + ")"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, start_new_session=True)
            output = []
            reader = threading.Thread(target=lambda: output.extend(proc.stdout))
            reader.start()
            try:
                proc.stdin.close()
                self.assertEqual(retire_process(proc, reader, 5), code)
                self.assertEqual(output, ["done\n"])
                self.assertTrue(proc.stdout.closed)
            finally:
                if proc.stdin and not proc.stdin.closed:
                    proc.stdin.close()
                proc.wait(timeout=5)
                reader.join(timeout=5)
                proc.stdout.close()


if __name__ == "__main__":
    unittest.main()
