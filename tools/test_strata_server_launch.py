"""CPU-only regression for the shared vanilla run-script command."""
from pathlib import Path
import sys
import unittest

from strata_control.server_launch import server_command


class ServerLaunchTests(unittest.TestCase):
    def test_native_run_script(self):
        root = Path("/fixture/Strata")
        cfg = root / "strata-iq3_xxs.json"
        expected = [sys.executable, str(root / "serve" / "server.py"), "--engine", "strata",
                    "--config", str(cfg), "--port", "8081"]
        self.assertEqual(server_command(root, cfg, 8081), expected)
        self.assertEqual(server_command(root, cfg, 8081, True), expected + ["--open"])

    def test_path_with_spaces_is_one_argument(self):
        root = Path("/fixture/my Strata")
        cfg = root / "my profile.json"
        argv = server_command(root, cfg, 8081, True)
        self.assertEqual(argv[1], str(root / "serve" / "server.py"))
        self.assertEqual(argv[5], str(cfg))
        self.assertEqual(len(argv), 9)


if __name__ == "__main__":
    unittest.main()
