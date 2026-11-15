import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from heterogeneous_oracle_build import build, git, sha256


class OracleBuildTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        subprocess.run(["git", "init", "-q", str(self.source)], check=True)
        (self.source / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.16)\nproject(test NONE)\n'
            'add_custom_target(strata COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_BINARY_DIR}/strata)\n')
        git(self.source, "add", "CMakeLists.txt")
        git(self.source, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture")
        self.revision = git(self.source, "rev-parse", "HEAD")
        self.output = self.root / "build"

    def build(self, revision=None):
        return build(self.source, revision or self.revision, self.output, Path("strata"), "strata", [])

    def test_fresh_build_manifest(self):
        manifest = json.loads(self.build().read_text())
        self.assertEqual(manifest["revision"], self.revision)
        self.assertEqual(manifest["executable_sha256"], sha256(self.output / "strata"))
        self.assertEqual(manifest["build_argv"][-2:], ["--parallel", "28"])
        self.assertIn("-DFETCHCONTENT_FULLY_DISCONNECTED=ON", manifest["configure_argv"])
        with self.assertRaises(FileExistsError):
            self.build()

    def test_dirty_rejected(self):
        (self.source / "untracked").touch()
        with self.assertRaisesRegex(ValueError, "not clean"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_wrong_revision_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.build("0" * 40)
        self.assertFalse(self.output.exists())

    def test_missing_binary_no_manifest(self):
        with self.assertRaises(FileNotFoundError):
            build(self.source, self.revision, self.output, Path("absent"), "strata", [])
        self.assertFalse((self.output / "oracle-build.json").exists())


if __name__ == "__main__":
    unittest.main()
