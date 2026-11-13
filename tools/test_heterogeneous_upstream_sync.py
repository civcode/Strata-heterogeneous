#!/usr/bin/env python3
"""CPU-only local-git contract tests for the manual upstream comparison tool."""
import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import heterogeneous_upstream_sync as sync


def run(repo, *args):
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr)
    return p.stdout.strip()


class SyncTests(unittest.TestCase):
    def test_classification(self):
        self.assertEqual(sync.category(".github/workflows/check.yml").split(" ")[0], "workflow-policy")
        self.assertEqual(sync.category("cmake/backends/hip.cmake"), "build-integration")
        self.assertEqual(sync.category("include/strata/core/verify.hpp"), "engine-runtime")
        self.assertEqual(sync.category("experimental/heterogeneous/README.md"), "heterogeneous-runtime")

    def test_branch_divergence_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run(root, "init", "-q", "-b", "main")
            run(root, "config", "user.name", "Test")
            run(root, "config", "user.email", "test@example.invalid")
            (root / "base.txt").write_text("base\n")
            run(root, "add", "base.txt")
            run(root, "commit", "-qm", "base")
            run(root, "checkout", "-qb", "feature")
            (root / "feature.txt").write_text("ours\n")
            run(root, "add", "feature.txt")
            run(root, "commit", "-qm", "feature")
            run(root, "checkout", "-q", "main")
            (root / "cmake").mkdir()
            (root / "cmake/backends.cmake").write_text("upstream\n")
            run(root, "add", "cmake/backends.cmake")
            run(root, "commit", "-qm", "upstream")
            upstream_head = run(root, "rev-parse", "HEAD")
            run(root, "update-ref", "refs/remotes/upstream/main", upstream_head)
            run(root, "remote", "add", "upstream", "https://github.com/Niko1221/Strata.git")
            run(root, "checkout", "-q", "feature")
            state = sync.report(root, "refs/remotes/upstream/main", root / "no-checkpoint.json", False)
            self.assertEqual((state["working_only_commits"], state["upstream_only_commits"]), (1, 1))
            self.assertEqual(state["changed_files"][0]["path"], "cmake/backends.cmake")
            self.assertEqual(state["changed_files"][0]["category"], "build-integration")
            snapshot = root / "checkpoint.json"
            snapshot.write_text(json.dumps({"upstream_repository": "Niko1221/Strata",
                                            "observed_upstream_sha": upstream_head}))
            state = sync.report(root, "refs/remotes/upstream/main", snapshot, False)
            self.assertEqual(state["changed_files"], [])
            self.assertEqual(len(sync.report(root, "refs/remotes/upstream/main", snapshot, True)["changed_files"]), 1)
            before = run(root, "status", "--porcelain")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(sync.main(["--repo", str(root), "--json"]), 0)
            self.assertEqual(json.loads(stdout.getvalue())["upstream_head"], upstream_head)
            self.assertEqual(before, run(root, "status", "--porcelain"))

    def test_reject_wrong_remote(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run(root, "init", "-q")
            run(root, "remote", "add", "upstream", "https://example.com/not-strata.git")
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(sync.main(["--repo", str(root), "--fetch"]), 2)


if __name__ == "__main__":
    unittest.main()
