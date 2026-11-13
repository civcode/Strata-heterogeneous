#!/usr/bin/env python3
"""Manual, non-merging upstream divergence and impact report for Strata-heterogeneous.

No GitHub Actions, cherry-pick, merge, rebase, reset, or branch push is invoked.
The optional --fetch updates only refs/remotes/upstream/main after a remote
URL check. Without --fetch this tool does not modify the repository.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

EXPECTED_URLS = {
    "https://github.com/Niko1221/Strata",
    "https://github.com/Niko1221/Strata.git",
    "git@github.com:Niko1221/Strata.git",
    "ssh://git@github.com/Niko1221/Strata.git",
}


def git(repo: Path, *args: str, binary: bool = False):
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True)
    if result.returncode:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout if binary else result.stdout.decode("utf-8", "replace").strip()


def category(path: str) -> str:
    if path.startswith(".github/workflows/"):
        return "workflow-policy (never import automatically)"
    if path == "CMakeLists.txt" or path == "CMakePresets.json" or path.startswith("cmake/"):
        return "build-integration"
    if path.startswith("experimental/heterogeneous/"):
        return "heterogeneous-runtime"
    if path.startswith(("include/strata/", "src/", "cuda/", "hip/")):
        return "engine-runtime"
    if path.startswith(("tests/", "tools/")):
        return "tests-and-tooling"
    return "other"


def changed_files(repo: Path, start: str, end: str):
    # --no-renames makes each record exactly (status, path) with -z.
    parts = git(repo, "diff", "--name-status", "--no-renames", "-z", start, end,
                "--", binary=True).split(b"\0")
    if parts[-1:] == [b""]:
        parts.pop()
    if len(parts) % 2:
        raise RuntimeError("unexpected git diff --name-status output")
    return [{"status": parts[i].decode("ascii"),
             "path": parts[i + 1].decode("utf-8", "surrogateescape"),
             "category": category(parts[i + 1].decode("utf-8", "surrogateescape"))}
            for i in range(0, len(parts), 2)]


def report(repo: Path, upstream_ref: str, snapshot_path: Path, all_upstream: bool):
    head = git(repo, "rev-parse", "--verify", "HEAD^{commit}")
    upstream = git(repo, "rev-parse", "--verify", upstream_ref + "^{commit}")
    ancestor = git(repo, "merge-base", head, upstream)
    # rev-list order: left-only HEAD, right-only upstream.
    counts = git(repo, "rev-list", "--left-right", "--count", f"{head}...{upstream}").split()
    if len(counts) != 2:
        raise RuntimeError("cannot determine ahead/behind counts")
    since = ancestor
    if snapshot_path.is_file() and not all_upstream:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if snapshot.get("upstream_repository") != "Niko1221/Strata":
            raise RuntimeError("upstream checkpoint has an unexpected repository")
        since = snapshot["observed_upstream_sha"]
        if subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor",
                           since, upstream], capture_output=True).returncode:
            raise RuntimeError("checkpoint is not an ancestor of upstream; inspect manually")
    files = changed_files(repo, since, upstream)
    return {
        "working_head": head,
        "upstream_head": upstream,
        "merge_base": ancestor,
        "working_only_commits": int(counts[0]),
        "upstream_only_commits": int(counts[1]),
        "change_range_start": since,
        "change_range_end": upstream,
        "changed_files": files,
        "notice": "Read-only comparison; no upstream commits integrated. Check workflow triggers before copying any workflow file.",
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path.cwd())
    p.add_argument("--remote", default="upstream")
    p.add_argument("--fetch", action="store_true", help="explicitly fetch upstream/main; never merge")
    p.add_argument("--all-upstream", action="store_true", help="list changes since merge-base, not last observed snapshot")
    p.add_argument("--json", action="store_true", help="machine-readable report")
    p.add_argument("--max-files", type=int, default=50)
    opts = p.parse_args(argv)
    try:
        repo = Path(git(opts.repo, "rev-parse", "--show-toplevel"))
        if opts.max_files < 0:
            raise ValueError("--max-files must be nonnegative")
        remote_url = git(repo, "remote", "get-url", opts.remote)
        if remote_url not in EXPECTED_URLS:
            raise RuntimeError(f"{opts.remote} points to an unexpected repository: {remote_url}")
        if opts.fetch:
            git(repo, "fetch", "--no-tags", opts.remote,
                f"refs/heads/main:refs/remotes/{opts.remote}/main")
        upstream_ref = f"refs/remotes/{opts.remote}/main"
        state = report(repo, upstream_ref, repo / ".ai/upstream/checkpoint.json", opts.all_upstream)
        state["upstream_remote_url"] = remote_url
    except (RuntimeError, ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        print(f"upstream-sync: {exc}", file=sys.stderr)
        return 2
    if opts.json:
        print(json.dumps(state, indent=2, ensure_ascii=True))
    else:
        print("Strata heterogeneous upstream comparison (no integration performed)")
        for key in ("working_head", "upstream_head", "merge_base",
                    "working_only_commits", "upstream_only_commits",
                    "change_range_start", "change_range_end"):
            print(f"{key}: {state[key]}")
        files = state["changed_files"]
        print(f"Changed upstream paths in selected range: {len(files)}")
        for item in files[:opts.max_files]:
            print(f"  {item['status']:2} [{item['category']}] {item['path']}")
        if len(files) > opts.max_files:
            print(f"  ... {len(files) - opts.max_files} more (use --json to see all)")
        print(state["notice"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
