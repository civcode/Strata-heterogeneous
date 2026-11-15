#!/usr/bin/env python3
"""Build a provenance-recorded vanilla oracle from a clean, pinned checkout.

No checkout, fetch, dependency installation or GPU execution is performed here.
Use a separate clean worktree and locally available build dependencies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def identity(source, revision):
    if Path(git(source, "rev-parse", "--show-toplevel")).resolve() != source:
        raise ValueError("source must be the checkout root")
    head = git(source, "rev-parse", "HEAD")
    if head != revision:
        raise ValueError(f"source HEAD {head} does not match pinned revision {revision}")
    if git(source, "status", "--porcelain", "--untracked-files=all", "--ignore-submodules=none"):
        raise ValueError("source checkout is not clean")
    submodules = git(source, "submodule", "status", "--recursive").splitlines()
    # The leading space on an initialized matching first entry may be stripped by git().
    if any(line.startswith(("-", "+", "U")) for line in submodules):
        raise ValueError("submodule not initialized at pinned revision")
    return dict(revision=head, tree=git(source, "rev-parse", "HEAD^{tree}"), submodules=submodules)


def sha256(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command, log):
    with log.open("xb") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)


def build(source, revision, build_dir, binary, target, configure):
    source, build_dir = source.resolve(), build_dir.resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("pin a full lowercase Git commit SHA")
    if build_dir == source or source in build_dir.parents:
        raise ValueError("fresh build directory must be outside source checkout")
    if binary.is_absolute() or ".." in binary.parts:
        raise ValueError("binary must be relative to build directory")
    if not re.fullmatch(r"[A-Za-z0-9_.+-]+", target) or target.startswith("-"):
        raise ValueError("invalid CMake target")
    before = identity(source, revision)
    build_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    configure_cmd = ["cmake", "-S", str(source), "-B", str(build_dir),
                     *configure, "-DFETCHCONTENT_FULLY_DISCONNECTED=ON"]
    build_cmd = ["cmake", "--build", str(build_dir), "--target", target, "--parallel", "28"]
    run(configure_cmd, build_dir / "oracle-configure.log")
    cache = build_dir / "CMakeCache.txt"
    cache_text = cache.read_text(encoding="utf-8")
    if f"CMAKE_HOME_DIRECTORY:INTERNAL={source}" not in cache_text.splitlines():
        raise ValueError("CMake configured a different source")
    run(build_cmd, build_dir / "oracle-build.log")
    if identity(source, revision) != before:
        raise ValueError("source identity changed during build")
    executable = (build_dir / binary).resolve(strict=True)
    if build_dir not in executable.parents or not executable.is_file():
        raise ValueError("binary must be a file produced inside the fresh build")
    manifest = dict(schema="strata-oracle-build/v1", source=str(source), **before,
                    executable=str(executable), executable_sha256=sha256(executable),
                    cmake_version=subprocess.check_output(["cmake", "--version"], text=True).splitlines()[0],
                    python_executable=sys.executable, configure_argv=configure_cmd, build_argv=build_cmd,
                    cache_sha256=sha256(cache), configure_log_sha256=sha256(build_dir / "oracle-configure.log"),
                    build_log_sha256=sha256(build_dir / "oracle-build.log"))
    path = build_dir / "oracle-build.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=Path("strata"))
    parser.add_argument("--target", default="strata")
    parser.add_argument("configure", nargs=argparse.REMAINDER, help="CMake options after --")
    args = parser.parse_args()
    configure = args.configure[1:] if args.configure[:1] == ["--"] else args.configure
    try:
        print(build(args.source, args.revision, args.build_dir, args.binary, args.target, configure))
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"Oracle build failed: {exc}; inspect fresh build logs if present", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
