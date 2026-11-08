#!/usr/bin/env python3
"""Opt-in Phase 11B real-model layer-split *startup failure* gate.

A test-only HIP interposer refuses ONE stream creation on logical GPU 1.
A failed stream creation is a supported startup refusal, not a GPU fault.
The intended validation point is the later GpuStage's stream creation after
its weight load. The gate insists the engine actually reaches that branch.

Run ONE UUID ordering per invocation under Oberon's existing read-only GPU
journal/physical-ordinal supervisor. A failed journal check is an abort, not a
reason to run the opposite ordering. Default: dry-run with no filesystem writes.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
MARKER = re.compile(r"(?m)^PHASE11B_FAULT_INJECTED stream-create device=(\d+) skipped=(\d+)\s*$")
REFUSAL = re.compile(r"(?im)^strata generate: layer split, CUDA(\d+): its streams failed\s*$")
READY = re.compile(r"(?m)^READY(?:\s|$)")


class GateFailure(RuntimeError):
    pass


def require(ok: bool, why: str) -> None:
    if not ok:
        raise GateFailure(why)


def verify(exit_code: int, log: str, *, target: int, skip: int) -> None:
    """Negative and positive controls: exit alone never proves injected coverage."""
    markers = MARKER.findall(log)
    require(len(markers) == 1, "missing or multiple test-only injection markers")
    require(markers == [(str(target), str(skip))], "injection hit the wrong GPU or call")
    refused = list(REFUSAL.finditer(log))
    require(any(m.group(1) == str(target) for m in refused),
            "later-stage stream-creation refusal was not observed")
    require(log.index("PHASE11B_FAULT_INJECTED") < refused[0].start(),
            "refusal preceded fault injection marker")
    require(exit_code == 1, f"expected startup refusal exit 1, got {exit_code} (crash or wrong status)")
    require(not READY.search(log), "engine announced READY despite refused startup")


def command_from_config(config: dict, engine: Path, uuids: tuple[str, str]) -> tuple[list[str], dict, str | None]:
    """Use the same production Strata engine arguments as the validated A/B/A gate."""
    sys.path.insert(0, str(ROOT))
    from serve.server import child_env, engine_args
    c = copy.deepcopy(config)
    c["gpu"] = [0, 1]
    c["gpu_order"] = "as_given"
    c["layer_split"] = "auto"
    # No profile writes or new model assets. No forced VRAM overcommit.
    args = engine_args(c)
    require("--layer-split" in args, "layer split was not enabled by engine_args")
    env = child_env(c)
    env["ROCR_VISIBLE_DEVICES"] = ",".join(uuids)
    env["HSA_ENABLE_SDMA"] = "0"
    env["HIP_PLATFORM"] = "amd"
    # One run, no fallback/retry, bounded by this tool's external deadline.
    env["STRATA_START_RETRIES"] = "0"
    return [str(engine), "--serve", *args], env, c.get("cwd")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_owned(command: list[str], *, env: dict, cwd: str | None, log_path: Path,
              timeout_s: float) -> tuple[int, float]:
    """Own one child process group; on timeout kill ONLY that owned group."""
    t0 = time.monotonic()
    with log_path.open("x", encoding="utf-8") as log:
        proc = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            exit_code = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            # The new session guarantees this group was created for our child.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            raise GateFailure(f"startup exceeded {timeout_s:g}s; owned child group terminated") from exc
    return exit_code, time.monotonic() - t0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--engine", required=True, type=Path)
    ap.add_argument("--interposer", required=True, type=Path,
                    help="test-only .so compiled from tests/hip/phase11b_stream_create_fault.cpp")
    ap.add_argument("--output", required=True, type=Path, help="new evidence directory, never overwritten")
    ap.add_argument("--uuid-first", required=True)
    ap.add_argument("--uuid-second", required=True)
    ap.add_argument("--skip", type=int, default=0, help="matching device-1 stream creations to forward before injected failure")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--run", action="store_true", help="opt-in: starts one real-model HIP engine")
    a = ap.parse_args(argv)
    require(a.uuid_first != a.uuid_second and
            all(u.startswith("GPU-") for u in (a.uuid_first, a.uuid_second)),
            "two distinct R9700 UUIDs required")
    require(0 <= a.skip <= 16, "skip must be 0..16")
    require(30 <= a.timeout <= 300, "timeout must be 30..300 seconds")
    require(not a.output.exists(), "refusing to overwrite existing evidence")
    if not a.run:
        print("DRY RUN: one HIP startup, one injected stream-create failure on logical GPU1.")
        print("Requires external physical-device mapping and kernel-journal supervisor.")
        print("No files created and no GPU operations; --run explicitly enables the negative test.")
        return 0
    require(a.config.is_file() and a.engine.is_file() and a.interposer.is_file(),
            "config, engine or test-only interposer missing")
    config = json.loads(a.config.read_text(encoding="utf-8"))
    require(config.get("backend") in (None, "hip") and isinstance(config.get("args"), list),
            "this gate requires an existing HIP config with engine args")
    uuids = (a.uuid_first, a.uuid_second)
    command, env, cwd = command_from_config(config, a.engine, uuids)
    # No change to existing production environment injection: fail instead of
    # silently replacing another LD_PRELOAD owner/instrumentation interposer.
    require(not env.get("LD_PRELOAD"), "existing LD_PRELOAD: do not replace mapping or other hooks")
    env["LD_PRELOAD"] = str(a.interposer.resolve())
    env["STRATA_PHASE11B_FAULT_ENABLE"] = "stream-create-once"
    env["STRATA_PHASE11B_FAULT_DEVICE"] = "1"
    env["STRATA_PHASE11B_FAULT_SKIP"] = str(a.skip)
    a.output.mkdir(mode=0o700, parents=False, exist_ok=False)
    log_path = a.output / "startup-failure.log"
    result = {"phase": "11B", "scenario": "later-stage startup stream creation failure",
              "status": "unverified", "uuid_order": list(uuids), "target_device": 1,
              "skip": a.skip, "timeout_s": a.timeout,
              "command": command, "config_sha256": sha(a.config),
              "executable_sha256": sha(a.engine), "interposer_sha256": sha(a.interposer),
              "log": str(log_path)}
    try:
        rc, elapsed = run_owned(command, env=env, cwd=cwd, log_path=log_path, timeout_s=a.timeout)
        result.update(exit_code=rc, elapsed_s=elapsed)
        log = log_path.read_text(encoding="utf-8", errors="replace")
        verify(rc, log, target=1, skip=a.skip)
        result["status"] = "pass"
    except Exception as exc:
        result.update(status="fail", error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())
        raise
    finally:
        (a.output / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("PASS: later-stage startup refused injected HIP stream creation; no READY; owned child exited.")
    print(f"Evidence: {a.output}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (GateFailure, OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
