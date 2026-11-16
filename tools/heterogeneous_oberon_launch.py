"""Bind checked-in Q2_0 HTTP launch profiles to Oberon's qualified local files."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from strata_control.server_launch import server_command
from strata_control.mixed_runconfig import resolve, bind_standard_fields

MIXED = "cuda:0:0-15,hip:0:15-33,hip:1:33-48"
HIP = "hip:0:0-24,hip:1:24-48"
QUALIFIED = {
    MIXED: "ed360799dbee90d65aec1b34661a95ae1e95288f7fddbfbf004d6d158c4d5ef1",
    HIP: "70c96e56db5c35be1a5ab3761a5396c4e74066ebdba739fed4b6314d83fa3907",
}
REQUIRED_SETTINGS = dict(fingerprint=2026111301, max_context=262144, kv="int8",
                         shared=True, stream_tokens=True, group_native_hip=True,
                         workload_mode=True, long_context=True)
REFERENCE_PREFILL_CHUNK = 128
MTP_MAX_PREFILL_CHUNK = 64
MTP_MAX_NEW_TOKENS = 256
MAX_CONFIG_BYTES = 65536


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    with path.open("rb") as stream:
        raw = stream.read(MAX_CONFIG_BYTES + 1)
    need(len(raw) <= MAX_CONFIG_BYTES, f"config exceeds {MAX_CONFIG_BYTES} bytes: {path}")
    result = json.loads(raw)
    need(isinstance(result, dict), f"expected JSON object: {path}")
    return result


def find_reference(root, digest):
    """Find the previously recorded config, not the newest arbitrary JSON file."""
    need(root.is_dir(), f"qualification directory missing: {root}; use --reference-config")
    for path in sorted(root.rglob("*.json")):
        if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_CONFIG_BYTES:
            continue
        with path.open("rb") as stream:
            raw = stream.read(MAX_CONFIG_BYTES + 1)
        if hashlib.sha256(raw).hexdigest() == digest:
            return path
    raise ValueError("recorded qualification config not found; supply --reference-config with the matching local server JSON")


def file_path(value, executable=False):
    need(isinstance(value, str) and Path(value).is_absolute(), "reference paths must be absolute")
    path = Path(value).resolve(strict=True)
    need(path.is_file() and (not executable or os.access(path, os.X_OK)), f"unusable local file: {path}")
    return str(path)


def bind(profile, reference):
    cfg = copy.deepcopy(profile)
    public_opts = cfg.get("heterogeneous")
    opts = resolve(cfg)
    source = reference.get("heterogeneous")
    need(isinstance(opts, dict) and isinstance(source, dict), "expected heterogeneous profile and reference")
    plan = opts.get("plan")
    need(plan in QUALIFIED and source.get("plan") == plan, "reference topology differs from launch profile")
    need(cfg.get("host") == "127.0.0.1" and cfg.get("max_queue") == 0, "launch profile must be loopback and single-request")
    for key, value in REQUIRED_SETTINGS.items():
        need(type(opts.get(key)) is type(value) and opts[key] == value and
             type(source.get(key)) is type(value) and source[key] == value,
             f"reference or profile differs from qualified settings: {key}")
    for values in (source, opts):
        need(values.get("lookup_draft", False) is False and values.get("restart_clean", False) is False,
             "lookup and automatic restart are not supported by these profiles")
        need(type(values.get("prompt_chunk", 1)) is int and values.get("prompt_chunk", 1) == 1,
             "expected qualified verifier prompt chunk 1")
    # The recorded 0044 reference qualifies the ordinary target stage layout
    # and supplies its local file paths, not the MTP request policy. Keep its
    # original settings gate separate from the opt-in MTP profile.
    need(type(source.get("native_prefill_chunk")) is int and
         source["native_prefill_chunk"] == REFERENCE_PREFILL_CHUNK,
         "reference differs from qualified settings: native_prefill_chunk")
    need(all(source.get(key) is None for key in ("mtp", "spec", "spec_min_p")),
         "expected ordinary 0044 target reference, not an MTP reference")
    bulk = opts.get("native_prefill_chunk")
    budget = opts.get("max_new_tokens")
    mtp = opts.get("mtp")
    if mtp is None:
        need(opts.get("spec") is None and opts.get("spec_min_p") is None,
             "spec and spec_min_p require explicit mtp runtime")
        need(type(bulk) is int and bulk == REFERENCE_PREFILL_CHUNK,
             "profile differs from qualified settings: native_prefill_chunk")
        need(type(budget) is int and 1 <= budget <= 4096, "max_new_tokens must be 1..4096")
    else:
        # prepare_server checks the real runtime, capabilities, source and
        # exact worker capacity. Do not claim full-context MTP qualification.
        need(isinstance(mtp, str) and bool(mtp) and Path(mtp).is_absolute(),
             "MTP runtime must be an absolute local directory path")
        need(type(opts.get("spec")) is int and opts["spec"] == 4 and
             type(opts.get("spec_min_p")) in (int, float) and opts["spec_min_p"] == 0.5,
             "MTP requires spec=4 and spec_min_p=0.5")
        need(type(bulk) is int and 1 <= bulk <= MTP_MAX_PREFILL_CHUNK,
             "MTP native_prefill_chunk must be 1..64")
        need(type(budget) is int and 1 <= budget <= MTP_MAX_NEW_TOKENS,
             "MTP max_new_tokens must be 1..256")
        need(opts.get("pipeline_mode") is None, "MTP does not support explicit pipeline_mode")
    keys = ("coordinator", "hip_worker") + (("cuda_worker",) if plan == MIXED else ())
    need(not ({"coordinator", "hip_worker", "cuda_worker", "config"} & opts.keys()),
         "binary and worker config paths are bound from the reference, not the profile")
    for key in keys:
        path = file_path(source.get(key), executable=True)
        if "args" in cfg and key == "coordinator":
            cfg["exe"] = path
        else:
            public_opts[key] = path
    stages = source.get("config")
    need(isinstance(stages, list) and len(stages) == (3 if plan == MIXED else 2), "wrong number of stage configs")
    public_opts["config"] = [file_path(p) for p in stages]
    need(len(set(public_opts["config"])) == len(stages), "duplicate stage configs")
    return cfg


def write_config(root, cfg):
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory = Path(tempfile.mkdtemp(prefix="server-", dir=root))
    path = directory / "server.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(cfg, stream, indent=2)
        stream.write("\n")
    path.chmod(0o600)
    return path


def guard_engine():
    # Linux pgrep -x sees the short vanilla comm; -f also catches native
    # heterogeneous executable names, which exceed Linux's comm limit.
    patterns = [("-x", "strata"), ("-f", r"(^|[[:space:]/])strata_heterogeneous_(live_smoke|cuda_worker|hip_worker)([[:space:]]|$)"),
                ("-f", r"(serve/server\.py|serve\.server).*--engine[[:space:]]+(strata|heterogeneous)([[:space:]]|$)")]
    for option, pattern in patterns:
        status = subprocess.run(["pgrep", option, pattern], stdout=subprocess.DEVNULL, check=False).returncode
        need(status == 1, "another Strata process may be running, or process guard failed; inspect manually")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--reference-config", type=Path, help="explicit matching local server JSON instead of the 0044 reference")
    p.add_argument("--port", type=int)
    p.add_argument("--open", action=argparse.BooleanOptionalAction, default=True,
                   help="open the native chat page on readiness (default); --no-open for headless runs")
    p.add_argument("--prepare-only", action="store_true", help="bind and run CPU preflight; do not start GPU workers")
    a = p.parse_args()
    os.umask(0o077)
    root = Path.home() / "tmp/strata-heterogeneous"
    try:
        profile = read_json(a.profile)
        # Same normalization as the native server, before binding.
        from serve.server import split_joined_args
        split_joined_args(profile)
        port = a.port if a.port is not None else profile.get("port", 8081)
        need(type(port) is int and 1 <= port <= 65535, "port must be 1..65535")
        plan = profile["heterogeneous"]["plan"]
        need(plan in QUALIFIED, "unknown launch topology")
        reference = a.reference_config or find_reference(root / "handoff-0044", QUALIFIED[plan])
        cfg = bind(profile, read_json(reference))
        if a.prepare_only:
            from serve.heterogeneous import prepare_server
            from serve.server import child_env
            _, _, source, _, _, _ = prepare_server(cfg, env=child_env(cfg, offline=True, mixed_runtime=True))
            bind_standard_fields(cfg, source)
            print(write_config(root / "launchers", cfg))
            return 0
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Shared lock for both launchers, retained across exec for the server's
        # lifetime. The process guard is additional, not a GPU-wide mutex.
        with (root / "oberon-launch.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            guard_engine()
            # Ordinary runs keep worker diagnostics in the native-style log.
            # Existing explicit profiling runs retain their terminal capture
            # unless the operator supplied a log path.
            if os.environ.get("STRATA_HETERO_PROFILE") != "1":
                cfg.setdefault("log", str(root / (a.profile.stem + ".log")))
            cfg["port"] = port
            if "args" in cfg:
                from serve.heterogeneous import prepare_server
                from serve.server import child_env
                _, _, source, _, _, _ = prepare_server(cfg, env=child_env(cfg, offline=True, mixed_runtime=True))
                bind_standard_fields(cfg, source)
            path = write_config(root / "launchers", cfg)
            os.set_inheritable(lock.fileno(), True)
            cmd = server_command(Path(__file__).resolve().parents[1], path, port, a.open)
            os.execv(sys.executable, [*cmd, "--offline"])
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
