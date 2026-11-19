#!/usr/bin/env python3
"""A mixed-split run config and its start script, made from a vanilla run config of the same model.

    python3 mixed_split/tools/make_mixed_config.py VANILLA_CONFIG --build-tag 0091 [--name iq3_xxs-mixed]
        [--front-layers 15] [--port N] [--repo DIR]

VANILLA_CONFIG: the `strata-<model>.json` vanilla's setup wrote (for example ~/workspace/Strata/strata-iq3_xxs.json).
It is only read.  Its model files, pack, drafter, context, control vector, cache settings, port and environment are
kept; what changes:
  - the host engine: this checkout's `build-<tag>-hip/strata`; the front stage: `build-<tag>-cuda/strata`
    (`--front-exe`, `--front-layers K`); the vision helpers: `build-<tag>-vision-hip` / `-vision-cpu`;
  - the options the engine refuses beside the mixed split are left out (`--split-skip-if-fits`, `--batch`, `--slots`,
    `--peer-device`, `--expert-cache-remote`), `"parallel": 1`;
  - `"env"` gets `"HSA_ENABLE_SDMA": "0"` (.ai/PROJECT.md, the first rule);
  - `cwd` is this checkout, the log goes to `runs/<name>/server.log`.
Writes `strata-<name>.json` and `run-<name>.sh` into this checkout.  The script starts the server through
`serve_mixed.py` (the launch check, SIGINT handling, `HSA_ENABLE_SDMA=0`); stop it with Ctrl-C (SIGINT), never kill -9.
The model needs two or more AMD cards in the vanilla config (`"gpu": [0, 1]`).
"""
import argparse, json, os, shlex, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
REFUSED = {"--split-skip-if-fits": 0, "--batch": 1, "--slots": 1, "--peer-device": 1, "--expert-cache-remote": 1}
FRONT = ("--front-exe", "--front-layers", "--front-arg")


def convert(cfg: dict, repo: Path, tag: str, name: str, front_layers: int, port=None) -> tuple[dict, list]:
    """The mixed config, and the notes on what was changed or left out."""
    notes = []
    if cfg.get("backend") != "hip":
        raise SystemExit('make_mixed_config: the vanilla config is not an AMD one ("backend": "hip")')
    gpus = cfg.get("gpu")
    if not (isinstance(gpus, list) and len(gpus) >= 2):
        raise SystemExit('make_mixed_config: the vanilla config runs on one AMD card; the mixed split needs two or more '
                         '("gpu": [0, 1])')
    out = json.loads(json.dumps(cfg))
    hip, cuda = repo / f"build-{tag}-hip" / "strata", repo / f"build-{tag}-cuda" / "strata"
    args, kept, i = [str(x) for x in cfg.get("args", [])], [], 0
    while i < len(args):
        a = args[i]
        if a in FRONT:
            i += 2
            continue
        if a in REFUSED:
            notes.append(f"left out {' '.join(args[i:i + 1 + REFUSED[a]])} (not supported beside the mixed split)")
            i += 1 + REFUSED[a]
            continue
        kept.append(a)
        i += 1
    out["args"] = kept + ["--front-exe", str(cuda), "--front-layers", str(front_layers)]
    out["exe"] = str(hip)
    out["cwd"] = str(repo)
    out["log"] = str(repo / "runs" / name / "server.log")
    out["layer_split"] = cfg.get("layer_split") or "auto"
    env = dict(cfg.get("env") or {})
    if env.get("HSA_ENABLE_SDMA") not in (None, "0"):
        notes.append(f"HSA_ENABLE_SDMA was {env['HSA_ENABLE_SDMA']}: set to 0 (.ai/PROJECT.md, the first rule)")
    env["HSA_ENABLE_SDMA"] = "0"
    out["env"] = env
    if (cfg.get("parallel") or 1) > 1:
        notes.append(f"parallel {cfg['parallel']} -> 1 (one request at a time beside the mixed split)")
    out["parallel"] = 1
    v = out.get("vision")
    if isinstance(v, dict):
        if v.get("exe"):
            v["exe"] = str(repo / f"build-{tag}-vision-hip" / "bin" / "strata-vision-hip") if v.get("gpu") else \
                str(repo / f"build-{tag}-vision-cpu" / "bin" / "strata-vision")
        if v.get("cpu_exe") or v.get("gpu"):
            v["cpu_exe"] = str(repo / f"build-{tag}-vision-cpu" / "bin" / "strata-vision")
    for k in ("args_written", "user_args"):   # setup's own bookkeeping: this file is not one setup wrote
        out.pop(k, None)
    out["mixed_split"] = {"front_exe": str(cuda), "front_layers": front_layers, "made_from": None}
    if port:
        out["port"] = port
    if "qwen3.8" not in str(cfg.get("model_name") or "").lower():
        notes.append("the mixed split is tested with Qwen3.8-Flash-Next only")
    return out, notes


def script(repo: Path, name: str, cfg_path: Path, port) -> str:
    q = shlex.quote
    return f"""#!/bin/sh
# {name}: Strata, mixed split (this checkout's builds), made by mixed_split/tools/make_mixed_config.py.
# Stop it with Ctrl-C (SIGINT) and wait for it to finish; never kill -9 (.ai/PROJECT.md).
cd {q(str(repo))} || exit 1
mkdir -p {q(str(repo / 'runs' / name))}
exec {q(str(repo / '.venv' / 'bin' / 'python'))} mixed_split/tools/serve_mixed.py --config {q(str(cfg_path))} \\
    --run-dir {q(str(repo / 'runs' / name))}{f' --port {int(port)}' if port else ''}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vanilla_config")
    ap.add_argument("--build-tag", required=True, help="the build directories' tag: build-<tag>-hip, -cuda, -vision-hip, -vision-cpu")
    ap.add_argument("--name", help="default: <model>-mixed, from the vanilla file's name")
    ap.add_argument("--front-layers", type=int, default=15, help="layers on the NVIDIA card (15 tested on 16 GB)")
    ap.add_argument("--port", type=int, help="default: the vanilla config's port")
    ap.add_argument("--repo", default=str(REPO), help="this checkout (default: the one this script is in)")
    a = ap.parse_args()
    src = Path(a.vanilla_config).expanduser().resolve()
    cfg = json.loads(src.read_text(encoding="utf-8-sig"))
    repo = Path(a.repo).expanduser().resolve()
    stem = src.stem[len("strata-"):] if src.stem.startswith("strata-") else src.stem
    name = a.name or f"{stem}-mixed"
    if not 2 <= a.front_layers <= 40:
        raise SystemExit("--front-layers takes 2 to 40")
    out, notes = convert(cfg, repo, a.build_tag, name, a.front_layers, a.port)
    out["mixed_split"]["made_from"] = str(src)
    missing = [p for p in (out["exe"], out["mixed_split"]["front_exe"]) if not Path(p).is_file()]
    cfg_path = repo / f"strata-{name}.json"
    cfg_path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    run = repo / f"run-{name}.sh"
    run.write_text(script(repo, name, cfg_path, a.port), encoding="utf-8")
    run.chmod(0o755)
    print(f"wrote {cfg_path}\nwrote {run}")
    for n in notes:
        print("  note:", n)
    for p in missing:
        print(f"  WARNING: {p} does not exist yet (build it first, and stamp BUILD_COMMIT)")
    print(f"start: {run}   (stop: Ctrl-C, then wait)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
