#!/usr/bin/env python3
"""A mixed-split run config and start script, in vanilla's own form, made from a vanilla run config of the same model.

    .venv/bin/python mixed_split/tools/make_mixed_config.py VANILLA_CONFIG --build-tag 0092 [--name iq3_xxs-mixed]
        [--front-layers 15] [--port N]

VANILLA_CONFIG: the `strata-<model>.json` vanilla's setup wrote (for example ~/workspace/Strata/strata-iq3_xxs.json).
It is only read.  The two files written into this checkout have exactly vanilla's structure:
  - `strata-<name>.json`: the vanilla config's keys and nothing else (setup's `args_written` / `user_args` bookkeeping
    left out, as in a config written by hand).  What changes, all through vanilla's own keys:
      "exe": this checkout's `build-<tag>-hip/strata`;
      "args": `--front-exe <build-<tag>-cuda/strata> --front-layers K` added (the engine's options for the mixed split);
              the options the engine refuses beside it left out (`--split-skip-if-fits`, `--batch`, `--slots`,
              `--peer-device`, `--expert-cache-remote`); `--mmap-experts` added when the vanilla config has neither it
              nor `--shared-expert-arena` (without one of them the host and the front stage would each hold a full copy
              of the experts in RAM);
      "env": `"HSA_ENABLE_SDMA": "0"` (.ai/PROJECT.md, the first rule);
      "vision": this checkout's `build-<tag>-vision-hip` / `-vision-cpu` helpers;
      "cwd": this checkout; "log": `strata-<name>.log` here (vanilla's place); "parallel": 1.
  - `run-<name>.sh`: vanilla's `write_run_script` form (`cd` here, `exec` the venv's python on `serve/server.py
    --engine strata --config ... --port ...`, `--open` unless the config says `"open_browser": false`).
The launch check (`check_launch.py`) runs on the result before anything is written; a config that fails it is not
written.  Stop the server with Ctrl-C and wait for it to drain; never kill -9.
"""
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
REFUSED = {"--split-skip-if-fits": 0, "--batch": 1, "--slots": 1, "--peer-device": 1, "--expert-cache-remote": 1}
FRONT = ("--front-exe", "--front-layers", "--front-arg")
SETUP_BOOKKEEPING = ("args_written", "user_args")


def convert(cfg: dict, repo: Path, tag: str, name: str, front_layers: int, port=None) -> tuple[dict, list]:
    """The mixed config (vanilla's keys only), and notes on what was changed or left out."""
    notes = []
    if cfg.get("backend") != "hip":
        raise SystemExit('make_mixed_config: the vanilla config is not an AMD one ("backend": "hip")')
    gpus = cfg.get("gpu")
    if not (isinstance(gpus, list) and len(gpus) >= 2):
        raise SystemExit('make_mixed_config: the vanilla config runs on one AMD card; the mixed split needs two or more '
                         '("gpu": [0, 1])')
    out = json.loads(json.dumps(cfg))
    for k in SETUP_BOOKKEEPING:
        out.pop(k, None)
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
    if "--mmap-experts" not in kept and "--shared-expert-arena" not in kept:
        kept.append("--mmap-experts")
        notes.append("added --mmap-experts (otherwise the host and the front stage would each load the experts into RAM)")
    out["args"] = kept + ["--front-exe", str(cuda), "--front-layers", str(front_layers)]
    out["exe"] = str(hip)
    out["cwd"] = str(repo)
    out["log"] = str(repo / f"strata-{name}.log")
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
    if port:
        out["port"] = port
    if "qwen3.8" not in str(cfg.get("model_name") or "").lower():
        notes.append("the mixed split is tested with Qwen3.8-Flash-Next only")
    return out, notes


def run_script(repo: Path, cfg_path: Path, port: int, open_browser: bool) -> str:
    """vanilla setup.py's write_run_script, Linux form."""
    serve = [str(repo / ".venv" / "bin" / "python"), str(repo / "serve" / "server.py"), "--engine", "strata", "--config",
             str(cfg_path), "--port", str(port)] + (["--open"] if open_browser else [])
    return "#!/bin/sh\ncd \"" + str(repo) + "\"\nexec " + " ".join(f'"{x}"' for x in serve) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vanilla_config")
    ap.add_argument("--build-tag", required=True, help="the build directories' tag: build-<tag>-hip, -cuda, -vision-hip, -vision-cpu")
    ap.add_argument("--name", help="default: <model>-mixed, from the vanilla file's name")
    ap.add_argument("--front-layers", type=int, default=15, help="layers on the NVIDIA card (15 tested on 16 GB)")
    ap.add_argument("--port", type=int, help="default: the vanilla config's port")
    ap.add_argument("--no-check", action="store_true", help="write without the launch check (only to inspect the files)")
    a = ap.parse_args()
    src = Path(a.vanilla_config).expanduser().resolve()
    cfg = json.loads(src.read_text(encoding="utf-8-sig"))
    stem = src.stem[len("strata-"):] if src.stem.startswith("strata-") else src.stem
    name = (a.name or f"{stem}-mixed").lower()
    if not 2 <= a.front_layers <= 40:
        raise SystemExit("--front-layers takes 2 to 40")
    out, notes = convert(cfg, REPO, a.build_tag, name, a.front_layers, a.port)
    if not a.no_check:
        from check_launch import check_config
        try:
            print("launch check OK:", check_config(out))
        except (ValueError, OSError) as e:
            raise SystemExit(f"launch check failed, nothing written: {e}") from e
    cfg_path = REPO / f"strata-{name}.json"
    cfg_path.write_text(json.dumps(out, indent=1), encoding="utf-8")       # (vanilla's write_config form)
    run = REPO / f"run-{name}.sh"
    run.write_text(run_script(REPO, cfg_path, int(out.get("port") or 8080), out.get("open_browser") is not False),
                   encoding="utf-8")
    run.chmod(0o755)
    print(f"wrote {cfg_path}\nwrote {run}")
    for n in notes:
        print("  note:", n)
    print(f"start: {run}   (stop: Ctrl-C, then wait for the drain)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
