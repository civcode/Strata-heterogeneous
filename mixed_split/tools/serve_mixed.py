#!/usr/bin/env python3
"""Start serve/server.py with one of the example run configs in mixed_split/configs/.

    .venv/bin/python mixed_split/tools/serve_mixed.py --config mixed_split/configs/mixed.example.json [--port N]
        [--set NAME=VALUE ...] [--run-dir DIR] [--no-shared-arena] [-- extra server.py arguments]

The config's ${NAME} placeholders are filled from --set, then the environment (mixed_split/configs/README.md lists them;
STRATA_REPO and RUN_DIR default to this checkout and its runs/ directory).  The filled config is written to RUN_DIR, so
the log and every other path the server writes go there, never into another checkout.

A mixed split (a config whose args name --front-exe, without --mmap-experts) gets vanilla's --shared-expert-arena: a memory object made here,
as serve/replicas.py makes one for replicas, so both processes map one copy of the experts (/dev/shm is too small for
it on a 64 GB PC).  This process then becomes the server (exec), keeping that memory object open, so the path stays
valid for as long as the server runs.  Stop it as any server: SIGINT or SIGTERM to this process (same pid).
"""
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from mixed_split_common import expand, shared_arena  # noqa: E402
from check_launch import check_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--port", type=int)
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--run-dir")
    ap.add_argument("--no-shared-arena", action="store_true")
    own = sys.argv[1:sys.argv.index("--")] if "--" in sys.argv else sys.argv[1:]
    extra = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    a = ap.parse_args(own)
    if any(x == "--config" or x.startswith("--config=") or x == "--engine" or x.startswith("--engine=") for x in extra):
        ap.error("server --config and --engine overrides bypass the launch check")
    names = {"STRATA_REPO": REPO}
    for kv in a.set:
        k, v = kv.split("=", 1)
        names[k] = v
    run_dir = os.path.abspath(a.run_dir or names.get("RUN_DIR") or os.environ.get("RUN_DIR") or os.path.join(REPO, "runs"))
    names["RUN_DIR"] = run_dir
    os.makedirs(run_dir, exist_ok=True)
    cfg = expand(json.load(open(a.config)), names)
    try:
        check_config(cfg)
    except (ValueError, OSError, subprocess.SubprocessError) as e:
        raise SystemExit(f"launch check: {e}") from e
    args = [str(x) for x in cfg.get("args", [])]
    # (--mmap-experts: no resident arena to share - vanilla refuses the two together; both processes then read the
    # experts through the same file mapping, one copy in the page cache)
    if ("--front-exe" in args and not a.no_shared_arena and "--shared-expert-arena" not in args
            and "--mmap-experts" not in args):
        path = shared_arena(inheritable=True)
        cfg["args"] = args + ["--shared-expert-arena", path]
        print(f"serve_mixed: one shared expert arena for both processes ({path})", flush=True)
    if a.port:
        cfg["port"] = a.port
    name = os.path.splitext(os.path.basename(a.config))[0].replace(".example", "")
    out = os.path.join(run_dir, name + ".json")
    with open(out, "w") as f:
        json.dump(cfg, f, indent=1)
    os.environ.setdefault("HSA_ENABLE_SDMA", "0")   # (the known-good vanilla run config's setting)
    py = os.path.join(REPO, ".venv", "bin", "python")
    argv = [py, os.path.join(REPO, "serve", "server.py"), "--engine", "strata", "--config", out]
    if a.port:
        argv += ["--port", str(a.port)]
    print("serve_mixed: " + " ".join(argv + extra), flush=True)
    os.chdir(REPO)
    os.execv(py, argv + extra)


if __name__ == "__main__":
    main()
