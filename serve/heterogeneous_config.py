"""The supported vanilla run-config flags for the resident heterogeneous engine.

Keep topology/transport options separate; never silently ignore an engine flag.
Old diagnostic/qualification JSON is still accepted without rewriting it.
"""
from pathlib import Path

_COMMON = {"--max-context": "max_context", "--kv": "kv", "--prefill": "native_prefill_chunk"}
_MODEL = {"--pack", "--native", "--ple-gguf", "--expert-profile"}


def options(cfg):
    raw = cfg.get("heterogeneous")
    if not isinstance(raw, dict):
        raise ValueError("heterogeneous engine needs a heterogeneous config object")
    out = dict(raw)
    if "exe" in cfg:
        if "coordinator" in out:
            raise ValueError("use exe, not heterogeneous.coordinator, in a run config")
        out["coordinator"] = str(resolve(cfg, cfg["exe"]))
    if "args" not in cfg:
        return out
    flags = read_args(cfg["args"])
    for flag, key in _COMMON.items():
        if key in out:
            raise ValueError(f"use {flag} in args, not heterogeneous.{key}")
        if flag not in flags:
            raise ValueError(f"heterogeneous run config needs {flag} in args")
        value = flags[flag]
        if flag == "--kv":
            if value not in ("int8", "fp16"):
                raise ValueError("heterogeneous --kv supports int8 or fp16")
            out[key] = "f16" if value == "fp16" else value
        else:
            if not value.isascii() or not value.isdecimal():
                raise ValueError(f"{flag} requires an explicit integer in heterogeneous mode")
            out[key] = int(value)
    return out


def read_args(args):
    if not isinstance(args, list) or len(args) % 2 or not all(isinstance(x, str) and x for x in args):
        raise ValueError("args must contain separate quoted flag/value pairs")
    flags = {}
    for flag, value in zip(args[::2], args[1::2]):
        if flag not in _COMMON.keys() | _MODEL:
            raise ValueError(f"unsupported heterogeneous engine argument: {flag}")
        if flag in flags or value.startswith("--"):
            raise ValueError(f"duplicate or missing argument: {flag}")
        flags[flag] = value
    return flags


def resolve(cfg, value):
    if not isinstance(value, str) or not value:
        raise ValueError("expected a nonempty path in the run config")
    return (Path(cfg.get("cwd") or ".") / value).resolve()


def stage_model_args(stage_path):
    # Reuse the existing bounded v2 stage parser, not another config syntax.
    from tools.heterogeneous_hip_pair_config import read_config
    stage, _ = read_config(Path(stage_path))
    return ["--pack", stage["pack_dir"], "--native", stage["gguf_shards"].split(";")[0],
            "--ple-gguf", stage["ple_gguf_path"], "--expert-profile", stage["profile_path"]]


def verify_model_args(cfg, opts):
    if "args" not in cfg:
        return
    flags = read_args(cfg["args"])
    actual = read_args(stage_model_args(opts["config"][0]))
    for flag, value in actual.items():
        if flag not in flags or resolve(cfg, flags[flag]) != Path(value).resolve():
            raise ValueError(f"{flag} does not match the bound native model source")
    expected = Path(actual["--pack"]) / "tokenizer"
    if "tokenizer" not in cfg or resolve(cfg, cfg["tokenizer"]) != expected.resolve():
        raise ValueError("tokenizer does not match the bound native pack")

def native_run_config(cfg, prepared):
    """Translate a local qualification binding into an ordinary Strata run config.

    Only backend-specific worker arguments survive as extensions in args.
    Normal serving does not select ResidentEngine or a second Python launcher.
    """
    from copy import deepcopy
    opts, argv, source, _tok, stops, _template = prepared
    if not opts.stream_tokens or opts.lookup_draft or opts.prompt_chunk != 1 or opts.restart_clean:
        raise ValueError("native serve requires streamed, uncached, one-row greedy execution")
    out = deepcopy(cfg)
    out.pop("heterogeneous", None)
    out["exe"] = argv[0]
    out["cwd"] = str(Path(out.get("cwd") or ".").resolve())
    flags = read_args(out["args"])
    if int(flags["--max-context"]) != source["context_limit"]:
        raise ValueError("native context differs from requested context")
    # Resolve all model paths to the attested worker inputs.
    for flag in _MODEL:
        i = out["args"].index(flag) + 1
        out["args"][i] = str(resolve(cfg, out["args"][i]))
    out["tokenizer"] = str(resolve(cfg, out["tokenizer"]))
    if out.get("log"):
        out["log"] = str(resolve(cfg, out["log"]))
    out["args"] += ["--layer-split", ",".join(
        item.rsplit(":", 1)[1].split("-", 1)[0] for item in opts.plan.split(",")[1:])]
    # The coordinator validates the declared model identity and ordered stage
    # files again before initializing any GPU. No legacy JSON is used at run time.
    out["args"] += ["--eos-ids", ",".join(str(t) for t in sorted(stops))]
    skip_values = {"--kv", "--source-gguf", "--context-limit", "--stop-ids",
                   "--native-prefill-chunk"}
    switches = {"--shared", "--group-native-hip", "--long-context", "--workload-mode"}
    values = {"--plan", "--fingerprint", "--layers", "--hc", "--n-embd", "--timeout-ms",
              "--cuda-worker", "--hip-worker", "--config", "--source-index",
              "--generate", "--prompt-chunk"}
    i = 1
    while i < len(argv):
        flag = argv[i]
        if flag in ("--resident", "--stream-tokens"):
            i += 1
        elif flag in switches:
            out["args"].append(flag)
            i += 1
        elif flag in skip_values or flag in values:
            if i + 1 >= len(argv):
                raise ValueError(f"missing native value for {flag}")
            if flag in values:
                out["args"] += ["--hetero-max-new" if flag == "--generate" else flag, argv[i+1]]
            i += 2
        else:
            raise ValueError(f"unexpected native option in serve binding: {flag}")
    return out
