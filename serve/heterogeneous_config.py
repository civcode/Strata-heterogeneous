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


# Public launch profiles use the ordinary args list, not a second JSON
# schema. Only preparation of historical qualification files uses a binding.
def profile_binding(profile):
    from copy import deepcopy
    if "heterogeneous" in profile:
        return deepcopy(profile)  # historical, not emitted by current launchers
    cfg = deepcopy(profile)
    args = cfg.get("args")
    if not isinstance(args, list) or not all(isinstance(a, str) and a for a in args):
        raise ValueError("profile args must be a list of strings")
    values = {"--hetero-devices": "devices", "--layer-split": "split",
              "--fingerprint": "fingerprint",
              "--timeout-ms": "timeout_ms", "--prompt-chunk": "prompt_chunk"}
    switches = {"--shared": "shared", "--group-native-hip": "group_native_hip",
                "--workload-mode": "workload_mode", "--long-context": "long_context"}
    # Legacy source preflight needs a bounded diagnostic generation budget.
    # The application engine takes its output budget from GEN, not this value.
    binding = {"stream_tokens": True, "lookup_draft": False, "restart_clean": False,
               "max_new_tokens": 1}
    ordinary = []
    seen = set()
    i = 0
    while i < len(args):
        flag = args[i]
        if flag in seen:
            raise ValueError(f"duplicate profile argument: {flag}")
        seen.add(flag)
        if flag in switches:
            binding[switches[flag]] = True
            i += 1
            continue
        if flag not in values and flag not in _COMMON.keys() | _MODEL:
            raise ValueError(f"unsupported profile argument: {flag}")
        if i + 1 >= len(args) or args[i+1].startswith("--"):
            raise ValueError(f"missing value for {flag}")
        value = args[i+1]
        if flag in values:
            key = values[flag]
            if key not in ("devices", "split"):
                if not value.isascii() or not value.isdecimal():
                    raise ValueError(f"{flag} requires an integer")
                value = int(value)
            binding[key] = value
        else:
            ordinary += [flag, value]
        i += 2
    devices = binding.pop("devices", "").split(",")
    split = binding.pop("split", "").split(",")
    if (len(devices) < 2 or len(split) != len(devices)-1 or
            any(not n.isascii() or not n.isdecimal() for n in split)):
        raise ValueError("--layer-split requires one boundary per later device")
    bounds = [0, *map(int, split), binding.get("layers", 48)]
    if any(a >= b for a,b in zip(bounds,bounds[1:])):
        raise ValueError("--layer-split boundaries must increase within the model")
    binding["plan"] = ",".join(f"{device}:{a}-{b}" for device,a,b in zip(devices,bounds,bounds[1:]))
    cfg["args"] = ordinary
    cfg["heterogeneous"] = binding
    return cfg


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
    topology = opts.plan.split(",")
    out["args"] += ["--layer-split", ",".join(
        item.rsplit(":", 1)[1].split("-", 1)[0] for item in topology[1:]),
        "--hetero-devices", ",".join(item.rsplit(":", 1)[0] for item in topology)]
    # The coordinator validates the declared model identity and ordered stage
    # files again before initializing any GPU. No legacy JSON is used at run time.
    out["args"] += ["--eos-ids", ",".join(str(t) for t in sorted(stops))]
    skip_values = {"--plan", "--kv", "--source-gguf", "--context-limit", "--stop-ids",
                   "--native-prefill-chunk", "--generate"}
    switches = {"--shared", "--group-native-hip", "--long-context", "--workload-mode"}
    values = {"--fingerprint", "--layers", "--hc", "--n-embd", "--timeout-ms",
              "--cuda-worker", "--hip-worker", "--config", "--source-index",
              "--prompt-chunk"}
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
                out["args"] += [flag, argv[i+1]]
            i += 2
        else:
            raise ValueError(f"unexpected native option in serve binding: {flag}")
    return out
