"""Native run-config entry points for the current mixed-runtime capability set.

Use the native argument grouping and user override implementation. This is a
capability/admission adapter, not a replacement for generate.cpp's parser.
Legacy experimental profiles are supported only for existing qualification.
"""
from __future__ import annotations

from pathlib import Path
from strata_control import user_args
from strata_control.mixed_placement import layer_plan

MOVED = {"max_context", "kv", "mtp", "spec", "spec_min_p", "native_spec_window", "native_prefill_chunk"}
FLAGS = {"--max-context", "--kv", "--kv-resident", "--mtp", "--spec", "--spec-min-p",
         "--pack", "--native", "--ple-gguf", "--expert-profile", "--prefill",
         "--layer-split", "--hetero-devices"}


def _one_value_args(cfg):
    for key in ("args", "args_written", "user_args"):
        if key in cfg and (not isinstance(cfg[key], list) or
                           any(not isinstance(value, str) for value in cfg[key])):
            raise ValueError(f"{key} must be an array of strings")
    effective = user_args.merge(cfg["args"], cfg.get("user_args", []))
    values = {}
    for group in user_args.groups(effective):
        flag = user_args.flag_of(group)
        if flag not in FLAGS:
            raise ValueError(f"native option not implemented for mixed workers: {flag or group[0]}")
        if len(group) != 2 or not group[1]:
            raise ValueError(f"{flag} requires one value")
        # Native generate.cpp accepts successive occurrences (last wins).
        values[flag] = group[1]
    return values


def resolve(cfg):
    """Return private worker policy without mutating the persisted run config."""
    options = cfg.get("heterogeneous")
    if not isinstance(options, dict):
        raise ValueError("heterogeneous config must contain mixed placement metadata")
    options = dict(options)
    if "args" not in cfg:
        if any(key in cfg for key in ("user_args", "args_written", "exe", "tokenizer")):
            raise ValueError("native config fields require args")
        return options
    duplicated = MOVED & options.keys()
    if duplicated:
        raise ValueError("native args must not be duplicated in heterogeneous: " + ", ".join(sorted(duplicated)))
    args = _one_value_args(cfg)
    placement = {"--layer-split", "--hetero-devices"} & args.keys()
    if placement:
        if len(placement) != 2 or "plan" in options:
            raise ValueError("use --layer-split and --hetero-devices together, not heterogeneous.plan")
        options["plan"] = layer_plan(args["--hetero-devices"], args["--layer-split"],
                                     options.get("layers", 48))
    if not {"--max-context", "--kv"} <= args.keys():
        raise ValueError("mixed admission currently requires explicit --max-context and --kv")
    context = args["--max-context"]
    if not context.isascii() or not context.isdecimal() or int(context) <= 0:
        raise ValueError("--max-context requires a positive integer")
    options["max_context"] = int(context)
    kv = args["--kv"]
    if kv not in ("fp16", "int8"):
        raise ValueError("mixed native --kv currently supports fp16 or int8")
    options["kv"] = "f16" if kv == "fp16" else kv
    if args.get("--kv-resident", "0") != "0":
        raise ValueError("mixed workers currently require --kv-resident 0")
    if "--prefill" not in args:
        raise ValueError("mixed admission currently requires explicit --prefill chunk")
    prefill = args["--prefill"]
    if not prefill.isascii() or not prefill.isdecimal() or int(prefill) <= 0:
        raise ValueError("mixed workers currently require a positive explicit --prefill chunk; auto is not implemented")
    options["native_prefill_chunk"] = int(prefill)
    if "--mtp" in args:
        if args.get("--spec") not in ("4", "5"):
            raise ValueError("mixed native MTP currently requires explicit --spec 4 or 5")
        try:
            minimum = float(args.get("--spec-min-p", "nan"))
        except ValueError:
            minimum = float("nan")
        if minimum != 0.5:
            raise ValueError("mixed native MTP currently requires explicit --spec-min-p 0.5")
        options.update(mtp=args["--mtp"], spec=4, spec_min_p=minimum,
                       native_spec_window=int(args["--spec"]))
    elif "--spec" in args or "--spec-min-p" in args:
        raise ValueError("mixed native --spec and --spec-min-p require --mtp")
    if "exe" in cfg:
        if "coordinator" in options:
            raise ValueError("native exe must not be duplicated as heterogeneous.coordinator")
        if not isinstance(cfg["exe"], str) or not cfg["exe"]:
            raise ValueError("exe must be a nonempty path")
        options["coordinator"] = cfg["exe"]
    return options


def verify_source(cfg, source):
    """Verify ordinary path options against the coordinator's native attestation."""
    if "args" not in cfg:
        return
    args = _one_value_args(cfg)
    # Worker source attests absolute paths. Resolve run-config relative paths
    # in the engine's cwd, exactly as the native subprocess would.
    cwd = Path(cfg.get("cwd") or Path.cwd())
    def path(value):
        p = Path(value).expanduser()
        return (cwd / p).resolve(strict=True)
    for flag, key in (("--pack", "pack_dir"), ("--native", "gguf"),
                      ("--ple-gguf", "ple_gguf"), ("--expert-profile", "profile_path")):
        if flag in args:
            actual = source.get(key)
            if not actual or path(args[flag]) != path(actual):
                raise ValueError(f"{flag} differs from the admitted native worker source")
    if "tokenizer" in cfg:
        if (not isinstance(cfg["tokenizer"], str) or
            Path(cfg["tokenizer"]).expanduser().resolve(strict=True) !=
                (Path(source["pack_dir"]) / "tokenizer").resolve(strict=True)):
            raise ValueError("tokenizer differs from the admitted native pack tokenizer")


def bind_standard_fields(cfg, source):
    """Fill model identity from validated native worker sources; stamp generated args."""
    if "args" not in cfg:
        return
    verify_source(cfg, source)
    args = cfg["args"]
    supplied = _one_value_args(cfg)
    for flag, key in (("--pack", "pack_dir"), ("--native", "gguf"),
                      ("--ple-gguf", "ple_gguf"), ("--expert-profile", "profile_path")):
        if flag not in supplied and source.get(key):
            args.extend([flag, source[key]])
    cfg.setdefault("tokenizer", str(Path(source["pack_dir"]) / "tokenizer"))
    user_args.stamp(cfg)
