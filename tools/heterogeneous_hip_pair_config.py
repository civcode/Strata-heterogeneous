#!/usr/bin/env python3
"""Derive two full-resident HIP configs from the approved mixed v2 configs.

CPU only. Does not open model assets, launch workers, or qualify VRAM fit.
Native coordinator --describe-source/preflight remains mandatory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re

FIELDS = set("version pack_dir profile_path index_sha256 model_fingerprint stage_index device_ordinal first_layer end_layer n_layers hc n_embd n_vocab max_cells max_t cpu_workers max_cache_slots io_timeout_ms free_reserve_bytes arena_budget_bytes gguf_shards".split())
IDENTITY = "pack_dir profile_path index_sha256 model_fingerprint n_layers hc n_embd n_vocab gguf_shards".split()
LIMITS = ("max_t", "io_timeout_ms")


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def read_config(path):
    require(path.is_absolute() and path.is_file(), "absolute source config required")
    with path.open("rb") as f:
        raw = f.read(16385)
    require(0 < len(raw) <= 16384 and b"\r" not in raw and b"\x00" not in raw, "invalid config size or control byte")
    fields = {}
    for line in raw.decode("utf-8").split("\n"):
        if not line or line.startswith("#"):
            continue
        require(len(line.encode()) <= 4096 and line.count("=") == 1, "invalid config line")
        k, v = line.split("=")
        require(k and v and k not in fields, "duplicate or empty config field")
        fields[k] = v
    require(fields.keys() == FIELDS or fields.keys() == FIELDS | {"ple_gguf_path"}, "unexpected v2 schema")
    require(fields["version"] == "2", "v2 source required")
    return fields, hashlib.sha256(raw).hexdigest()


def number(fields, key):
    value = fields[key]
    require(re.fullmatch(r"[0-9]+", value) is not None, "invalid numeric field: " + key)
    return int(value)


def derive(sources, output, split=24, context=2048):
    require(len(sources) == 3 and len({p.resolve() for p in sources}) == 3, "three distinct mixed source configs required")
    require(output.is_absolute() and output.parent.is_dir() and not output.exists(), "new absolute output directory required")
    require(type(split) is int and 2 <= split <= 47 and type(context) is int and 2 <= context <= 32768, "split/context outside experiment bounds")
    parsed = [read_config(p) for p in sources]
    configs = [f for f, _ in parsed]
    entry, hip0, hip1 = configs
    lo = 0
    for i, f in enumerate(configs):
        hi = number(f, "end_layer")
        require(number(f, "stage_index") == i and number(f, "first_layer") == lo and lo < hi <= 48, "source stages must be ordered and contiguous")
        require(number(f, "n_layers") == 48 and number(f, "max_cache_slots") == (hi-lo)*512, "source must be the fully resident 48-layer model")
        require(all(f[k] == entry[k] for k in IDENTITY), "source model identity mismatch")
        require(number(f, "free_reserve_bytes") >= 536870912 and number(f, "arena_budget_bytes") == 0, "require existing >=512 MiB reserve and uncapped arena")
        require(number(f, "max_cells") >= context and number(f, "max_t") == 8, "source capacity/window cannot serve requested context")
        lo = hi
    require(lo == 48 and (number(hip0, "device_ordinal"), number(hip1, "device_ordinal")) == (0, 1), "expected HIP ordinals 0,1 and complete model")
    require(all(hip0[k] == hip1[k] for k in LIMITS), "HIP window/timeout mismatch")
    require(number(entry, "end_layer") > 1 and "ple_gguf_path" in entry and "ple_gguf_path" not in hip0 and "ple_gguf_path" not in hip1 and entry["ple_gguf_path"] in entry["gguf_shards"].split(";"), "entry PLE shard must be an authorized original shard")
    plan = f"hip:0:0-{split},hip:1:{split}-48"
    files = {}
    for i, (base, lb, le) in enumerate(((hip0, 0, split), (hip1, split, 48))):
        f = dict(base)
        f.update(stage_index=str(i), first_layer=str(lb), end_layer=str(le), max_cache_slots=str((le-lb)*512), max_cells=str(context))
        if i == 0:
            f["ple_gguf_path"] = entry["ple_gguf_path"]
        content = "# M0035 HIP-only candidate; native preflight and memory qualification required.\n" + "".join(f"{k}={v}\n" for k, v in f.items())
        require(len(content.encode()) <= 16384 and all(len(x.encode()) <= 4096 for x in content.splitlines()), "generated config exceeds native bounds")
        files[f"hip{i}.cfg"] = content
    evidence = dict(schema="strata-hip-pair-config/v1", plan=plan, context=context,
                    fingerprint=entry["model_fingerprint"], index_sha256=entry["index_sha256"],
                    sources=[{"path":str(p), "sha256":h} for p, (_, h) in zip(sources, parsed)],
                    generated={n:hashlib.sha256(c.encode()).hexdigest() for n,c in files.items()}, qualification="NOT_RUN")
    files["manifest.json"] = json.dumps(evidence, indent=2) + "\n"
    # Exclusive new private directory; never alter inputs or reuse a previous run.
    output.mkdir(mode=0o700)
    # On an I/O failure retain the partial directory for diagnosis; do not reuse it.
    for name, content in files.items():
        with os.fdopen(os.open(output/name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8") as f:
            f.write(content)
    return evidence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, nargs=3, required=True, metavar="CFG")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--split", type=int, default=24)
    p.add_argument("--context", type=int, default=2048)
    a = p.parse_args()
    try:
        print(json.dumps(derive(a.source, a.output_dir, a.split, a.context), indent=2))
    except (OSError, ValueError, KeyError) as e:
        p.error(str(e))


if __name__ == "__main__":
    main()
