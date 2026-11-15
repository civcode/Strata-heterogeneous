#!/usr/bin/env python3
"""Derive private full-resident Q2_0 long-context configs; never launch GPUs.

Run as python -m tools.heterogeneous_long_context_config from the checkout.
Layer splits are candidates; native allocator admission remains mandatory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from serve.heterogeneous_long_context import LONG_CONTEXT_RESERVE, valid_cells
from tools.heterogeneous_hip_pair_config import IDENTITY, read_config, number, require

Q2_INDEX = "c344c01f59f547ed8a59003857a82eb90499e5299154241dbca787acb54b5b7c"


def derive(sources, output, split, context):
    require(valid_cells(context), "context must be 256-aligned, 2048..262144")
    require(len(split) in (2, 3) and all(type(n) is int and n >= 4 for n in split)
            and sum(split) == 48, "candidate requires two or three stage counts >=4 totaling 48")
    require(len(sources) in (2, 3) and len({p.resolve() for p in sources}) == len(sources),
            "two HIP or three mixed distinct source configs required")
    require(len(split) <= len(sources), "mixed plan requires three source configs")
    require(output.is_absolute() and output.parent.is_dir() and not output.exists(),
            "new absolute private output directory required")
    parsed = [read_config(p) for p in sources]
    configs = [f for f, _ in parsed]
    entry = configs[0]
    require(entry["index_sha256"] == Q2_INDEX and tuple(number(entry, k) for k in
            ("n_layers", "hc", "n_embd", "n_vocab")) == (48, 4, 2560, 248320),
            "expected approved Q2_0 model and geometry")
    require("ple_gguf_path" in entry and entry["ple_gguf_path"] in entry["gguf_shards"].split(";"),
            "entry must have its existing authorized PLE shard")
    lo = 0
    for i, f in enumerate(configs):
        hi = number(f, "end_layer")
        require(number(f, "stage_index") == i and number(f, "first_layer") == lo < hi <= 48,
                "source layers must be contiguous and ordered")
        require(all(f[k] == entry[k] for k in IDENTITY), "source model identity differs")
        require(number(f, "max_cache_slots") == (hi-lo)*512 and number(f, "max_t") == 8,
                "source must use full expert slots and max_t=8")
        require(number(f, "free_reserve_bytes") >= LONG_CONTEXT_RESERVE and number(f, "arena_budget_bytes") == 0,
                "source must preserve >=1 GiB reserve and uncapped owned arena")
        require(i == 0 or "ple_gguf_path" not in f, "only entry may own PLE")
        lo = hi
    ordinals = (0, 0, 1) if len(sources) == 3 else (0, 1)
    require(lo == 48 and tuple(number(f, "device_ordinal") for f in configs) == ordinals,
            "source device ordinal or coverage mismatch")
    require(configs[-2]["io_timeout_ms"] == configs[-1]["io_timeout_ms"], "HIP timeout mismatch")
    bases = configs[-2:] if len(split) == 2 else configs
    backends = ("hip", "hip") if len(split) == 2 else ("cuda", "hip", "hip")
    files, plan = {}, []
    lo = 0
    for i, (f0, count, backend) in enumerate(zip(bases, split, backends)):
        f = dict(f0)
        hi = lo + count
        f.update(stage_index=str(i), first_layer=str(lo), end_layer=str(hi),
                 max_cells=str(context), max_cache_slots=str(count*512))
        if i == 0:
            f["ple_gguf_path"] = entry["ple_gguf_path"]
        else:
            f.pop("ple_gguf_path", None)
        plan.append(f"{backend}:{f['device_ordinal']}:{lo}-{hi}")
        content = "# Q8 long-context candidate; native admission required.\n" + "".join(f"{k}={v}\n" for k, v in f.items())
        require(len(content.encode()) <= 16384 and all(len(s.encode()) <= 4096 for s in content.splitlines()),
                "generated config exceeds native bounds")
        files[f"stage{i}.cfg"] = content
        lo = hi
    evidence = dict(schema="strata-long-context-config/v1", plan=",".join(plan), context=context,
                    kv="int8", fingerprint=entry["model_fingerprint"], pack_index_sha256=Q2_INDEX,
                    sources=[dict(path=str(p), sha256=h) for p, (_, h) in zip(sources, parsed)],
                    generated={n: hashlib.sha256(c.encode()).hexdigest() for n,c in files.items()},
                    qualification="NOT_RUN")
    files["manifest.json"] = json.dumps(evidence, indent=2) + "\n"
    output.mkdir(mode=0o700)
    for name, content in files.items():
        with os.fdopen(os.open(output/name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8") as stream:
            stream.write(content)
    return evidence


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, nargs="+", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--split", type=int, nargs="+", required=True)
    p.add_argument("--context", type=int, required=True)
    a = p.parse_args()
    try:
        print(json.dumps(derive(a.source, a.output_dir, a.split, a.context), indent=2))
    except (OSError, ValueError, KeyError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
