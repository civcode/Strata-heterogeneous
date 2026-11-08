#!/usr/bin/env python3
"""Phase 11B: bounded, opt-in two-GPU *real-model* layer-split/restore gate.

This drives the existing serve.server.StrataEngine API; it does not alter models,
configs, drivers or the engine. By default it is a dry run. With --run it starts
one production engine per GPU UUID order, sends A -> B -> A, and requires evidence
that both stages ran and A was restored from a parked conversation image.

This is an execution/ownership gate, NOT token-parity, long-context, P2P, or
throughput qualification. The caller must independently inspect kernel journals.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import threading
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class GateFailure(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateFailure(message)


def parse_ids_file(path: Path) -> list[int]:
    words = path.read_text(encoding="utf-8").replace("\n", ",").split(",")
    try:
        ids = [int(word.strip()) for word in words if word.strip()]
    except ValueError as exc:
        raise GateFailure(f"invalid token-id fixture {path}: {exc}") from exc
    require(len(ids) >= 128, f"{path}: need at least 128 real prompt tokens")
    require(all(0 <= x < 2**31 for x in ids), f"{path}: invalid token ID")
    return ids


def make_tokenized_prompts(config: dict, target: int) -> tuple[list[int], list[int]]:
    """Use the same tokenizer/template loading that Strata's existing cache tests use."""
    sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
    from conversation_cache_parity import load_tokenizer  # imported only when executing
    from serve.frontend import ChatTemplate

    tokenizer_dir = Path(config["tokenizer"])
    tokenizer = load_tokenizer(tokenizer_dir)
    template = ChatTemplate(tokenizer_dir / "chat_template.jinja")

    def make(tag: str, theme: str) -> list[int]:
        lines: list[str] = []
        for i in range(4096):
            lines.append(f"{tag} record {i}: {theme}; index {i % 17}.")
            if i % 8 != 7:
                continue
            text = f"Conversation {tag}. Retain these records:\n" + "\n".join(lines)
            ids = tokenizer.encode(
                template.render([{"role": "user", "content": text}], enable_thinking=False),
                parse_special=True,
            )
            if len(ids) >= target:
                return [int(t) for t in ids]
        raise GateFailure(f"could not build {target}-token prompt fixture")

    return make("A", "amber and cedar"), make("B", "quartz and willow")


def verify_run(log: str, records: list[dict], order: tuple[str, str]) -> None:
    """Accept only observed two-stage execution plus an actual parked-image restore."""
    require(len(records) == 3, "incomplete A/B/A request sequence")
    require([r["name"] for r in records] == ["A", "B", "A-return"], "request order changed")
    stage_lines = [line for line in log.splitlines() if "layer split: layers " in line]
    require(any("CUDA0" in line and "CUDA1" in line for line in stage_lines),
            "no production two-stage CUDA0/CUDA1 layer-split startup line")
    for r in records:
        require(r.get("tokens") and all(isinstance(t, int) for t in r["tokens"]),
                f"{r['name']}: no generated token IDs")
        require(r.get("finish") in ("stop", "length"), f"{r['name']}: non-normal DONE status")
    require(records[2].get("reused", 0) > 0,
            "A-return did not report reused tokens (no validated restore)")
    require(re.search(r"\brestored\s+\d+\s+tokens\b", records[2].get("log_excerpt", ""), re.I),
            "A-return lacks a fresh parked-snapshot restore log line")
    # The test intentionally does NOT require A and A-return's tokens to match;
    # previous Strata reports demonstrated execution-path numerical differences.
    require(order[0] != order[1], "GPU UUIDs are not distinct")


def run_one(config: dict, executable: Path, output: Path, uuids: tuple[str, str],
            ids_a: list[int], ids_b: list[int], max_new: int, cache_mib: int) -> dict:
    # Import late so --help, --dry-run and unit tests need no serving dependencies.
    sys.path.insert(0, str(ROOT))
    from serve.server import StrataEngine, child_env, engine_args

    label = "-".join((uuids[0][-8:], uuids[1][-8:]))
    log_path = output / f"layer-split-{label}.log"
    require(not log_path.exists(), f"refusing to overwrite existing log: {log_path}")
    c = copy.deepcopy(config)
    c["gpu"] = [0, 1]
    c["gpu_order"] = "as_given"
    c["layer_split"] = "auto"
    c["log"] = str(log_path)
    c["args"] = list(config["args"]) + [
        "--conversation-cache-mib", str(cache_mib),
        "--conversation-cache-slots", "4",
        "--prompt-cache", "4",
        "--prompt-cache-every", "128",
    ]
    env = child_env(c)
    env["ROCR_VISIBLE_DEVICES"] = ",".join(uuids)
    env["HSA_ENABLE_SDMA"] = "0"
    env["HIP_PLATFORM"] = "amd"
    env["STRATA_SNAPSHOT_VERIFY"] = "1"
    args = engine_args(c)
    require("--layer-split" in args,
            "serve.server.engine_args did not enable layer splitting")
    records: list[dict] = []
    engine = None
    try:
        engine = StrataEngine(str(executable), args, cwd=c.get("cwd"),
                              log=str(log_path), env=env)
        for name, ids in (("A", ids_a), ("B", ids_b), ("A-return", ids_a)):
            beginning = log_path.stat().st_size if log_path.exists() else 0
            tokens = [int(t) for t in engine.generate(
                ids, max_new, {"temperature": 0}, threading.Event()) if t is not None]
            last = dict(engine.last or {})
            with log_path.open("rb") as f:
                f.seek(beginning)
                excerpt = f.read().decode("utf-8", "replace")
            records.append({"name": name, "prompt_tokens": len(ids),
                            "tokens": tokens, "finish": last.get("finish"),
                            "reused": int(last.get("reused") or 0),
                            "log_excerpt": excerpt})
    finally:
        if engine is not None:
            engine.close()
    log = log_path.read_text(encoding="utf-8", errors="replace")
    verify_run(log, records, uuids)
    return {"uuid_order": list(uuids), "status": "pass", "records": records,
            "log": str(log_path), "engine_args": args}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="existing read-only Strata JSON run profile")
    parser.add_argument("--engine", type=Path, required=True, help="HIP strata executable built from Phase 11B worktree")
    parser.add_argument("--output", type=Path, required=True, help="NEW evidence directory, never overwritten")
    parser.add_argument("--uuid-a", required=True, help="first R9700 GPU UUID")
    parser.add_argument("--uuid-b", required=True, help="second R9700 GPU UUID")
    parser.add_argument("--ids-a", type=Path, help="optional existing comma-separated A prompt token fixture")
    parser.add_argument("--ids-b", type=Path, help="optional existing comma-separated B prompt token fixture")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--max-new", type=int, default=1)
    parser.add_argument("--cache-mib", type=int, default=2048)
    parser.add_argument("--run", action="store_true", help="actually start two real-model GPU engines")
    a = parser.parse_args(argv)
    require(a.uuid_a != a.uuid_b and all(s.startswith("GPU-") for s in (a.uuid_a, a.uuid_b)),
            "two distinct ROCm GPU UUIDs are required")
    require(128 <= a.prompt_tokens <= 2048, "prompt-tokens must be 128..2048")
    require(1 <= a.max_new <= 8 and 128 <= a.cache_mib <= 8192,
            "bounded max-new/cache-mib limits exceeded")
    require(bool(a.ids_a) == bool(a.ids_b), "supply both token fixtures or neither")
    require(not a.output.exists(), "output already exists; never overwrite existing evidence")
    if not a.run:
        print("DRY RUN: two sequential bounded HIP engines, UUID orders A,B and B,A.")
        print("Each executes GEN A -> GEN B -> GEN A and verifies a real parked-state restore.")
        print("No files, models, drivers or GPUs touched. Add --run during an exclusive GPU window.")
        return 0
    require(a.config.is_file() and a.engine.is_file(), "profile or executable not found")
    config = json.loads(a.config.read_text(encoding="utf-8"))
    require(config.get("backend") in (None, "hip"), "this gate is for HIP only")
    require(isinstance(config.get("args"), list), "profile args are missing")
    if a.ids_a:
        ids_a, ids_b = parse_ids_file(a.ids_a), parse_ids_file(a.ids_b)
    else:
        ids_a, ids_b = make_tokenized_prompts(config, a.prompt_tokens)
    require(ids_a != ids_b and max(map(len, (ids_a, ids_b))) <= 4096,
            "distinct, bounded A and B prompts required")
    a.output.mkdir(mode=0o700, parents=False, exist_ok=False)
    info = {"phase": "11B", "scenario": "layer-split A/B/A", "results": [],
            "qualification": "bounded HIP real-model ownership only; not P2P/long-context/CUDA"}
    try:
        for pair in ((a.uuid_a, a.uuid_b), (a.uuid_b, a.uuid_a)):
            info["results"].append(run_one(config, a.engine, a.output, pair,
                                           ids_a, ids_b, a.max_new, a.cache_mib))
            (a.output / "results.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
        info["traceback"] = traceback.format_exc()
        (a.output / "results.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        raise
    print(f"PASS: both UUID visibility-list orders executed; verify physical ordinal mapping in logs; evidence: {a.output}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (GateFailure, OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
