"""Explicit, source-checkout heterogeneous text completion using native workers.

No GPU runtime, sampling implementation, fallback tokenizer or server here.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def add_arguments(parser):
    parser.add_argument("--coordinator", required=True, help="built strata_heterogeneous_live_smoke executable")
    parser.add_argument("--cuda-worker", required=True)
    parser.add_argument("--hip-worker", required=True)
    parser.add_argument("--config", action="append", required=True, help="one worker config per stage, in plan order")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--fingerprint", type=int, required=True)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--hc", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=2560)
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt", help="text completion prompt or one chat user message")
    prompt.add_argument("--prompt-file", type=Path, help="UTF-8 text file, at most 64 KiB")
    parser.add_argument("--chat", action="store_true", help="render one user message with the exact model chat template")
    parser.add_argument("--max-new-tokens", type=int, default=16, help="1..256, existing greedy head")
    parser.add_argument("--prompt-chunk", type=int, default=1, help="1..8, limited by every worker's max_t")
    parser.add_argument("--timeout-ms", type=int, default=120000, help="per worker I/O operation")
    parser.add_argument("--shared", action="store_true", help="use existing memfd transport")
    parser.add_argument("--run", action="store_true", help="explicitly start native GPU workers; default is preflight only")
    parser.add_argument("--output", choices=("human", "json"), default="human")


def _path(value, executable=False):
    path = Path(value).expanduser().resolve(strict=True)
    if not path.is_file() or (executable and not os.access(path, os.X_OK)):
        raise ValueError(f"not a usable {'executable' if executable else 'file'}: {path}")
    return str(path)


def _native(argv, *, text=None, marker=None):
    # Keep the native worker diagnostics off the CLI's machine-output stream.
    # No shell; subprocess stdin is closed before the coordinator launches GPUs.
    result = subprocess.run(argv, input=text if text is not None else "",
                            stdout=subprocess.PIPE, encoding="utf-8", errors="replace", check=False)
    records = []
    for line in result.stdout.splitlines():
        if marker and line.startswith(marker + " "):
            records.append(line[len(marker) + 1:])
        else:
            print(line, file=sys.stderr)
    if result.returncode:
        raise ValueError(f"heterogeneous coordinator failed (exit {result.returncode}); see native diagnostics")
    if marker:
        if len(records) != 1:
            raise ValueError(f"expected exactly one {marker} record")
        record = json.loads(records[0])
        if not isinstance(record, dict):
            raise ValueError(f"invalid {marker} record")
        return record
    return None


def run(args):
    if not 1 <= args.max_new_tokens <= 256 or not 1 <= args.prompt_chunk <= 8:
        raise ValueError("max-new-tokens must be 1..256 and prompt-chunk 1..8")
    if args.prompt_file is not None:
        with args.prompt_file.open("rb") as stream:
            data = stream.read(65537)
        if len(data) > 65536:
            raise ValueError("prompt exceeds 64 KiB")
        prompt = data.decode("utf-8")
    else:
        prompt = args.prompt
        if len(prompt.encode("utf-8")) > 65536:
            raise ValueError("prompt exceeds 64 KiB")
    argv = [_path(args.coordinator, True), "--plan", args.plan, "--fingerprint", str(args.fingerprint),
            "--layers", str(args.layers), "--hc", str(args.hc), "--n-embd", str(args.n_embd),
            "--cuda-worker", _path(args.cuda_worker, True), "--hip-worker", _path(args.hip_worker, True),
            "--timeout-ms", str(args.timeout_ms)]
    for config in args.config:
        argv.extend(("--config", _path(config)))
    if args.shared:
        argv.append("--shared")
    # The native config parser determines the authorized tokenizer source;
    # never accept an independently supplied tokenizer or ByteTokenizer.
    source = _native(argv + ["--describe-source"], marker="HETERO_SOURCE")
    gguf = source.get("gguf")
    index = source.get("pack_index_sha256")
    vocab = source.get("vocab")
    cells = source.get("max_cells")
    max_t = source.get("max_t")
    if (not isinstance(gguf, str) or not Path(gguf).is_absolute() or
            not isinstance(index, str) or len(index) != 64 or
            any(c not in "0123456789abcdef" for c in index) or
            any(type(n) is not int or n <= 0 for n in (vocab, cells, max_t))):
        raise ValueError("invalid validated model source description")
    try:
        from tools.strata_tokenizer import Tokenizer
    except ImportError as exc:
        raise ValueError("run from the Strata source checkout with its existing tokenizer dependencies installed") from exc
    tok = Tokenizer.from_gguf(gguf)
    if len(tok.tokens) != vocab or tok.pre != "qwen35":
        raise ValueError("tokenizer vocabulary or family does not match the supported Qwen worker model")
    if args.chat:
        # Use the serving frontend and verify the installed template against
        # the same original primary shard, never a different model fallback.
        from tools.gguf_reader import GGUFFile
        try:
            from serve.frontend import ChatTemplate, literal_tags, mark_think_literals, unmark_think_literals
        except ImportError as exc:
            raise ValueError("chat mode requires the existing Strata frontend dependencies") from exc
        pack = source.get("pack_dir")
        if not isinstance(pack, str) or not Path(pack).is_absolute():
            raise ValueError("missing validated pack directory for chat template")
        template = ChatTemplate(Path(pack) / "tokenizer" / "chat_template.jinja")
        original = GGUFFile(Path(gguf)).metadata.get("tokenizer.chat_template")
        if not isinstance(original, str) or template.source != original:
            raise ValueError("pack chat template does not match the original GGUF")
        tags = literal_tags(tok.control_tokens)
        messages, tools, changed = mark_think_literals([{"role": "user", "content": prompt}], None, tags)
        rendered = template.render(messages, tools=tools, enable_thinking=False)
        rendered, plain = unmark_think_literals(rendered, tags) if changed else (rendered, ())
        ids = tok.encode(rendered, parse_special=True, plain=plain)
    else:
        ids = tok.encode(prompt, parse_special=False)
    if not ids or len(ids) > 4096:
        raise ValueError("prompt must encode to 1..4096 tokens")
    if len(ids) + args.max_new_tokens > cells or min(len(ids), args.prompt_chunk) > max_t:
        raise ValueError("prompt/generation budget exceeds validated worker context or max_t")
    stop_ids = {tok.special_tokens[marker] for marker in ("<|im_end|>", "<|endoftext|>")
                if marker in tok.special_tokens}
    for key in ("tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id"):
        if key in tok.special_ids:
            stop_ids.add(tok.special_ids[key])
    if not stop_ids or any(type(t) is not int or not 0 <= t < vocab for t in stop_ids):
        raise ValueError("no valid model end-of-generation token IDs")
    native = argv + ["--generate", str(args.max_new_tokens), "--prompt-chunk", str(args.prompt_chunk),
                     "--source-gguf", gguf, "--source-index", index,
                     "--stop-ids", ",".join(str(t) for t in sorted(stop_ids))]
    text = " ".join(str(t) for t in ids) + "\n"
    result = {"prompt_tokens": len(ids), "prompt_token_ids": ids, "max_new_tokens": args.max_new_tokens,
              "prompt_chunk": args.prompt_chunk, "chat": args.chat, "source": source, "stop_ids": sorted(stop_ids)}
    if not args.run:
        _native(native, text=text)
        result["status"] = "preflight"
        if args.output == "human":
            print(f"PASS: {len(ids)} prompt tokens; no GPU workers started. Add --run for native generation.")
    else:
        generated = _native(native + ["--run"], text=text, marker="HETERO_GENERATION")
        tokens = generated.get("token_ids")
        reason = generated.get("finish_reason")
        if (generated.get("prompt_tokens") != len(ids) or not isinstance(tokens, list) or
                len(tokens) > args.max_new_tokens or
                any(type(t) is not int or not 0 <= t < vocab or t in stop_ids for t in tokens) or
                reason not in ("stop", "length") or
                (reason == "length" and len(tokens) != args.max_new_tokens) or
                (reason == "stop" and len(tokens) >= args.max_new_tokens)):
            raise ValueError("native generation returned inconsistent completion metadata")
        result.update(status="complete", token_ids=tokens, completion_tokens=len(tokens),
                      finish_reason=reason, text=tok.decode(tokens))
        if args.output == "human":
            print(result["text"])
    if args.output == "json":
        print(json.dumps(result, ensure_ascii=True))
    return 0
