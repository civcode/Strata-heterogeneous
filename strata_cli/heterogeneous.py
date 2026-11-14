"""Explicit, source-checkout heterogeneous text completion using native workers.

No GPU runtime, sampling implementation, fallback tokenizer or server here.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


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
    prompt.add_argument("--write-server-config", type=Path, help="validate and write a new loopback server JSON; no GPU startup")
    parser.add_argument("--model-name", default="qwen3.8-flash-next", help="served model name for generated config")
    parser.add_argument("--chat", action="store_true", help="render one user message with the exact model chat template")
    parser.add_argument("--max-new-tokens", type=int, default=16, help="1..256, existing greedy head")
    parser.add_argument("--prompt-chunk", type=int, default=1, help="1..8; 0 selects largest safe verifier window")
    parser.add_argument("--max-context", type=int, default=4096, help="total prompt + completion cap, 2..8192; worker capacities still apply")
    parser.add_argument("--stream-tokens", action="store_true", help="opt-in committed-token resident protocol (CLI still writes final JSON)")
    parser.add_argument("--lookup-draft", action="store_true", help="opt-in verified suffix lookup; buffered mode, chunk >= 2")
    parser.add_argument("--restart-clean", action="store_true", help="opt-in server restart only after verified old process-group exit")
    parser.add_argument("--benchmark-repeat", type=int, default=1, help="repeat resident prompt sequence 1..20 times; JSON includes timing")
    parser.add_argument("--timeout-ms", type=int, default=120000, help="per worker I/O operation")
    parser.add_argument("--shared", action="store_true", help="use existing memfd transport")
    parser.add_argument("--run", action="store_true", help="explicitly start native GPU workers; default is preflight only")
    parser.add_argument("--resident", action="store_true", help="reuse one native worker set across independent prompts")
    parser.add_argument("--additional-prompt", action="append", default=[], help="additional independent prompt in resident mode")
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


def prepare_native(args):
    """CPU-only source/identity and tokenizer loading for the CLI and server."""
    from serve.heterogeneous_contract import generation_limits
    if not 1 <= args.max_new_tokens <= 256 or not 0 <= args.prompt_chunk <= 8:
        raise ValueError("max-new-tokens must be 1..256 and prompt-chunk 0..8")
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
    context, chunk = generation_limits(cells, max_t, vocab, args.hc, args.n_embd,
                                       getattr(args, "max_context", 4096),
                                       args.max_new_tokens, args.prompt_chunk)
    args.prompt_chunk = chunk
    lookup = getattr(args, "lookup_draft", False)
    if lookup and (getattr(args, "stream_tokens", False) or chunk < 2):
        raise ValueError("lookup-draft requires buffered generation and chunk >= 2")
    source.update(context_limit=context, prompt_chunk=chunk,
                  stream_tokens=getattr(args, "stream_tokens", False),
                  restart_clean=getattr(args, "restart_clean", False), lookup_draft=lookup)
    try:
        from tools.strata_tokenizer import Tokenizer
    except ImportError as exc:
        raise ValueError("run from the Strata source checkout with its existing tokenizer dependencies installed") from exc
    tok = Tokenizer.from_gguf(gguf)
    if len(tok.tokens) != vocab or tok.pre != "qwen35":
        raise ValueError("tokenizer vocabulary or family does not match the supported Qwen worker model")
    stop_ids = {tok.special_tokens[marker] for marker in ("<|im_end|>", "<|endoftext|>")
                if marker in tok.special_tokens}
    for key in ("tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id"):
        if key in tok.special_ids:
            stop_ids.add(tok.special_ids[key])
    if not stop_ids or any(type(t) is not int or not 0 <= t < vocab for t in stop_ids):
        raise ValueError("no valid model end-of-generation token IDs")
    native = argv + ["--generate", str(args.max_new_tokens), "--prompt-chunk", str(args.prompt_chunk),
                     "--source-gguf", gguf, "--source-index", index,
                     "--stop-ids", ",".join(str(t) for t in sorted(stop_ids)),
                     "--context-limit", str(context)]
    if source["stream_tokens"]:
        native.append("--stream-tokens")
    if lookup:
        native.append("--lookup-draft")
    return native, source, tok, stop_ids


def model_template(source):
    from tools.gguf_reader import GGUFFile
    from serve.frontend import ChatTemplate
    pack = source.get("pack_dir")
    if not isinstance(pack, str) or not Path(pack).is_absolute():
        raise ValueError("missing validated pack directory for chat template")
    template = ChatTemplate(Path(pack) / "tokenizer" / "chat_template.jinja")
    original = GGUFFile(Path(source["gguf"])).metadata.get("tokenizer.chat_template")
    if not isinstance(original, str) or template.source != original:
        raise ValueError("pack chat template does not match the original GGUF")
    return template


def encode_prompt(tok, source, prompt, chat, max_new, chunk):
    if len(prompt.encode("utf-8")) > 65536:
        raise ValueError("prompt exceeds 64 KiB")
    if chat:
        from serve.frontend import literal_tags, mark_think_literals, unmark_think_literals
        template = model_template(source)
        tags = literal_tags(tok.control_tokens)
        messages, tools, changed = mark_think_literals([{"role": "user", "content": prompt}], None, tags)
        rendered = template.render(messages, tools=tools, enable_thinking=False)
        rendered, plain = unmark_think_literals(rendered, tags) if changed else (rendered, ())
        ids = tok.encode(rendered, parse_special=True, plain=plain)
    else:
        ids = tok.encode(prompt, parse_special=False)
    from serve.heterogeneous_contract import validate_request
    validate_request(ids, max_new, source["vocab"],
                     source.get("context_limit", min(source["max_cells"], 4096)))
    if min(len(ids), chunk) > source["max_t"]:
        raise ValueError("prompt chunk exceeds validated worker max_t")
    return ids


def write_server_config(args):
    from serve.heterogeneous import prepare_server
    cfg = {"model_name": args.model_name, "host": "127.0.0.1",
           "max_queue": 0, "reasoning_effort": "none",
           "heterogeneous": {key: getattr(args, key) for key in
               ("plan", "fingerprint", "layers", "hc", "n_embd", "max_new_tokens",
                "prompt_chunk", "max_context", "timeout_ms", "shared", "stream_tokens",
                "restart_clean", "lookup_draft")}}
    for key in ("coordinator", "cuda_worker", "hip_worker"):
        cfg["heterogeneous"][key] = _path(getattr(args, key), True)
    cfg["heterogeneous"]["config"] = [_path(p) for p in args.config]
    if not isinstance(args.model_name, str) or not args.model_name.strip():
        raise ValueError("model-name must be nonempty")
    prepare_server(cfg)
    with args.write_server_config.open("x", encoding="utf-8") as out:
        json.dump(cfg, out, indent=2)
        out.write("\n")
    print(json.dumps({"status": "preflight", "server_config": str(args.write_server_config.resolve())}))
    return 0


def run(args):
    if not 1 <= args.max_new_tokens <= 256 or not 0 <= args.prompt_chunk <= 8:
        raise ValueError("max-new-tokens must be 1..256 and prompt-chunk 0..8")
    if not 1 <= args.benchmark_repeat <= 20:
        raise ValueError("benchmark-repeat must be 1..20")
    if args.write_server_config is not None:
        if args.run or args.resident or args.chat or args.additional_prompt or args.benchmark_repeat != 1:
            raise ValueError("config generation is CPU-only and cannot be combined with generation")
        return write_server_config(args)
    if args.restart_clean:
        raise ValueError("--restart-clean applies only to generated server config")
    if (args.stream_tokens or args.benchmark_repeat != 1) and not args.resident:
        raise ValueError("stream-tokens and benchmark-repeat require --resident")
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
    native, source, tok, stop_ids = prepare_native(args)
    vocab = source["vocab"]
    ids = encode_prompt(tok, source, prompt, args.chat, args.max_new_tokens, args.prompt_chunk)
    if args.additional_prompt and not args.resident:
        raise ValueError("--additional-prompt requires --resident")
    if args.resident:
        from serve.heterogeneous import ResidentEngine
        from threading import Event
        prompts = [ids] + [encode_prompt(tok, source, txt, args.chat, args.max_new_tokens, args.prompt_chunk)
                           for txt in args.additional_prompt]
        if len(prompts) * args.benchmark_repeat > 100:
            raise ValueError("resident benchmark limited to 100 total requests")
        prompts *= args.benchmark_repeat
        resident = native + ["--resident"]
        _native(resident)
        if not args.run:
            result = {"status": "preflight", "source": source, "prompt_token_ids": prompts}
        else:
            start = time.perf_counter()
            engine = ResidentEngine(resident, source, stop_ids, args.max_new_tokens,
                                    args.prompt_chunk, args.timeout_ms, ValueError)
            startup_ms = (time.perf_counter() - start) * 1000
            try:
                completions = []
                for ids in prompts:
                    for _ in engine.generate(ids, args.max_new_tokens, {"temperature": 0}, Event()):
                        pass
                    record = dict(engine.last_completion)
                    record.update(text=tok.decode(record["token_ids"]), prompt_token_ids=ids,
                                  timing=engine.last_metrics)
                    completions.append(record)
                result = {"status": "complete", "coordinator_pid": engine.proc.pid,
                          "source": source, "completions": completions,
                          "startup_ms": startup_ms}
            finally:
                start_close = time.perf_counter()
                engine.close()
            result["shutdown_ms"] = (time.perf_counter() - start_close) * 1000
        if args.output == "json":
            print(json.dumps(result, ensure_ascii=True))
        elif args.run:
            for item in result["completions"]:
                print(item["text"])
        else:
            print("PASS: resident prompt preflight; no GPU workers started")
        return 0
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
