#!/usr/bin/env python3
"""Focused resident 256K lifecycle qualification. No GPU without --run."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from threading import Event


def require(ok, message):
    if not ok:
        raise ValueError(message)


def token_digest(ids):
    return hashlib.sha256(','.join(map(str, ids)).encode('ascii')).hexdigest()


def exercise(engine, short, long, budget=4):
    """One worker set; two full long requests, two cancellation cases and reset controls.

    A cancelled response intentionally contains no returned tokens even when a
    prediction was already streamed. Validate committed positions separately.
    """
    require(budget == 4 and len(short) >= 2 and len(long) > len(short), 'invalid test shape')
    require(engine.stream_tokens and engine.source.get('native_prefill_chunk'), 'streamed native prefill required')
    require(engine.max_context == 262144 and engine.source.get('long_context') is True, '256K mode required')
    require(0 <= engine.max_context - len(long) - budget <= 16, 'near-capacity prompt required')
    records = []

    def request(label, ids, cancel_at=None):
        cancel = Event()
        emitted = []
        fired = False
        for i, token in enumerate(engine.generate(ids, budget, {'temperature': 0}, cancel)):
            if token is not None and token not in engine.stop_ids:
                emitted.append(token)
            if not fired and ((cancel_at == 'prefill' and i == 0 and token is None) or
                              (cancel_at == 'prediction' and token is not None and token not in engine.stop_ids)):
                cancel.set()
                fired = True
        result = engine.last_completion
        metrics = engine.last_metrics
        require(result and metrics and engine.alive(), 'missing completion or resident session lost')
        require(result['request_id'] == len(records) + 1 and result['prompt_tokens'] == len(ids), 'request identity mismatch')
        expected_pos = len(ids)-1 if cancel_at == 'prefill' else len(ids) if cancel_at else len(ids)+budget-1
        require(metrics['committed_prompt_tokens'] == min(expected_pos, len(ids)) and
                metrics['decode_feedback_tokens'] == max(0, expected_pos-len(ids)), 'committed positions mismatch')
        if cancel_at:
            require(fired and result['finish_reason'] == 'cancelled' and result['token_ids'] == [], 'cancellation not acknowledged')
            require(len(emitted) == (cancel_at == 'prediction'), 'unexpected cancelled streamed predictions')
        else:
            require(result['finish_reason'] == 'length' and len(emitted) == budget and emitted == result['token_ids'], 'expected full deterministic four-token completion')
        records.append(dict(case=label, request=result['request_id'], prompt_tokens=len(ids),
                            finish=result['finish_reason'], committed_position=expected_pos,
                            streamed_tokens=len(emitted), output_sha256=token_digest(emitted)))
        return emitted

    short_a = request('short_baseline', short)
    long_a = request('long_a', long)
    request('cancel_after_long_prefill', long, 'prefill')
    require(request('short_after_prefill_cancel', short) == short_a, 'short reset after prefill cancellation differs')
    request('cancel_after_long_prediction', long, 'prediction')
    require(request('long_b', long) == long_a, 'repeated long request differs')
    require(request('short_final', short) == short_a, 'final short reset differs')
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--server-config', type=Path, required=True)
    p.add_argument('--prompt-file', type=Path, required=True)
    p.add_argument('--run', action='store_true')
    a = p.parse_args()
    from serve.heterogeneous import prepare_server, ResidentEngine
    from serve.heterogeneous_long_context import LONG_PROMPT_BYTES
    from strata_cli.heterogeneous import encode_prompt
    os.umask(0o077)
    cfg = json.loads(a.server_config.read_text(encoding='utf-8'))
    args, argv, source, tok, stops, _ = prepare_server(cfg)
    require(args.long_context and args.max_context == 262144 and args.kv == 'int8' and args.stream_tokens and
            not args.restart_clean and args.max_new_tokens >= 4, 'requires 256K INT8 streamed resident config without restart')
    with a.prompt_file.open('rb') as stream:
        raw = stream.read(LONG_PROMPT_BYTES + 1)
    require(len(raw) <= LONG_PROMPT_BYTES, 'prompt file exceeds long-context bound')
    long = encode_prompt(tok, source, raw.decode('utf-8'), False, 4, args.prompt_chunk)
    short = encode_prompt(tok, source, ' hello' * 7, False, 4, args.prompt_chunk)
    require(0 <= 262144-len(long)-4 <= 16 and 2 <= len(short) < len(long), 'expected near-capacity and short prompts')
    identity = dict(context=262144, plan=args.plan, prompt_tokens=len(long), prompt_sha256=token_digest(long))
    if not a.run:
        print(json.dumps(dict(status='PREFLIGHT_ONLY', **identity)))
        return
    # No automatic restart and no signal-based cleanup. The caller must monitor
    # device memory independently and retain private stderr for native traces.
    os.environ['STRATA_HETERO_WINDOW_TRACE'] = '1'
    engine = ResidentEngine(argv, source, stops, args.max_new_tokens, args.prompt_chunk, args.timeout_ms, RuntimeError)
    pid = engine.proc.pid
    try:
        cases = exercise(engine, short, long)
    finally:
        engine.close()
    try:
        os.killpg(pid, 0)  # existence probe only; never send a signal
    except ProcessLookupError:
        pass
    else:
        raise RuntimeError('previous worker process group still exists; operator inspection required')
    print(json.dumps(dict(status='LIFECYCLE_CASES_PASS', coordinator_pid=pid,
                          cases=cases, **identity), indent=2))


if __name__ == '__main__':
    main()
