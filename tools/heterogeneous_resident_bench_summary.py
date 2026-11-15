#!/usr/bin/env python3
"""Compare four trace-off resident CLI JSON blocks in A/B/B/A order.

Only matched fixed-length completions are eligible. Keep raw run artifacts and
all failed attempts separately; this reader cannot attest hardware or trace-off.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

IDENTITY = ("pack_index_sha256", "gguf", "pack_dir", "vocab", "max_cells",
            "max_t", "context_limit", "prompt_chunk", "native_prefill_chunk",
            "group_native_hip", "stream_tokens", "lookup_draft", "workload_mode", "long_context")


def need(ok, msg):
    if not ok:
        raise ValueError(msg)


def digest(ids):
    return hashlib.sha256(','.join(map(str, ids)).encode('ascii')).hexdigest()


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def summary(values):
    return dict(median=statistics.median(values), minimum=min(values), maximum=max(values))


def compare(blocks, axis, warmup=1):
    need(len(blocks) == 4 and axis in ('kv', 'topology') and warmup == 1, 'expected A/B/B/A and one warmup per block')
    need(all(b.get('status') == 'complete' for b in blocks), 'incomplete resident block')
    sources = [b['source'] for b in blocks]
    first = sources[0]
    for source in sources:
        need(all(k in source and source[k] == first[k] for k in IDENTITY), 'unmatched source or runtime settings')
        need(source['stream_tokens'] is True and source['group_native_hip'] is True and
             source['lookup_draft'] is False and source['restart_clean'] is False,
             'requires streamed grouped native resident benchmark without lookup or restart')
        need(source['kv'] in ('int8', 'f16') and type(source['plan_fingerprint']) is int,
             'missing KV or plan identity')
    key = 'kv' if axis == 'kv' else 'plan_fingerprint'
    fixed = 'plan_fingerprint' if axis == 'kv' else 'kv'
    labels = [s[key] for s in sources]
    need(labels[0] == labels[3] and labels[1] == labels[2] and labels[0] != labels[1] and
         all(s[fixed] == first[fixed] for s in sources), 'wrong A/B/B/A arm order or changed fixed dimension')
    if axis == 'kv':
        need(first['long_context'] is False, 'long-context mode is INT8-only; use a shared ordinary capacity')
    sizes = [len(b['completions']) for b in blocks]
    need(len(set(sizes)) == 1 and 4 <= sizes[0] <= 20, 'equal blocks of 1 warmup + at least 3 measured requests required')
    prompt = blocks[0]['completions'][0]['prompt_token_ids']
    budget = len(blocks[0]['completions'][0]['token_ids'])
    need(isinstance(prompt, list) and bool(prompt) and budget >= 2 and len(prompt) + budget <= first['context_limit'],
         'invalid fixed prompt or generation budget')
    vocab = first['vocab']
    need(all(type(t) is int and 0 <= t < vocab for t in prompt), 'invalid input tokens')
    rows = []
    for i, b in enumerate(blocks):
        need(all(positive(b[k]) for k in ('startup_ms', 'shutdown_ms')), 'missing startup/teardown timing')
        samples = []
        hashes = []
        for j, record in enumerate(b['completions']):
            tokens = record['token_ids']
            need(record['request_id'] == j + 1 and record['prompt_tokens'] == len(prompt) and
                 record['prompt_token_ids'] == prompt and record['finish_reason'] == 'length' and
                 record['stop_token'] == -1 and isinstance(tokens, list) and len(tokens) == budget and
                 all(type(t) is int and 0 <= t < vocab for t in tokens), 'unmatched or incomplete fixed-length request')
            t = record['timing']
            need(t['prompt_tokens'] == t['committed_prompt_tokens'] == len(prompt) and
                 t['completion_tokens'] == budget and t['decode_feedback_tokens'] == budget-1,
                 'inconsistent committed token counts')
            p, d, e, f = (t[k] for k in ('prefill_ms', 'decode_ms', 'elapsed_ms', 'first_output_ms'))
            need(all(positive(x) for x in (p,d,e,f)) and p+d <= e + 1e-6 and f <= e + 1e-6,
                 'missing or invalid native host timings')
            if j >= warmup:
                hashes.append(digest(tokens))
                samples.append(dict(prefill_tokens_s=len(prompt)*1000/p,
                                    decode_feedback_tokens_s=(budget-1)*1000/d,
                                    ttft_ms=f, elapsed_ms=e))
        rows.append(dict(block=i+1, arm=labels[i], startup_ms=b['startup_ms'], shutdown_ms=b['shutdown_ms'],
                         output_hashes=hashes, samples=samples))
    arms = []
    for label in labels[:2]:
        chosen = [r for r in rows if r['arm'] == label]
        values = [s for r in chosen for s in r['samples']]
        arms.append(dict(arm=label, measured=len(values),
                         metrics={k: summary([s[k] for s in values]) for k in values[0]},
                         distinct_output_hashes=len({h for r in chosen for h in r['output_hashes']})))
    return dict(status='MATCHED_HOST_TIMING_ONLY', axis=axis, context=first['context_limit'],
                prompt_tokens=len(prompt), prompt_sha256=digest(prompt), completion_tokens=budget,
                warmups_per_block=warmup, arms=arms, blocks=rows,
                distinct_output_hashes=len({h for r in rows for h in r['output_hashes']}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--axis', required=True, choices=('kv', 'topology'))
    p.add_argument('--blocks', nargs=4, required=True, type=Path, metavar='JSON')
    a = p.parse_args()
    try:
        need(len({f.resolve() for f in a.blocks}) == 4, 'four distinct run artifacts required')
        print(json.dumps(compare([json.loads(f.read_text()) for f in a.blocks], a.axis), indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        p.error(str(exc))


if __name__ == '__main__':
    main()
