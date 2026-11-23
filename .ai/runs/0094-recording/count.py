"""Numbers-only summary of a record_proxy.py recording (no text leaves this script)."""
import json, sys, statistics
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
hdr = recs[0]; rs = [r for r in recs if not r.get('header')]
chat = [r for r in rs if r.get('method') == 'POST' and r.get('path', '').endswith('/chat/completions')]
gen = sum((r.get('usage') or {}).get('completion_tokens') or 0 for r in chat)
pred_ms = sum((r.get('timings') or {}).get('predicted_ms') or 0 for r in chat)
pn = sum((r.get('timings') or {}).get('predicted_n') or 0 for r in chat)
dn = sum((r.get('timings') or {}).get('draft_n') or 0 for r in chat); da = sum((r.get('timings') or {}).get('draft_n_accepted') or 0 for r in chat)
canc = sum(1 for r in chat if r.get('client_cancelled')); failed = sum(1 for r in chat if r.get('status') != 200 or not r.get('usage'))
prompts = [(r.get('usage') or {}).get('prompt_tokens') for r in chat]
comp = [(i + 1, prompts[i - 1], prompts[i]) for i in range(1, len(prompts)) if prompts[i] and prompts[i - 1] and prompts[i] < 0.6 * prompts[i - 1]]
drops = [(i + 1, prompts[i - 1], prompts[i]) for i in range(1, len(prompts)) if prompts[i] and prompts[i - 1] and prompts[i] < prompts[i - 1]]
fin = {}
for r in chat: fin[r.get('finish_reason')] = fin.get(r.get('finish_reason'), 0) + 1
ch = {k: sum((r.get('chunks') or {}).get(k, 0) for r in chat) for k in ('reasoning', 'content', 'tool_calls')}
gaps = [chat[i]['t_start'] - chat[i - 1]['t_end'] for i in range(1, len(chat))]
cached = sum((r.get('usage') or {}).get('prompt_tokens_details', {}).get('cached_tokens') or 0 for r in chat)
prompt_n = sum((r.get('timings') or {}).get('prompt_n') or 0 for r in chat); prompt_ms = sum((r.get('timings') or {}).get('prompt_ms') or 0 for r in chat)
dom = {}
for r in chat:
    c = r.get('chunks') or {}; k = max(('reasoning', 'content', 'tool_calls'), key=lambda x: c.get(x, 0)); dom[k] = dom.get(k, 0) + 1
out = dict(records=len(rs), chat_requests=len(chat), cancelled=canc, failed=failed, finish=fin,
           generated_tokens=gen, predicted_n=pn, decode_s=round(pred_ms / 1000, 1), engine_decode_tps=round(pn / (pred_ms / 1000), 2),
           draft_acceptance_pct=round(100 * da / dn, 2), chunks=ch, dominant_phase_requests=dom,
           prompt_tokens_min=min(p for p in prompts if p), prompt_tokens_max=max(p for p in prompts if p), prompt_tokens_median=statistics.median([p for p in prompts if p]), no_usage=[i+1 for i,p in enumerate(prompts) if not p],
           prompt_evaluated_tokens=prompt_n, prompt_s=round(prompt_ms / 1000, 1), cached_tokens_total=cached,
           compactions_lt60pct=comp, prompt_shrinks=drops,
           session_wall_s=round(chat[-1]['t_end'] - chat[0]['t_start'], 1), pauses_s=round(sum(g for g in gaps if g > 0), 1),
           pause_max_s=round(max(gaps), 1), busy_s=round(sum(r['t_end'] - r['t_start'] for r in chat), 1))
print(json.dumps(out, indent=1))
