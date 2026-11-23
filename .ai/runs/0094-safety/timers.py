#!/usr/bin/env python3
"""Aggregate the engine's per-request timer lines (STRATA_DECODE_TIMING, STRATA_VERIFY_PROFILE, STRATA_SPLIT_TIMING)
of one replay into per-phase tables.  Numbers only.

    timers.py SERVER_LOG REPLAY_DIR [--json OUT] [--markdown OUT]

Requests are matched in order: the n-th "strata serve: prompt N tokens = ..." line is the replay's n-th request.
Lines printed before it (pipeline, classes, decode timing, GPU stages) and after it (split timing, front stage timing,
the conversation park) belong to the same request.  A request's phase is its dominant phase by stream chunks.
"""
import argparse, json, re, statistics
from collections import defaultdict
from pathlib import Path

F = r'([0-9.]+)'


def parse(log):
    reqs, pend = [], {}
    cur = None
    for line in open(log, errors='replace'):
        line = line.rstrip('\n')
        m = re.match(r'strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms .*?, (\d+) generated in (\d+) ms', line)
        if m:
            cur = dict(pend)
            pend = {}
            cur.update(prompt=int(m[1]), reused=int(m[2]), read=int(m[3]), prompt_ms=int(m[4]), gen=int(m[5]), gen_ms=int(m[6]))
            reqs.append(cur)
            continue
        m = re.match(r'strata pipeline: (\d+) windows in (\d+) ms \(' + F + r' ms/window\): (\d+) speculative, (\d+) on the path, (\d+) rolled back, (\d+) below the gate', line)
        if m:
            pend.update(pl_windows=int(m[1]), pl_ms=int(m[2]), pl_spec=int(m[4]), pl_on=int(m[5]), pl_rb=int(m[6]), pl_gate=int(m[7]))
            continue
        m = re.match(r'strata pipeline classes: fresh (\d+) windows ' + F + r' ms ' + F + r' tok \| speculative (\d+) windows ' + F + r' ms ' + F + r' tok \| forced-chain disagreements (\d+), chain late (\d+)', line)
        if m:
            pend.update(fresh_n=int(m[1]), fresh_ms=float(m[2]), fresh_tok=float(m[3]), spec_n=int(m[4]), spec_ms=float(m[5]),
                        spec_tok=float(m[6]), disagree=int(m[7]), chain_late=int(m[8]))
            continue
        m = re.match(r'strata pipeline: drafter on \S+: ' + F + r' ms of host time per chain launch', line)
        if m:
            pend['drafter_host_ms'] = float(m[1])
            continue
        m = re.match(r'strata decode timing: (\d+) windows, avg T ' + F + ', ' + F + r' tokens/window, ' + F + r' ms/window = verify ' + F
                     + r' \(GPU-reach wait ' + F + r' \+ per-layer host ' + F + r' .*?\+ stage ' + F + r'\) \+ commit/emit ' + F + r' \+ draft ' + F
                     + r'.*adapt wait ' + F + r' join ' + F, line)
        if m:
            pend.update(dt_windows=int(m[1]), dt_T=float(m[2]), dt_tpw=float(m[3]), dt_ms=float(m[4]), dt_verify=float(m[5]),
                        dt_reach=float(m[6]), dt_host=float(m[7]), dt_stage=float(m[8]), dt_commit=float(m[9]), dt_draft=float(m[10]),
                        dt_adapt=float(m[11]), dt_join=float(m[12]))
            continue
        m = re.match(r'strata decode GPU stages( , stage 1|, stage 1)? \(ms/window\): (.*)\| total ' + F + r' ms/window over (\d+) windows', line)
        if m:
            k = 's1' if m[1] else 's0'
            pend[k + '_gpu_ms'] = float(m[3])
            pend[k + '_gpu_n'] = int(m[4])
            h = re.search(r' head ' + F, m[2])
            if h:
                pend[k + '_head_ms'] = float(h[1])
            vh = re.findall(r'VRAM hits ' + F, m[2])
            pend[k + '_vramhits_ms'] = sum(float(x) for x in vh)
            continue
        if cur is None:
            continue
        m = re.match(r'strata serve: mixed split timing \(STRATA_SPLIT_TIMING\), per window since the last report: (\d+) windows; link run '
                     + F + r' ms, link commit ' + F + r' ms \(\d+\), first local stage.s wait for the front stage.s ready word ' + F
                     + r' ms \(\d+\); stage 0 GPU ' + F + r' ms \(\d+\), stage 1 GPU ' + F, line)
        if m:
            cur.update(ser_windows=int(m[1]), ser_link_run=float(m[2]), ser_link_commit=float(m[3]), ser_ready_wait=float(m[4]),
                       ser_s0=float(m[5]), ser_s1=float(m[6]))
            continue
        m = re.match(r'strata serve: mixed split timing \(STRATA_SPLIT_TIMING\), link calls since the last report: (.*)', line)
        if m:
            calls = {}
            for c in m[1].split(', '):
                cm = re.match(r'(.+?) (\d+) x ' + F + r' ms = ' + F + ' ms', c)
                if cm:
                    calls[cm[1]] = [int(cm[2]), float(cm[4])]
            cur['link_calls'] = calls
            continue
        m = re.match(r'strata serve: mixed split timing \(STRATA_SPLIT_TIMING\), pipelined windows since the last report: first local stage.s wait for the front stage.s ready word '
                     + F + r' ms per window \((\d+)\), longer than 0.1 ms in (\d+) of (\d+) read; wrong guesses: the front window had ended at the verdict in (\d+) \(' + F
                     + r' ms before it\), still ran in (\d+) \(' + F, line)
        if m:
            cur.update(pl_ready_wait=float(m[1]), pl_ready_n=int(m[2]), wg_ended=int(m[5]), wg_ended_ms=float(m[6]), wg_ran=int(m[7]), wg_ran_ms=float(m[8]))
            continue
        m = re.match(r'strata serve: front stage timing \(STRATA_SPLIT_TIMING\), pipelined windows since the last request: (\d+); GPU ' + F
                     + r' ms for the (\d+) kept, ' + F + r' ms for the (\d+) a wrong guess undid', line)
        if m:
            cur.update(fr_windows=int(m[1]), fr_kept_ms=float(m[2]), fr_kept=int(m[3]), fr_undone_ms=float(m[4]), fr_undone=int(m[5]))
            continue
        m = re.match(r'strata serve: front stage timing \(STRATA_SPLIT_TIMING\), per window (.*?): (\d+) windows; (.*)', line)
        if m:
            g = re.search(r'GPU ' + F, m[3])
            cur.update(fr_ser_windows=int(m[2]), fr_ser_gpu=float(g[1]) if g else None)
            continue
        m = re.match(r'strata serve: conversation cache: parked (\d+) tokens in ' + F + ' ms', line)
        if m:
            cur['park_ms'] = cur.get('park_ms', 0) + float(m[2])
            continue
        m = re.match(r'strata serve: the graphics driver paused the GPU for (\d+) ms', line)
        if m:
            cur['gpu_pause_ms'] = cur.get('gpu_pause_ms', 0) + int(m[1])
    return reqs


def wavg(rows, key, wkey):
    num = sum((r.get(key) or 0) * (r.get(wkey) or 0) for r in rows if r.get(key) is not None)
    den = sum((r.get(wkey) or 0) for r in rows if r.get(key) is not None)
    return round(num / den, 3) if den else None


def ssum(rows, key):
    return sum(r.get(key) or 0 for r in rows)


def table(rows):
    t = {}
    t['requests'] = len({r['request'] for r in rows})
    t['engine_calls'] = len(rows)
    t['generated'] = ssum(rows, 'gen')
    t['decode_s'] = round(ssum(rows, 'gen_ms') / 1000, 1)
    t['tok_s'] = round(t['generated'] / t['decode_s'], 1) if t['decode_s'] else None
    t['prompt_read'] = ssum(rows, 'read')
    t['prompt_s'] = round(ssum(rows, 'prompt_ms') / 1000, 1)
    t['dt_windows'] = ssum(rows, 'dt_windows')
    for k in ('dt_ms', 'dt_tpw', 'dt_T', 'dt_verify', 'dt_commit', 'dt_draft', 'dt_adapt', 'dt_reach', 'dt_host'):
        t[k] = wavg(rows, k, 'dt_windows')
    t['pl_windows'] = ssum(rows, 'pl_windows')
    for k in ('pl_spec', 'pl_on', 'pl_rb', 'pl_gate', 'fresh_n', 'spec_n', 'disagree', 'chain_late'):
        t[k] = ssum(rows, k)
    t['fresh_ms'] = wavg(rows, 'fresh_ms', 'fresh_n')
    t['fresh_tok'] = wavg(rows, 'fresh_tok', 'fresh_n')
    t['spec_ms'] = wavg(rows, 'spec_ms', 'spec_n')
    t['spec_tok'] = wavg(rows, 'spec_tok', 'spec_n')
    t['s0_gpu_ms'] = wavg(rows, 's0_gpu_ms', 's0_gpu_n')
    t['s1_gpu_ms'] = wavg(rows, 's1_gpu_ms', 's1_gpu_n')
    t['s1_head_ms'] = wavg(rows, 's1_head_ms', 's1_gpu_n')
    t['s0_vramhits_ms'] = wavg(rows, 's0_vramhits_ms', 's0_gpu_n')
    t['s1_vramhits_ms'] = wavg(rows, 's1_vramhits_ms', 's1_gpu_n')
    t['fr_kept_ms'] = wavg(rows, 'fr_kept_ms', 'fr_kept')
    t['fr_undone_ms'] = wavg(rows, 'fr_undone_ms', 'fr_undone')
    t['fr_windows'] = ssum(rows, 'fr_windows')
    t['pl_ready_wait'] = wavg(rows, 'pl_ready_wait', 'pl_ready_n')
    t['wg_ended'] = ssum(rows, 'wg_ended')
    t['wg_ran'] = ssum(rows, 'wg_ran')
    t['ser_windows'] = ssum(rows, 'ser_windows')
    for k in ('ser_ready_wait', 'ser_s0', 'ser_s1', 'ser_link_run', 'ser_link_commit'):
        t[k] = wavg(rows, k, 'ser_windows')
    calls = defaultdict(lambda: [0, 0.0])
    for r in rows:
        for k, (n, ms) in (r.get('link_calls') or {}).items():
            calls[k][0] += n
            calls[k][1] += ms
    t['link_calls'] = {k: [n, round(ms / n, 3) if n else 0, round(ms / 1000, 2)] for k, (n, ms) in calls.items()}
    t['park_s'] = round(ssum(rows, 'park_ms') / 1000, 1)
    t['gpu_pause_s'] = round(ssum(rows, 'gpu_pause_ms') / 1000, 1)
    if t['fresh_n'] and t['spec_n']:
        tot_ms = t['fresh_n'] * t['fresh_ms'] + t['spec_n'] * t['spec_ms']
        t['class_ms_check'] = round(tot_ms / 1000, 1)
        t['miss_cost_ms'] = round(t['fresh_ms'] - t['spec_ms'], 2)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log')
    ap.add_argument('replay')
    ap.add_argument('--json')
    a = ap.parse_args()
    reqs = parse(a.log)
    rows = [json.loads(l) for l in open(Path(a.replay) / 'requests.jsonl') if l.strip()]
    # the server can run one request as two engine calls (a continuation with a 19-token prompt read); such an extra
    # call is kept as its own row, with the phase of the request it continues
    j, prev = 0, None
    for q in reqs:
        if j < len(rows) and q['prompt'] == rows[j]['prompt_tokens']:
            ph = rows[j]['phases']
            q['phase'] = max(('reasoning', 'content', 'tool_calls'), key=lambda k: ph[k]['chunks'])
            q['request'] = j + 1
            j += 1
        else:
            assert prev is not None, q['prompt']
            q['phase'], q['request'], q['continuation'] = prev['phase'], prev['request'], True
        prev = q
    assert j == len(rows), (j, len(rows))
    out = {'all': table(reqs)}
    for p in ('reasoning', 'content', 'tool_calls'):
        out[p] = table([q for q in reqs if q['phase'] == p])
    js = json.dumps(out, indent=1)
    if a.json:
        open(a.json, 'w').write(js)
    print(js)


if __name__ == '__main__':
    main()
