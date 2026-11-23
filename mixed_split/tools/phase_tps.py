#!/usr/bin/env python3
"""Decode speed per phase (thinking / answering / writing tool calls), sampled from Strata's GET /metrics, and plots.

    phase_tps.py collect [--url http://127.0.0.1:8081] [--duration 900] [--interval 0.25] [--out samples.jsonl]
    phase_tps.py table samples.jsonl
    phase_tps.py plot samples.jsonl [samples2.jsonl ...] [--out phase_tps.html] [--window 30]

`collect` is the Human User's benchmark script, with its sampling logic unchanged: every --interval it reads
/metrics; between two consecutive samples in the same phase, both "generating", tokens and elapsed time rising and no
request finished in between, it adds the new tokens and the wall time between the samples to that phase.  It prints
the progress line every 5 s and, at the end (--duration, or Ctrl-C; 0 = until Ctrl-C), the "Final Strata Phase
Statistics" table.  Added: every counted sample is written to --out (t since start, phase, tokens, seconds), so the
table can be printed again and the plots drawn.  replay_session.py runs the same collector beside each replay.

`plot` writes one self-contained HTML page (no network, no libraries): the running average tok/s of each phase and
overall (a) over the session's wall time and (b) over the tokens generated so far, with a hover crosshair, direct
labels, a "last N seconds" view beside the running average, and the numbers as a table.  Several sample files: one
page, one section per file.
"""
import argparse, html, json, sys, threading, time, urllib.request
from collections import defaultdict
from pathlib import Path

PHASES = ("thinking", "answering", "writing tool calls")


def classify(phase):
    if phase in ("thinking", "answering"):
        return phase
    if phase and phase.startswith("writing a tool call"):
        return "writing tool calls"
    return None


def format_time(seconds):
    minutes, seconds = divmod(max(0, int(seconds)), 60)
    return f"{minutes:02d}:{seconds:02d}"


class Stats:
    def __init__(self):
        self.s = defaultdict(lambda: [0, 0.0, 0])

    def add(self, phase, dn, dt):
        self.s[phase][0] += dn
        self.s[phase][1] += dt
        self.s[phase][2] += 1

    def get(self, phase):
        if phase == "overall":
            return [sum(self.s[p][i] for p in PHASES) for i in range(3)]
        return self.s[phase]

    def avg_tps(self, phase):
        tokens, seconds, _ = self.get(phase)
        return f"{tokens / seconds:.2f}" if seconds > 0 else "N/A"

    def progress(self, remaining):
        left = f"[Remaining {format_time(remaining)}] " if remaining is not None else ""
        return (f"{left}Overall: {self.avg_tps('overall')} TPS | Thinking: {self.avg_tps('thinking')} TPS | "
                f"Answering: {self.avg_tps('answering')} TPS | Tool calls: {self.avg_tps('writing tool calls')} TPS")

    def summary(self):
        out = ["", "Final Strata Phase Statistics",
               f'{"Phase":<22} {"Avg TPS":>10} {"Tokens":>12} {"Seconds":>10} {"Samples":>9}', "-" * 70]
        for phase in (*PHASES, "overall"):
            tokens, seconds, samples = self.get(phase)
            tps = f"{tokens / seconds:.2f}" if seconds > 0 else "N/A"
            label = "OVERALL (all phases)" if phase == "overall" else phase
            if phase == "overall":
                out.append("-" * 70)
            out.append(f"{label:<22} {tps:>10} {tokens:>12} {seconds:>10.1f} {samples:>9}")
        return "\n".join(out)

    def as_dict(self):
        d = {}
        for phase in (*PHASES, "overall"):
            tokens, seconds, samples = self.get(phase)
            d[phase] = {"tokens": tokens, "seconds": round(seconds, 3), "samples": samples,
                        "tps": round(tokens / seconds, 2) if seconds > 0 else None}
        return d


def collect(url, out_path=None, duration=None, interval=0.25, report_every=5.0, stop=None, quiet=False):
    """The Human User's sampling loop.  Ends after `duration` s (None: never), when `stop` is set, or on Ctrl-C."""
    stats, previous = Stats(), None
    start = time.monotonic()
    end = start + duration if duration else None
    next_report = start + report_every
    out = open(out_path, "w") if out_path else None
    try:
        while (end is None or time.monotonic() < end) and not (stop is not None and stop.is_set()):
            try:
                with urllib.request.urlopen(url, timeout=3) as response:
                    data = json.load(response)
                now = time.monotonic()
                live = data.get("live", {})
                phase = classify(live.get("phase"))
                tokens = live.get("generated")
                elapsed = live.get("elapsed_s")
                requests = data.get("requests") or []
                last_done = requests[0].get("id") if requests else None
                current = (now, phase, tokens, elapsed, last_done, live.get("state"))
                if previous is not None:
                    p_now, p_phase, p_tokens, p_elapsed, p_done, p_state = previous
                    if (phase is not None and phase == p_phase and live.get("state") == "generating"
                            and p_state == "generating" and tokens is not None and p_tokens is not None
                            and elapsed is not None and p_elapsed is not None and elapsed >= p_elapsed
                            and tokens >= p_tokens and last_done == p_done):
                        dt = now - p_now
                        dn = tokens - p_tokens
                        if dt > 0 and dn > 0:
                            stats.add(phase, dn, dt)
                            if out:
                                out.write(json.dumps({"t": round(now - start, 4), "phase": phase, "tokens": dn,
                                                      "seconds": round(dt, 5)}) + "\n")
                                out.flush()
                previous = current
            except Exception as e:  # noqa: BLE001 - as in the original: report, forget the previous sample
                if not quiet:
                    print(f"Sampling error: {e}", flush=True)
                previous = None
            now = time.monotonic()
            if not quiet and now >= next_report:
                print(stats.progress(end - now if end else None), flush=True)
                next_report = now + report_every
            time.sleep(interval if end is None else min(interval, max(0, end - time.monotonic())))
    except KeyboardInterrupt:
        if not quiet:
            print("\nBenchmark interrupted by user.", flush=True)
    finally:
        if out:
            out.close()
    return stats


def start_collector(url, out_path, interval=0.25):
    """collect() in a thread (replay_session.py): returns (thread, stop event, result holder)."""
    stop, holder = threading.Event(), {}
    th = threading.Thread(target=lambda: holder.update(stats=collect(url, out_path, None, interval, stop=stop, quiet=True)),
                          daemon=True)
    th.start()
    return th, stop, holder


def load_samples(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def stats_of(samples):
    s = Stats()
    for x in samples:
        s.add(x["phase"], x["tokens"], x["seconds"])
    return s


# ----------------------------------------------------------------------------------------------------------- plots
SERIES = (("overall", "overall"), ("thinking", "thinking"), ("answering", "answering"),
          ("writing tool calls", "tool calls"))


def curves(samples, window):
    """Per sample: t, accumulated tokens, and per series the running average and the last-`window`-seconds average
    (tokens over sample seconds, in the samples of that series within the window)."""
    cum = {k: [0, 0.0] for k, _ in SERIES}
    rows, acc, recent = [], 0, []
    for x in samples:
        acc += x["tokens"]
        for k in ("overall", x["phase"]):
            cum[k][0] += x["tokens"]
            cum[k][1] += x["seconds"]
        recent.append(x)
        while recent and recent[0]["t"] < x["t"] - window:
            recent.pop(0)
        row = {"t": x["t"], "acc": acc}
        for k, _ in SERIES:
            row["run_" + k] = round(cum[k][0] / cum[k][1], 2) if cum[k][1] > 0 else None
            sel = [r for r in recent if k == "overall" or r["phase"] == k]
            sec = sum(r["seconds"] for r in sel)
            row["win_" + k] = round(sum(r["tokens"] for r in sel) / sec, 2) if sec > 0 else None
        rows.append(row)
    return rows


PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Strata phase tok/s</title>
<style>
:root{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100}
@media (prefers-color-scheme:dark){:root{--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500}}
body{background:var(--surface);color:var(--ink);font:14px/1.45 system-ui,sans-serif;margin:24px;max-width:1100px}
h1{font-size:20px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 4px}h3{font-size:14px;margin:18px 0 2px;color:var(--ink2);font-weight:600}
.note{color:var(--ink2);margin:0 0 10px}.controls{display:flex;gap:6px;margin:10px 0}
.controls button{background:none;border:1px solid var(--grid);color:var(--ink2);border-radius:6px;padding:4px 10px;cursor:pointer;font:inherit}
.controls button[aria-pressed=true]{border-color:var(--ink2);color:var(--ink)}
.legend{display:flex;gap:16px;margin:4px 0 0;color:var(--ink2);font-size:13px}.legend span::before{content:"";display:inline-block;width:14px;height:2px;margin:0 6px 3px 0;background:var(--c)}
.chart{position:relative}svg{display:block;width:100%;height:auto;overflow:visible}
.axis text,.lab{fill:var(--muted);font-size:11px}.grid line{stroke:var(--grid);stroke-width:1}
.end{font-size:12px;fill:var(--ink2)}.line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.tip{position:absolute;pointer-events:none;background:var(--surface);border:1px solid var(--grid);border-radius:6px;padding:6px 8px;font-size:12px;color:var(--ink);display:none;white-space:nowrap;box-shadow:0 2px 8px rgba(0,0,0,.08)}
.tip i{display:inline-block;width:10px;height:2px;margin:0 6px 3px 0}.cross{stroke:var(--muted);stroke-width:1;stroke-dasharray:3 3}
table{border-collapse:collapse;margin:8px 0;font-variant-numeric:tabular-nums}td,th{padding:3px 12px;text-align:right;border-bottom:1px solid var(--grid)}th{color:var(--ink2);font-weight:600}td:first-child,th:first-child{text-align:left}
details{margin:6px 0}summary{cursor:pointer;color:var(--ink2)}
</style></head><body>
<h1>Strata decode speed per phase</h1>
<p class="note">From GET /metrics, sampled every 0.25 s (phase_tps.py, the Human User's benchmark logic). Running average: tokens so far over seconds so far, per phase. Last %%WIN%% s: the same over the most recent %%WIN%% s of that phase's samples.</p>
<div class="controls" role="group" aria-label="View"><button data-v="run" aria-pressed="true">Running average</button><button data-v="win" aria-pressed="false">Last %%WIN%% s</button></div>
%%SECTIONS%%
<script>
const DATA=%%DATA%%;
const S=[["overall","overall","--s1"],["thinking","thinking","--s2"],["answering","answering","--s3"],["writing tool calls","tool calls","--s4"]];
let view="run";
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
function nice(max){const t=(max||1)/5,e=Math.pow(10,Math.floor(Math.log10(t)));for(const m of[1,2,2.5,5,10])if(m*e>=t)return m*e;return e*10}
function draw(el,rows,xkey,xlabel,xfmt){
  const W=1000,H=320,L=48,R=110,T=10,B=34,w=W-L-R,h=H-T-B;
  const xs=rows.map(r=>r[xkey]),xmax=Math.max(...xs,1);
  let ymax=0;rows.forEach(r=>S.forEach(([k])=>{const v=r[view+"_"+k];if(v!=null&&v>ymax)ymax=v}));
  const ys=nice(ymax),ytop=Math.ceil(ymax/ys)*ys||1,xst=nice(xmax),X=v=>L+v/xmax*w,Y=v=>T+h-v/ytop*h;
  let g='<g class="grid">';for(let v=0;v<=ytop+1e-9;v+=ys)g+=`<line x1="${L}" x2="${L+w}" y1="${Y(v)}" y2="${Y(v)}"/>`;g+='</g><g class="axis">';
  for(let v=0;v<=ytop+1e-9;v+=ys)g+=`<text x="${L-6}" y="${Y(v)+4}" text-anchor="end">${v}</text>`;
  for(let v=0;v<=xmax+1e-9;v+=xst)g+=`<text x="${X(v)}" y="${T+h+16}" text-anchor="middle">${xfmt(v)}</text>`;
  g+=`<text x="${L+w/2}" y="${H-2}" text-anchor="middle">${xlabel}</text><text x="12" y="${T+h/2}" transform="rotate(-90 12 ${T+h/2})" text-anchor="middle">tok/s</text></g>`;
  const ends=[];
  S.forEach(([k,lab,c])=>{let d="",last=null;rows.forEach(r=>{const v=r[view+"_"+k];if(v==null)return;d+=(d?"L":"M")+X(r[xkey]).toFixed(1)+" "+Y(v).toFixed(1);last=v});
    if(d){g+=`<path class="line" d="${d}" stroke="var(${c})"/>`;ends.push([Y(last),`${lab} ${last.toFixed(1)}`])}});
  ends.sort((a,b)=>a[0]-b[0]);for(let i=1;i<ends.length;i++)if(ends[i][0]-ends[i-1][0]<14)ends[i][0]=ends[i-1][0]+14;
  ends.forEach(([y,t])=>g+=`<text class="end" x="${L+w+8}" y="${y+4}">${t}</text>`);
  g+=`<line class="cross" x1="0" x2="0" y1="${T}" y2="${T+h}" style="display:none"/><rect x="${L}" y="${T}" width="${w}" height="${h}" fill="transparent"/>`;
  el.querySelector("svg").innerHTML=g;
  const svg=el.querySelector("svg"),tip=el.querySelector(".tip"),cross=svg.querySelector(".cross"),hit=svg.querySelector("rect");
  hit.onmousemove=e=>{const b=svg.getBoundingClientRect(),px=(e.clientX-b.left)/b.width*W,xv=(px-L)/w*xmax;
    let i=0,best=1e18;rows.forEach((r,j)=>{const d=Math.abs(r[xkey]-xv);if(d<best){best=d;i=j}});const r=rows[i];
    cross.setAttribute("x1",X(r[xkey]));cross.setAttribute("x2",X(r[xkey]));cross.style.display="";
    tip.innerHTML=`<b>${xfmt(r[xkey])}</b> ${xlabel}<br>`+S.map(([k,lab,c])=>r[view+"_"+k]==null?"":`<i style="background:${css(c)}"></i>${lab}: ${r[view+"_"+k].toFixed(1)} tok/s`).filter(Boolean).join("<br>");
    tip.style.display="block";const tx=(X(r[xkey])/W)*b.width;tip.style.left=Math.min(tx+12,b.width-tip.offsetWidth)+"px";tip.style.top="8px"};
  hit.onmouseleave=()=>{tip.style.display="none";cross.style.display="none"};
}
const tfmt=v=>{const m=Math.floor(v/60),s=Math.round(v%60);return m+":"+String(s).padStart(2,"0")},kfmt=v=>v>=1000?(v/1000).toFixed(v%1000?1:0)+"k":String(v);
function render(){document.querySelectorAll("[data-run]").forEach(sec=>{const rows=DATA[+sec.dataset.run].rows;
  draw(sec.querySelector(".c-time"),rows,"t","time since start (min:s)",tfmt);draw(sec.querySelector(".c-tok"),rows,"acc","tokens generated so far",kfmt)})}
document.querySelectorAll(".controls button").forEach(b=>b.onclick=()=>{view=b.dataset.v;document.querySelectorAll(".controls button").forEach(x=>x.setAttribute("aria-pressed",x===b));render()});
render();
</script></body></html>
"""


def section(i, name, stats, rows):
    legend = "".join(f'<span style="--c:var(--s{j + 1})">{lab}</span>' for j, (_, lab) in enumerate(SERIES))
    table = "".join(f"<tr><td>{'OVERALL (all phases)' if k == 'overall' else k}</td><td>{v['tps'] if v['tps'] is not None else 'N/A'}</td>"
                    f"<td>{v['tokens']}</td><td>{v['seconds']:.1f}</td><td>{v['samples']}</td></tr>"
                    for k, v in ((k, stats.as_dict()[k]) for k in (*PHASES, "overall")))
    raw = "".join(f"<tr><td>{r['t']:.1f}</td><td>{r['acc']}</td>" + "".join(
        f"<td>{r['run_' + k] if r['run_' + k] is not None else ''}</td>" for k, _ in SERIES) + "</tr>"
        for r in rows[:: max(1, len(rows) // 400)])
    return f"""<section data-run="{i}"><h2>{html.escape(name)}</h2>
<table><tr><th>Phase</th><th>Avg TPS</th><th>Tokens</th><th>Seconds</th><th>Samples</th></tr>{table}</table>
<h3>tok/s over time</h3><div class="legend">{legend}</div><div class="chart c-time"><svg viewBox="0 0 1000 320" role="img" aria-label="tok/s over time"></svg><div class="tip"></div></div>
<h3>tok/s over tokens generated</h3><div class="legend">{legend}</div><div class="chart c-tok"><svg viewBox="0 0 1000 320" role="img" aria-label="tok/s over tokens generated"></svg><div class="tip"></div></div>
<details><summary>Table view (running averages, up to 400 rows)</summary><table><tr><th>t (s)</th><th>tokens so far</th>{''.join(f'<th>{lab}</th>' for _, lab in SERIES)}</tr>{raw}</table></details>
</section>"""


def write_plot(files, out, window):
    runs, secs = [], []
    for i, f in enumerate(files):
        samples = load_samples(f)
        rows = curves(samples, window)
        runs.append({"name": str(f), "rows": rows})
        secs.append(section(i, str(f), stats_of(samples), rows))
    page = (PAGE.replace("%%WIN%%", str(int(window))).replace("%%SECTIONS%%", "\n".join(secs))
            .replace("%%DATA%%", json.dumps(runs)))
    Path(out).write_text(page, encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--url", default="http://127.0.0.1:8081")
    c.add_argument("--duration", type=float, default=15 * 60, help="seconds; 0 = until Ctrl-C")
    c.add_argument("--interval", type=float, default=0.25)
    c.add_argument("--report-every", type=float, default=5)
    c.add_argument("--out", default="phase_samples.jsonl")
    t = sub.add_parser("table")
    t.add_argument("samples")
    p = sub.add_parser("plot")
    p.add_argument("samples", nargs="+")
    p.add_argument("--out", default="phase_tps.html")
    p.add_argument("--window", type=float, default=30, help="seconds of the 'last N seconds' view")
    a = ap.parse_args()
    if a.cmd == "collect":
        print("Collecting Strata phase TPS...")
        print(f"Duration: {format_time(a.duration) if a.duration else 'until Ctrl+C'}")
        print(f"Sampling interval: {a.interval} seconds")
        print(f"Progress interval: {a.report_every} seconds")
        print("Press Ctrl+C to stop and display results.\n", flush=True)
        st = collect(a.url.rstrip("/") + "/metrics", a.out, a.duration or None, a.interval, a.report_every)
        print(st.summary())
        print(f"\nsamples: {a.out}  (phase_tps.py plot {a.out})")
    elif a.cmd == "table":
        print(stats_of(load_samples(a.samples)).summary())
    else:
        print("wrote", write_plot(a.samples, a.out, a.window))
    return 0


if __name__ == "__main__":
    sys.exit(main())
