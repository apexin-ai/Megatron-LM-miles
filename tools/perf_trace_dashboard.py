#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Analysis service for ``--perf-trace-dir`` output: watch a run, draw the training pictures.

Reads the JSONL the tracer writes (one line per rank per step) and serves:

    /                       HTML dashboard (auto-refresh, step selector)
    /api/summary            per-step aggregates across ranks (JSON)
    /api/step/<n>           per-rank totals for one step (JSON)
    /png/timeline?step=N    one row per rank: fwd/bwd layers + MoE A2A + data fetch
    /png/trends             step time and its parts over training
    /png/skew?step=N        per-rank total time, stragglers first
    /png/heatmap?step=N&metric=fwd|bwd    rank x layer heatmap
    /png/layers?rank=R&metric=fwd|bwd     per-layer distribution over the sampled steps

Deliberately dependency-light: stdlib ``http.server`` plus matplotlib. It only reads files, so
it can sit next to a running job and never touches the training process.

Usage:
    perf_trace_dashboard.py --dir /cpfs/.../perf --port 8080
    perf_trace_dashboard.py --dir <dir> --once --out report.html     # static report
"""

import argparse
import base64
import html
import io
import json
import os
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _load_report_module():
    """Load ``report.py`` by path so this tool needs neither torch nor an installed megatron.

    Importing ``megatron.core.perf_trace`` would run ``megatron/core/__init__.py``, which imports
    torch; a plotting service that only reads JSONL should not drag the whole training stack in.
    """
    import importlib.util

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, 'megatron', 'core', 'perf_trace', 'report.py')
    spec = importlib.util.spec_from_file_location('perf_trace_report', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_report = _load_report_module()
TraceStore = _report.TraceStore
draw_layer_distribution = _report.draw_layer_distribution
draw_layer_heatmap = _report.draw_layer_heatmap
draw_rank_skew = _report.draw_rank_skew
draw_step_timeline = _report.draw_step_timeline
draw_step_trends = _report.draw_step_trends
draw_layer_start_heatmap = _report.draw_layer_start_heatmap
draw_overlap = _report.draw_overlap

STORE: TraceStore = None  # set in main()
REFRESH_SECONDS = 5.0


def _png_response(png: bytes) -> bytes:
    return png


def _chart(store: TraceStore, kind: str, q: dict) -> bytes:
    """Dispatch one chart request to the drawing layer."""
    if kind == 'trends':
        return draw_step_trends(store)
    step = int(q.get('step', [store.latest_step() or 0])[0])
    if kind == 'timeline':
        return draw_step_timeline(store, step)
    if kind == 'skew':
        return draw_rank_skew(store, step)
    if kind == 'heatmap':
        metric = q.get('metric', ['fwd'])[0]
        return draw_layer_heatmap(store, step, metric)
    if kind == 'starts':
        metric = q.get('metric', ['fwd_start'])[0]
        return draw_layer_start_heatmap(store, step, metric)
    if kind == 'overlap':
        rank = int(q.get('rank', ['0'])[0])
        return draw_overlap(store, step, rank)
    if kind == 'layers':
        rank = int(q.get('rank', ['0'])[0])
        metric = q.get('metric', ['fwd'])[0]
        return draw_layer_distribution(store, store.steps(), rank, metric)
    raise KeyError(kind)


PAGE_CSS = """
 body { font-family: system-ui, -apple-system, sans-serif; margin: 18px; background: #fafafa;
        color: #222; }
 h1 { font-size: 19px; margin-bottom: 4px; }
 h2 { font-size: 15px; margin: 26px 0 8px; }
 .meta { color: #555; font-size: 13px; }
 table { border-collapse: collapse; font-size: 12px; background: #fff; }
 th, td { border: 1px solid #e3e3e3; padding: 3px 7px; text-align: right; }
 th { background: #f2f2f2; font-weight: 600; }
 td.l, th.l { text-align: left; }
 img { max-width: 100%; border: 1px solid #ddd; background: #fff; }
 form { margin: 12px 0; font-size: 14px; }
 input, select, button { font-size: 13px; padding: 2px 4px; }
 .warn { color: #c33; font-weight: 600; }
 .cols { display: flex; gap: 22px; flex-wrap: wrap; align-items: flex-start; }
"""


def _table(headers, rows, left_cols=1):
    """Minimal HTML table; the first ``left_cols`` columns are left-aligned."""
    out = ['<table>', '<tr>']
    for i, h in enumerate(headers):
        cls = ' class="l"' if i < left_cols else ''
        out.append(f'<th{cls}>{html.escape(str(h))}</th>')
    out.append('</tr>')
    for row in rows:
        out.append('<tr>')
        for i, cell in enumerate(row):
            cls = ' class="l"' if i < left_cols else ''
            out.append(f'<td{cls}>{html.escape(str(cell))}</td>')
        out.append('</tr>')
    out.append('</table>')
    return ''.join(out)


def render_index(store, step, rank, metric, refresh_seconds=15):
    """Build the dashboard page: numbers first, then the charts for the chosen step."""
    rows = store.summary()
    steps = [r['step'] for r in rows]

    step_rows = []
    for r in rows[-15:]:
        med = r['total']['median']
        ratio = (r['total']['max'] / med) if med else 0.0
        verdict = (
            'level'
            if ratio < 1.05
            else 'slight' if ratio < 1.2 else ('notable' if ratio < 1.5 else 'SEVERE')
        )
        cls = ' class="warn"' if verdict == 'SEVERE' else ''
        step_rows.append(
            [
                r['step'],
                r['ranks'],
                f"{med:.2f}",
                f"{r['total']['min']:.2f}",
                f"{r['total']['max']:.2f}",
                f"{ratio:.2f}",
                verdict,
            ]
        )
        step_rows[-1][-1] = verdict
    step_table = _table(
        ['step', 'ranks', 'median s', 'min s', 'max s', 'max/med', 'verdict'], step_rows
    )

    rank_rows = [
        [rk, f'{tot:.2f}', f'{fwd:.2f}', f'{bwd:.2f}', f'{a2a:.2f}']
        for rk, tot, fwd, bwd, a2a in store.rank_totals(step)[:10]
    ]
    rank_table = _table(['rank', 'total s', 'fwd s', 'bwd s', 'a2a s'], rank_rows)

    names, ranks, matrix = store.layer_matrix(step, metric)
    import statistics

    layer_rows = []
    for j, name in enumerate(names):
        col = [row[j] for row in matrix if row[j] == row[j]]
        if col:
            layer_rows.append([name, f'{statistics.median(col):.4f}', f'{max(col):.4f}'])
    layer_rows.sort(key=lambda r: -float(r[1]))
    layer_table = _table([f'layer ({metric})', 'median s', 'max s'], layer_rows[:10])

    a2a_rows = [
        [n, f'{v:.4f}']
        for n, v in sorted(store.a2a_by_layer(step).items(), key=lambda kv: -kv[1])[:10]
    ]
    a2a_table = _table(['layer', 'a2a s (mean/rank)'], a2a_rows) if a2a_rows else '<i>none</i>'

    cpu_rows = []
    if rows:
        cpu = next((r for r in rows if r['step'] == step), None)
        if cpu and cpu.get('cpu'):
            cpu_rows = [
                [k, f"{v['median']:.2f}"]
                for k, v in sorted(cpu['cpu'].items(), key=lambda kv: -kv[1]['median'])
            ]
    cpu_table = _table(['phase (CPU timer)', 'median s'], cpu_rows) if cpu_rows else '<i>none</i>'

    step_options = ''.join(
        f'<option value="{s}"{" selected" if s == step else ""}>{s}</option>' for s in steps[-200:]
    )
    sel = {v: ' selected' if int(refresh_seconds) == v else '' for v in (0, 5, 15, 60)}
    metric_options = ''.join(
        f'<option value="{m}"{" selected" if m == metric else ""}>{m}</option>'
        for m in ('fwd', 'bwd')
    )
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>training perf</title><style>{PAGE_CSS}</style></head><body>
<h1>training performance</h1>
<div class="meta">dir: {html.escape(store.dirpath)} &nbsp;|&nbsp; steps: {len(steps)}
 &nbsp;|&nbsp; latest: {steps[-1] if steps else '-'}</div>
<form method="get" action="/">
  step <select name="step">{step_options}</select>
  rank <input name="rank" value="{rank}" size="5">
  metric <select name="metric">{metric_options}</select>
  refresh <select name="refresh">
    <option value="0"{sel[0]}>off</option>
    <option value="5"{sel[5]}>5 s</option>
    <option value="15"{sel[15]}>15 s</option>
    <option value="60"{sel[60]}>60 s</option>
  </select>
  <button type="submit">apply</button>
</form>
<h2>step time (last steps) &mdash; max/median is the imbalance; SEVERE means one rank is far behind</h2>
{step_table}
<h2>whole training</h2>
<img src="/png/trends?t={os.getpid()}" width="1150">
<h2>step {step}: pipeline per rank</h2>
<img src="/png/timeline?step={step}&t={os.getpid()}" width="1250">
<div class="cols">
 <div><h2>slowest ranks</h2>{rank_table}</div>
 <div><h2>all-to-all by layer</h2>{a2a_table}</div>
 <div><h2>phase timers</h2>{cpu_table}</div>
</div>
<h2>when each rank started each layer &mdash; a bright row is a late rank, a bright cell is a late layer</h2>
<img src="/png/starts?step={step}&t={os.getpid()}" width="1250">
<h2>rank {rank}: compute vs comm, the gaps are the point</h2>
<img src="/png/overlap?step={step}&rank={rank}&t={os.getpid()}" width="1250">
<div class="cols">
 <div><h2>per-rank skew</h2><img src="/png/skew?step={step}&t={os.getpid()}" width="620"></div>
 <div><h2>rank x layer ({metric})</h2>
   <img src="/png/heatmap?step={step}&metric={metric}&t={os.getpid()}" width="620"></div>
</div>
<h2>most expensive layers ({metric})</h2>{layer_table}
<h2>per-layer distribution (rank {rank})</h2>
<img src="/png/layers?rank={rank}&metric={metric}&t={os.getpid()}" width="1150">
<script>
/* Reload only when there is something new: an unconditional timer makes the page flash and
   re-renders every chart server-side on each tick, which is pure waste on a finished run. */
var REFRESH = {refresh_seconds};
if (REFRESH > 0) {{
  var seen = null;
  setInterval(function () {{
    fetch('/api/latest').then(function (r) {{ return r.json(); }}).then(function (d) {{
      var key = d.latest_step + ':' + d.records;
      if (seen === null) {{ seen = key; return; }}
      if (key !== seen) {{ seen = key; location.reload(); }}
    }}).catch(function () {{}});
  }}, Math.max(3000, REFRESH * 1000));
}}
</script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter default logging
        sys.stderr.write('%s - %s\n' % (self.address_string(), fmt % args))

    def _send(self, body: bytes, ctype: str, code: int = 200):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - stdlib naming
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        q = urllib.parse.parse_qs(parsed.query)
        STORE.refresh()
        try:
            if path == '/':
                latest = STORE.latest_step()
                step = int(q.get('step', [latest or 0])[0])
                rank = int(q.get('rank', ['0'])[0])
                metric = q.get('metric', ['fwd'])[0]
                refresh = int(q.get('refresh', [str(REFRESH_SECONDS)])[0])
                body = render_index(STORE, step, rank, metric, refresh).encode()
                self._send(body, 'text/html; charset=utf-8')
            elif path == '/api/latest':
                steps = STORE.steps()
                self._send(
                    json.dumps(
                        {
                            'latest_step': steps[-1] if steps else None,
                            'records': sum(len(STORE.ranks(s)) for s in steps[-5:]),
                        }
                    ).encode(),
                    'application/json',
                )
            elif path == '/api/summary':
                self._send(json.dumps(STORE.summary()).encode(), 'application/json')
            elif path.startswith('/api/step/') and path.endswith('/layers'):
                step = int(path.split('/')[3])
                metric = q.get('metric', ['fwd'])[0]
                names, ranks, matrix = STORE.layer_matrix(step, metric)
                import statistics

                layers = []
                for j, name in enumerate(names):
                    col = [row[j] for row in matrix if row[j] == row[j]]
                    if col:
                        layers.append(
                            {'layer': name, 'median_s': statistics.median(col), 'max_s': max(col)}
                        )
                self._send(
                    json.dumps({'step': step, 'metric': metric, 'layers': layers}).encode(),
                    'application/json',
                )
            elif path.startswith('/api/step/'):
                step = int(path.rsplit('/', 1)[-1])
                rows = [
                    {'rank': r, 'total_s': t, 'fwd_s': f, 'bwd_s': b, 'a2a_s': a}
                    for r, t, f, b, a in STORE.rank_totals(step)
                ]
                self._send(json.dumps({'step': step, 'ranks': rows}).encode(), 'application/json')
            elif path.startswith('/png/'):
                kind = path[len('/png/') :]
                png = _chart(STORE, kind, q)
                self._send(_png_response(png), 'image/png')
            else:
                self._send(b'not found', 'text/plain', 404)
        except Exception as exc:  # a bad step must not kill the server
            self._send(f'error: {exc}'.encode(), 'text/plain', 400)


def build_static_report(store: TraceStore, out_path: str) -> None:
    """Render every chart once and write a self-contained HTML file."""
    store.refresh()
    latest = store.latest_step()
    if latest is None:
        raise SystemExit('no perf records found')
    charts = {
        'trends': draw_step_trends(store),
        'timeline': draw_step_timeline(store, latest),
        'skew': draw_rank_skew(store, latest),
        'heatmap_fwd': draw_layer_heatmap(store, latest, 'fwd'),
        'heatmap_bwd': draw_layer_heatmap(store, latest, 'bwd'),
        'layers': draw_layer_distribution(store, store.steps(), 0, 'fwd'),
    }
    parts = [
        '<!doctype html><meta charset="utf-8"><title>training perf report</title>',
        '<style>body{font-family:system-ui,sans-serif;margin:20px;background:#fafafa}'
        'img{max-width:100%;border:1px solid #ddd;background:#fff;margin:8px 0}'
        'h2{font-size:16px;margin-top:24px}.meta{color:#555;font-size:13px}</style>',
        f'<h1>training perf report</h1>',
        f'<div class="meta">dir: {html.escape(store.dirpath)} &nbsp;|&nbsp; step: {latest} '
        f'&nbsp;|&nbsp; steps recorded: {len(store.steps())}</div>',
    ]
    for title, png in charts.items():
        b64 = base64.b64encode(png).decode()
        parts.append(f'<h2>{html.escape(title)}</h2><img src="data:image/png;base64,{b64}">')
    with open(out_path, 'w') as fh:
        fh.write('\n'.join(parts))
    print(f'wrote {out_path}  ({os.path.getsize(out_path) / 1e6:.1f} MB)')


def main():
    global STORE
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--dir', required=True, help='the --perf-trace-dir used by the run')
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--max-steps', type=int, default=4096)
    ap.add_argument(
        '--refresh-seconds',
        type=int,
        default=15,
        help='page auto-refresh interval; 0 disables it. The page only reloads when '
        'the data actually changed, so a finished run sits still.',
    )
    ap.add_argument('--once', action='store_true', help='render a static report and exit')
    ap.add_argument('--out', default='perf_report.html')
    args = ap.parse_args()

    global REFRESH_SECONDS
    REFRESH_SECONDS = args.refresh_seconds
    STORE = TraceStore(args.dir, max_steps=args.max_steps)
    added = STORE.refresh()
    print(f'loaded {added} record(s) from {args.dir}; steps={len(STORE.steps())}')

    if args.once:
        build_static_report(STORE, args.out)
        return

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'serving on http://{args.host}:{args.port}/  (dir={args.dir})')
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print('\nstopped')


if __name__ == '__main__':
    main()
